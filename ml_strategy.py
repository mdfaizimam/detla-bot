# --- ml_strategy.py ---
# Complete Updated File
# FIX: Added 'strategy_lock' to prevent concurrent signal processing.
# FIX: Changed model target mapping to match train_model.py
# FIX: Added Stale MTF Data Filter to avoid trading on old candle data.
# FIX: Corrected TA key names in _prepare_features to match feature_engine
# FIX: Fixed critical typo (self.self_publish_signal) by publishing directly
# ✅ NEW: Use real OBV and ADX values from feature_engine payload

import asyncio
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import time 

from redis import asyncio as aioredis

from config import (
    ENRICHED_CHANNEL, 
    SIGNAL_CHANNEL, 
    TRADING_SYMBOLS, 
    PRIORITY_LIST,
    BASE_POSITION_SIZE,
    SIGNAL_CONFIDENCE,
    config,
    REDIS_URL
)
from risk_manager import RiskManager

log = logging.getLogger("ml_strategy")

# --- Constants ---
MODEL_DIR = "model"
MODEL_NAME = "signal_classifier.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# ✅ FIX: Target mapping must match train_model.py
# 0 = SHORT (-1), 1 = CHOP (0), 2 = LONG (1)
TARGET_MAP = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}

# ✅ NEW: Resolution map for stale data filter
# (Copied from feature_engine.py for isolation)
RESOLUTION_SECONDS = {
    "1m": 60, 
    "5m": 300, 
    "15m": 900, 
    "1h": 3600, 
    "4h": 14400, 
    "1d": 86400,
    "1w": 604800,
}


class MLForecastingStrategy:
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.model = self._load_model()
        self.config = config # Use derived config
        
        self.last_signal_ts = {symbol: 0 for symbol in TRADING_SYMBOLS}
        self.signal_cooldown = 15 # Cooldown in seconds per symbol
        
        # ✅ FIX: Lock to prevent race conditions on new signals
        self._strategy_lock = asyncio.Lock()

    def _load_model(self):
        """Loads the pre-trained XGBoost model."""
        if not os.path.exists(MODEL_PATH):
            log.error(f"❌ Model file not found at {MODEL_PATH}. Strategy cannot run.")
            log.error("Please run 'train_model.py' first.")
            return None
        try:
            model = joblib.load(MODEL_PATH)
            log.info(f"✅ Successfully loaded model from {MODEL_PATH}")
            return model
        except Exception as e:
            log.error(f"❌ Failed to load model: {e}", exc_info=True)
            return None

    async def start(self, risk_manager: RiskManager):
        """Main loop to listen to enriched data and run strategy."""
        if not self.model:
            log.error("Strategy stopping: Model not loaded.")
            return
            
        log.info("▶️ ML Strategy starting (listening for enriched data)...")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(ENRICHED_CHANNEL)
        
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    data = json.loads(msg.get("data"))
                    # Create a non-blocking task to handle the event
                    asyncio.create_task(self._handle_enriched_event(data, risk_manager))
                except Exception as e:
                    log.error(f"Error processing enriched message: {e}", exc_info=True)
        except asyncio.CancelledError:
            log.info("ML Strategy cancelled.")
        finally:
            await pubsub.unsubscribe(ENRICHED_CHANNEL)
            log.info("ML Strategy stopped.")

    async def _handle_enriched_event(self, data: dict, risk_manager: RiskManager):
        """
        Handles a new enriched data event.
        Applies all filters and generates a signal if conditions are met.
        """
        symbol = data.get("symbol")
        if not symbol:
            return

        # 1. Check Cooldown
        now = time.time()
        if (now - self.last_signal_ts[symbol]) < self.signal_cooldown:
            return
            
        # 2. Check Risk Manager (is bot allowed to trade?)
        if risk_manager.circuit_open:
            return
            
        # ✅ --- NEW: Stale MTF Data Filter ---
        # Prevents trading on stale trend data just before a new candle forms.
        try:
            if self.config["TREND_CHECK_ENABLED"]:
                now_ts_sec = data['timestamp'] / 1_000_000
                timeframe_key = self.config["TREND_TIMEFRAME"]
                
                if timeframe_key in RESOLUTION_SECONDS:
                    seconds_in_timeframe = RESOLUTION_SECONDS[timeframe_key]
                    
                    next_bar_ts = (int(now_ts_sec / seconds_in_timeframe) + 1) * seconds_in_timeframe
                    time_to_next_bar = next_bar_ts - now_ts_sec
                    
                    # 2-minute buffer before candle close
                    if time_to_next_bar < 120: 
                        log.debug(f"Skipping signal for {symbol}: {time_to_next_bar:.0f}s to {timeframe_key} bar close (stale data).")
                        return
                else:
                    log.warning(f"Invalid TREND_TIMEFRAME '{timeframe_key}' in config.")
        except Exception as e:
            log.error(f"Error in stale data filter: {e}")
            return # Fail safe
        # --- END STALE DATA FILTER ---

        # 3. Apply Heuristic & Strategy Filters
        # Use an async lock to ensure only one signal is processed at a time
        async with self._strategy_lock:
            # Re-check cooldown and circuit breaker inside the lock
            if (now - self.last_signal_ts[symbol]) < self.signal_cooldown or risk_manager.circuit_open:
                return
                
            signal_payload = await self._run_strategy(data)
            
            if signal_payload:
                self.last_signal_ts[symbol] = now
                
                # ✅ --- CRITICAL FIX: Publish directly using self.redis ---
                try:
                    await self.redis.publish(
                        SIGNAL_CHANNEL, json.dumps(signal_payload)
                    )
                    log.info(f"🚀 Published {signal_payload['direction']} Signal for {signal_payload['symbol']}")
                except Exception as e:
                    log.error(f"Failed to publish signal: {e}")
                # --- END FIX ---


    async def _run_strategy(self, data: dict) -> Optional[Dict[str, Any]]:
        """
        Core strategy logic:
        1. Run Heuristic Filters (OBI, TFI)
        2. Run ML Model Prediction
        3. Run Confirmation Filters (Trend, Volume, Funding, S/R)
        4. Calculate SL/TP
        5. Check R/R
        """
        symbol = data.get("symbol")
        
        # --- 1. Heuristic Filters (OBI & TFI) ---
        if not self._check_heuristic_filters(data):
            return None # Failed basic filter
            
        # --- 2. ML Model Prediction ---
        features_df, latest_candles = self._prepare_features(data)
        if features_df is None:
            log.debug(f"Not enough data to form features for {symbol}")
            return None
            
        try:
            probabilities = self.model.predict_proba(features_df.to_numpy())[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            direction = TARGET_MAP.get(prediction_idx, "NEUTRAL")
        except Exception as e:
            log.error(f"Model prediction failed for {symbol}: {e}", exc_info=True)
            return None
            
        # Check signal confidence and direction
        if direction == "NEUTRAL" or confidence < SIGNAL_CONFIDENCE:
            log.debug(f"Skipping signal for {symbol}: Direction={direction}, Conf={confidence:.2f}")
            return None
            
        log.info(f"ML Signal: {symbol} {direction} (Conf: {confidence:.2f})")

        # --- 3. Confirmation Filters ---
        if not self._check_confirmation_filters(data, direction):
            return None # Failed advanced filters

        # --- 4. Calculate SL/TP ---
        try:
            entry_price = float(data['mid_price'])
            atr = (
                data.get("tas", {})
                .get(self.config["ATR_TIMEFRAME"], {})
                .get("atr") 
            )
            if not atr:
                log.warning(f"Skipping {symbol}: Missing ATR data for SL/TP calc.")
                return None

            sl_price, tp_price = self._calculate_sl_tp(
                entry_price, atr, direction
            )
        except Exception as e:
            log.error(f"SL/TP calculation failed for {symbol}: {e}")
            return None

        # --- 5. Check Risk/Reward ---
        if not self._check_risk_reward(entry_price, sl_price, tp_price, direction):
            log.warning(f"Skipping {symbol} {direction}: Failed R/R check. Entry={entry_price}, SL={sl_price}, TP={tp_price}")
            return None

        # --- All Checks Passed ---
        log.info(f"✅ PASSED ALL FILTERS: {symbol} {direction} @ {entry_price} (SL={sl_price}, TP={tp_price})")
        
        signal_payload = {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "size_hint": self.config["BASE_POSITION_SIZE"], # Using static size
            "trigger_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "atr": atr,
            "candles": latest_candles # Pass candle data for context
        }
        return signal_payload

    def _check_heuristic_filters(self, data: dict) -> bool:
        """Check OBI and TFI against thresholds."""
        try:
            if abs(data.get("imbalance", 0)) < self.config["OBI_THRESHOLD"]:
                log.debug(f"Skipping {data['symbol']}: Heuristic OBI filter failed")
                return False
            if abs(data.get("tfi", 0)) < self.config["TFI_THRESHOLD"]:
                log.debug(f"Skipping {data['symbol']}: Heuristic TFI filter failed")
                return False
        except Exception:
            return False
        return True

    def _prepare_features(self, data: dict) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        Prepares a single-row DataFrame for the XGBoost model.
        Must match the features from 'train_model.py'.
        """
        try:
            # Use 5m timeframe as the base for features
            base_tf = "5m"
            tas = data.get("tas", {}).get(base_tf, {})
            if not tas:
                return None, None
                
            # These must match 'train_model.py'
            feature_cols = ['EMA_20', 'EMA_50', 'RSI', 'ATR', 'MACDh', 
                            'OBV', 'ADX', 'Vol_Ratio', 'Close_vs_EMA20']
            
            vol_sma_key = f"SMA_volume_{self.config['VOLUME_SMA_PERIOD']}"
            vol_sma = tas.get(vol_sma_key)
            
            # ✅ --- START FIX: Use all 9 REAL features ---
            ema_20 = tas.get('ema_20')
            ema_50 = tas.get('ema_50')
            rsi = tas.get('rsi_14')
            atr = tas.get('atr')
            macd_hist = tas.get('macd_hist')
            volume = tas.get('volume')
            close_price = tas.get('close')
            # --- NEWLY ADDED ---
            obv = tas.get('obv')
            adx = tas.get('adx')
            
            # Check for missing critical data
            required_keys = [
                ema_20, ema_50, rsi, atr, macd_hist, 
                volume, vol_sma, close_price,
                obv, adx # <-- Added to check
            ]
            if not all(k is not None for k in required_keys):
                # This check ensures all 9 features are present
                return None, None 
            # --- END FIX ---

            features = {
                "EMA_20": ema_20,
                "EMA_50": ema_50,
                "RSI": rsi,
                "ATR": atr,
                "MACDh": macd_hist,
                
                # ✅ --- USE REAL VALUES ---
                "OBV": obv,
                "ADX": adx,
                
                "Vol_Ratio": volume / vol_sma if vol_sma else 1.0,
                "Close_vs_EMA20": (close_price - ema_20) / close_price * 100 if close_price else 0,
            }
            
            df = pd.DataFrame([features])
            df = df[feature_cols] # Ensure column order
            
            # Get latest 10 candles for context (placeholder)
            latest_candles = list(data.get("tas", {}).get("1m", {}).values()) 
            
            return df, latest_candles

        except Exception as e:
            log.error(f"Error preparing features: {e}", exc_info=True)
            return None, None

    def _check_confirmation_filters(self, data: dict, direction: str) -> bool:
        """
        Checks Trend, Funding, Volume, and S/R filters.
        """
        symbol = data.get("symbol", "N/A")
        
        # --- 1. Trend Filter ---
        if self.config["TREND_CHECK_ENABLED"]:
            try:
                tf = self.config["TREND_TIMEFRAME"]
                tas = data.get("tas", {}).get(tf, {})
                if not tas.get("ema_20") or not tas.get("ema_50"):
                    log.warning(f"Skipping trend check for {symbol}: Missing EMA data on {tf}.")
                    return False
                
                is_uptrend = tas["ema_20"] > tas["ema_50"]
                
                if direction == "LONG" and not is_uptrend:
                    log.debug(f"Blocking {direction} signal for {symbol}: Not in uptrend on {tf}.")
                    return False
                if direction == "SHORT" and is_uptrend:
                    log.debug(f"Blocking {direction} signal for {symbol}: Not in downtrend on {tf}.")
                    return False
            except Exception as e:
                log.error(f"Error in Trend Filter for {symbol}: {e}")
                return False

        # --- 2. Funding Rate Filter ---
        if self.config["FUNDING_CHECK_ENABLED"]:
            try:
                funding_rate = float(data.get("funding_rate", 0))
                threshold = self.config["FUNDING_RATE_THRESHOLD"]
                
                if direction == "LONG" and funding_rate > threshold:
                    log.debug(f"Blocking {direction} signal for {symbol}: Funding too high ({funding_rate}).")
                    return False
                if direction == "SHORT" and funding_rate < -threshold:
                    log.debug(f"Blocking {direction} signal for {symbol}: Funding too low ({funding_rate}).")
                    return False
            except Exception as e:
                log.error(f"Error in Funding Filter for {symbol}: {e}")
                return False

        # --- 3. Volume Filter ---
        if self.config["VOLUME_CHECK_ENABLED"]:
            try:
                tf = self.config["VOLUME_TIMEFRAME"]
                sma_period = self.config["VOLUME_SMA_PERIOD"]
                multiplier = self.config["VOLUME_SURGE_MULTIPLIER"]
                
                tas = data.get("tas", {}).get(tf, {})
                vol_sma_key = f"SMA_volume_{sma_period}"
                
                if not tas.get("volume") or not tas.get(vol_sma_key):
                    log.warning(f"Skipping volume check for {symbol}: Missing Volume/Vol_SMA data on {tf}.")
                    return False
                    
                is_surge = tas["volume"] > (tas[vol_sma_key] * multiplier)
                
                if not is_surge:
                    log.debug(f"Blocking signal for {symbol}: No volume surge detected on {tf}.")
                    return False
            except Exception as e:
                log.error(f"Error in Volume Filter for {symbol}: {e}")
                return False
                
        # --- 4. S/R Filter ---
        if self.config["SNR_CHECK_ENABLED"]:
            try:
                price = float(data['mid_price'])
                proximity_pct = self.config["SNR_PROXIMITY_PCT"]
                
                # Get Daily pivots
                pivots = data.get("tas", {}).get("1d", {})
                levels = [
                    pivots.get("pivot"), pivots.get("R1"), pivots.get("S1"),
                    pivots.get("R2"), pivots.get("S2"), pivots.get("R3"), pivots.get("S3")
                ]
                # Get Prev Week High/Low
                levels.extend([data.get("PWH"), data.get("PWL")])
                
                for level in levels:
                    if level is None: continue
                    if abs(price - level) / price < proximity_pct:
                        log.debug(f"Blocking signal for {symbol}: Price {price} too close to S/R level {level}.")
                        return False
            except Exception as e:
                log.error(f"Error in S/R Filter for {symbol}: {e}")
                return False

        return True # All filters passed

    def _calculate_sl_tp(self, entry_price: float, atr: float, direction: str) -> Tuple[float, float]:
        """Calculates SL and TP based on ATR and R/R ratio."""
        
        sl_distance = atr * self.config["SL_ATR_MULTIPLIER"]
        # TP distance must be at least R:R * SL distance
        tp_distance = sl_distance * self.config["MIN_RISK_REWARD_RATIO"]
        
        # Add a small buffer to TP
        tp_distance *= (1.0 + self.config["TP_BUFFER_PCT"])

        if direction == "LONG":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else: # SHORT
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            
        # TODO: Add rounding based on product tick_size
        
        return sl_price, tp_price

    def _check_risk_reward(self, entry: float, sl: float, tp: float, direction: str) -> bool:
        """Final check on R/R ratio."""
        try:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            
            if risk == 0: return False
            
            ratio = reward / risk
            
            return ratio >= self.config["MIN_RISK_REWARD_RATIO"]
        except Exception:
            return False
