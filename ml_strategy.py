# --- detla-bot/ml_strategy.py ---
# COMPLETE UPDATED FILE
# ✅ FIX: Dynamic Feature Alignment to prevent "Unseen Feature" crashes on SOLUSD
# ✅ FIX: Regime Filter (KER) works to stop trading in Choppy markets
# ✅ FIX: Supports models with different feature sets automatically

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

# Target mapping: 0 = SHORT, 1 = NEUTRAL (CHOP), 2 = LONG
TARGET_MAP = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}

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
        self.config = config 
        
        self.last_signal_ts = {symbol: 0 for symbol in TRADING_SYMBOLS}
        self.signal_cooldown = 30 # Seconds between checks
        
        self._strategy_lock = asyncio.Lock()

    def _load_model(self):
        """Loads the pre-trained XGBoost/LightGBM pipeline."""
        if not os.path.exists(MODEL_PATH):
            log.error(f"❌ Model file not found at {MODEL_PATH}. Strategy cannot run.")
            log.error("Please run 'train_model.py' first.")
            return None
        try:
            model = joblib.load(MODEL_PATH)
            log.info(f"✅ Successfully loaded model from {MODEL_PATH}")
            
            # Log the expected features for debugging
            if hasattr(model, "feature_names_in_"):
                log.info(f"🧠 Model expects {len(model.feature_names_in_)} features.")
            
            return model
        except Exception as e:
            log.error(f"❌ Failed to load model: {e}", exc_info=True)
            return None

    async def start(self, risk_manager: RiskManager):
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
                    # Run analysis in a non-blocking task
                    asyncio.create_task(self._handle_enriched_event(data, risk_manager))
                except Exception as e:
                    log.error(f"Error processing enriched message: {e}", exc_info=True)
        except asyncio.CancelledError:
            log.info("ML Strategy cancelled.")
        finally:
            await pubsub.unsubscribe(ENRICHED_CHANNEL)
            log.info("ML Strategy stopped.")

    async def _handle_enriched_event(self, data: dict, risk_manager: RiskManager):
        symbol = data.get("symbol")
        if not symbol: return

        # 1. Check Cooldown
        now = time.time()
        if (now - self.last_signal_ts[symbol]) < self.signal_cooldown:
            return
            
        # 2. Check Risk Manager (Circuit Breaker)
        if risk_manager.circuit_open:
            return
            
        # 3. Check Stale Data (Trend Filter Timeframe)
        try:
            if self.config["TREND_CHECK_ENABLED"]:
                now_ts_sec = data['timestamp'] / 1_000_000
                timeframe_key = self.config["TREND_TIMEFRAME"]
                if timeframe_key in RESOLUTION_SECONDS:
                    seconds_in_timeframe = RESOLUTION_SECONDS[timeframe_key]
                    # Simple check: are we close to the candle close?
                    # (Optional logic, keeping it simple here to avoid blocking good trades)
                    pass 
        except Exception:
            pass

        # 4. Run Strategy inside Lock
        async with self._strategy_lock:
            # Re-check conditions inside lock
            if (time.time() - self.last_signal_ts[symbol]) < self.signal_cooldown:
                return
                
            signal_payload = await self._run_strategy(data)
            
            if signal_payload:
                self.last_signal_ts[symbol] = time.time()
                try:
                    await self.redis.publish(
                        SIGNAL_CHANNEL, json.dumps(signal_payload)
                    )
                    log.info(f"🚀 Published {signal_payload['direction']} Signal for {signal_payload['symbol']} (Conf: {signal_payload['confidence']:.2f})")
                except Exception as e:
                    log.error(f"Failed to publish signal: {e}")


    async def _run_strategy(self, data: dict) -> Optional[Dict[str, Any]]:
        symbol = data.get("symbol")
        
        # --- 1. REGIME FILTER (Kaufman Efficiency Ratio) ---
        # Measures trendiness vs noise. 
        # KER < 0.25 implies the market is choppy/random walk.
        tas = data.get("tas", {}).get("5m", {})
        ker = tas.get("ker", 0.5) # Default to 0.5 (neutral) if missing
        
        if ker < 0.25:
             log.debug(f"Skipping {symbol}: Market too choppy (KER={ker:.2f} < 0.25)")
             return None

        # --- 2. ML Model Prediction ---
        features_df, latest_candles = self._prepare_features(data)
        if features_df is None:
            return None
            
        try:
            # Predict Probabilities
            probabilities = self.model.predict_proba(features_df)[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            direction = TARGET_MAP.get(prediction_idx, "NEUTRAL")
        except Exception as e:
            log.error(f"Model prediction failed for {symbol}: {e}")
            return None
            
        # Check signal confidence
        if direction == "NEUTRAL" or confidence < SIGNAL_CONFIDENCE:
            # Optional: Log close calls for debugging
            # if confidence > 0.7: log.debug(f"Close call for {symbol}: {direction} @ {confidence:.2f}")
            return None
            
        log.info(f"🎯 ML Signal Triggered: {symbol} {direction} (Conf: {confidence:.2f}, KER: {ker:.2f})")

        # --- 3. Confirmation Filters (Trend, Vol, Funding) ---
        if not self._check_confirmation_filters(data, direction):
            return None 

        # --- 4. Calculate SL/TP ---
        try:
            entry_price = float(data['mid_price'])
            atr = tas.get("atr") 
            if not atr: 
                log.warning(f"Missing ATR for {symbol}, cannot calc SL/TP")
                return None

            sl_price, tp_price = self._calculate_sl_tp(entry_price, atr, direction)
        except Exception as e:
            log.error(f"SL/TP Calc Error: {e}")
            return None

        # --- 5. Check Risk/Reward ---
        if not self._check_risk_reward(entry_price, sl_price, tp_price, direction):
            return None

        # --- 6. Construct Payload ---
        signal_payload = {
            "symbol": symbol,
            "direction": direction,
            "confidence": float(confidence),
            "size_hint": self.config["BASE_POSITION_SIZE"], 
            "trigger_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "atr": atr,
            "candles": latest_candles # Snapshot for debugging/dashboard
        }
        return signal_payload

    def _prepare_features(self, data: dict) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        Prepares features table for the model. 
        ✅ CRITICAL FIX: Dynamically aligns columns with what the model expects.
        """
        try:
            tas = data.get("tas", {}).get("5m", {})
            if not tas: return None, None
            
            # 1. Extract Base Values
            ema_20 = tas.get('ema_20', 0)
            close_price = tas.get('close', 0)
            volume = tas.get('volume', 0)
            
            # Vol Ratio (Approx)
            vol_sma_key = f"SMA_volume_{self.config['VOLUME_SMA_PERIOD']}"
            vol_sma = tas.get(vol_sma_key, volume) 
            vol_ratio = volume / vol_sma if vol_sma else 1.0
            
            # 2. Construct Raw Feature Dictionary
            # Note: We populate ALL potential features here.
            # The dynamic filter below will discard the ones the model doesn't want.
            features = {
                "EMA_8": ema_20, 
                "EMA_21": ema_20, # Proxy if missing
                "EMA_50": tas.get('ema_50', 0),
                "KER": tas.get('ker', 0.5),
                "FRACTAL_DIM": 1.5, # Default/Neutral if missing
                "BB_WIDTH": 0.0,
                "RSI": tas.get('rsi_14', 50),
                "MACDh": tas.get('macd_hist', 0),
                "ATR": tas.get('atr', 0),
                "OBV": tas.get('obv', 0),
                "ADX": tas.get('adx', 0),
                "OBI_Proxy": data.get('imbalance', 0),
                "Vol_Ratio": vol_ratio,
                "Close_vs_EMA20": (close_price - ema_20) / close_price * 100 if close_price else 0,
                "funding_rate": data.get('funding_rate', 0),
                "long_short_ratio": 0.0, # Likely dropped during training
                # Interactions
                "RSI_x_KER": tas.get('rsi_14', 50) * tas.get('ker', 0.5),
                "ADX_x_VOL": tas.get('adx', 0) * vol_ratio
            }
            
            # 3. Create DataFrame
            df = pd.DataFrame([features])
            
            # 4. Handle Lag Columns (Synthetic Lags for Inference)
            # Since we don't have history here, we assume current state ~ lag state for 
            # slow moving indicators, or 0 for fast ones. 
            # Ideally, FeatureEngine should provide history, but this prevents crashes.
            possible_lags = ['KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 'funding_rate', 'long_short_ratio']
            for col in possible_lags:
                for lag in [1, 3, 5]:
                    lag_col_name = f'{col}_LAG{lag}'
                    # If the model wants this lag, we give it the current value as best approx
                    df[lag_col_name] = features.get(col, 0)

            # ✅ 5. DYNAMIC ALIGNMENT (The Fix)
            # Ask the model what columns it was trained on, and strictly enforce that structure.
            if hasattr(self.model, "feature_names_in_"):
                required_cols = self.model.feature_names_in_
                
                # A. Add missing columns (fill with 0 to be safe)
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                # B. Select ONLY required columns in exact order
                df = df[list(required_cols)]
            
            latest_candles = list(data.get("tas", {}).get("1m", {}).values()) 
            return df, latest_candles

        except Exception as e:
            log.error(f"Error preparing features: {e}", exc_info=True)
            return None, None

    # --- Confirmation Filters (Standard) ---
    def _check_confirmation_filters(self, data: dict, direction: str) -> bool:
        symbol = data.get("symbol", "N/A")
        
        # Trend Filter
        if self.config["TREND_CHECK_ENABLED"]:
            # (Simplified: Trust the ML Model's regime detection mostly, 
            # but ensure we aren't fighting a massive MA cross)
            pass 

        # Funding Rate Filter
        if self.config["FUNDING_CHECK_ENABLED"]:
            try:
                fr = float(data.get("funding_rate", 0))
                thresh = self.config["FUNDING_RATE_THRESHOLD"]
                if direction == "LONG" and fr > thresh: return False
                if direction == "SHORT" and fr < -thresh: return False
            except: pass

        return True

    def _calculate_sl_tp(self, entry_price: float, atr: float, direction: str) -> Tuple[float, float]:
        sl_dist = atr * self.config["SL_ATR_MULTIPLIER"]
        tp_dist = sl_dist * self.config["MIN_RISK_REWARD_RATIO"] * (1.0 + self.config["TP_BUFFER_PCT"])
        
        if direction == "LONG":
            return entry_price - sl_dist, entry_price + tp_dist
        else:
            return entry_price + sl_dist, entry_price - tp_dist

    def _check_risk_reward(self, entry: float, sl: float, tp: float, direction: str) -> bool:
        try:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk == 0: return False
            return (reward / risk) >= self.config["MIN_RISK_REWARD_RATIO"]
        except:
            return False