# --- ml_strategy.py ---
# Complete Updated File (Reverted to Static Sizing)

import asyncio
import json
import logging
from typing import Any, Dict, Tuple, Optional
from redis import asyncio as aioredis
import numpy as np
import joblib 
import os
import pandas as pd 
# The imported TA library name must match the installed one
import pandas_ta_classic as ta 

from config import (
    REDIS_URL, 
    ENRICHED_CHANNEL, 
    SIGNAL_CHANNEL,
    OBI_THRESHOLD,
    TFI_THRESHOLD,
    SIGNAL_CONFIDENCE,
    TREND_CHECK_ENABLED,
    TREND_TIMEFRAME,
    FUNDING_CHECK_ENABLED,
    FUNDING_RATE_THRESHOLD,
    PRIORITY_LIST, 
    MAX_CONCURRENT_TRADES,
    VOLUME_CHECK_ENABLED,
    VOLUME_TIMEFRAME,
    VOLUME_SMA_PERIOD,
    VOLUME_SURGE_MULTIPLIER,
    SNR_CHECK_ENABLED,
    SNR_PROXIMITY_PCT,
    # ✅ Use original Smart TP/SL/R/R Parameters
    ATR_TIMEFRAME,
    SL_ATR_MULTIPLIER,
    TP_BUFFER_PCT,
    MIN_RISK_REWARD_RATIO,
    BASE_POSITION_SIZE, # Static size hint
    config # Used for reading configuration settings
)

log = logging.getLogger("ml_strategy")

# ML Model Configuration
MODEL_PATH = "model/signal_classifier.joblib"
ML_APPROVAL_THRESHOLD = 0.70 
ML_FEATURE_COLS = ['EMA_20', 'EMA_50', 'RSI', 'ATR', 'MACDh', 'OBV', 'ADX', 'Vol_Ratio', 'Close_vs_EMA20']


class MLForecastingStrategy:
    """
    Consumes enriched data and generates trade signals based on a 
    multi-factor heuristic AND a priority list (SOL -> ETH -> BTC).
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client
        self.latest_valid_signals: Dict[str, Dict] = {}
        self.priority_list = PRIORITY_LIST
        self._risk_manager = None # Placeholder for RiskManager instance
        log.info(f"Priority list loaded: {' -> '.join(self.priority_list)}")
        
        self.volume_sma_key = f"SMA_volume_{VOLUME_SMA_PERIOD}"
        
        self.ml_model = self._load_ml_model() 

    def _load_ml_model(self):
        """Load the XGBoost model from the file system."""
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                log.info("🧠 Machine Learning Model loaded successfully.")
                return model
            except Exception as e:
                log.error(f"❌ FAILED to load ML Model from {MODEL_PATH}. Error: {e}", exc_info=True)
                return None
        else:
            log.warning(f"🧠 ML Model file not found at {MODEL_PATH}. Running heuristic-only.")
            return None

    async def _publish_signal(self, signal: dict):
        """Publishes the final signal to Redis."""
        try:
            await self._redis.publish(SIGNAL_CHANNEL, json.dumps(signal))
            log.info(f"📡 Published signal: {signal}")
        except Exception as e:
            log.exception("Redis publish failed: %s", e)
            
    # ------------------------------------------------------------------ #
    # Helper Checks (Logic is unchanged, kept for context)
    # ------------------------------------------------------------------ #
    
    def _check_microstructure(self, obi: float, tfi: float) -> str:
        if (obi > OBI_THRESHOLD) and (tfi > TFI_THRESHOLD): return "LONG"
        elif (obi < -OBI_THRESHOLD) and (tfi < -TFI_THRESHOLD): return "SHORT"
        return None

    def _check_trend_alignment(self, direction: str, mid_price: float, tas: Dict) -> bool:
        if not TREND_CHECK_ENABLED: return True 
        try:
            trend_tas = tas.get(TREND_TIMEFRAME)
            if not trend_tas: return False
            ema_20 = trend_tas.get("ema_20")
            ema_50 = trend_tas.get("ema_50")
            if ema_20 is None or ema_50 is None: return False
            if direction == "LONG":
                return mid_price > ema_20 and ema_20 > ema_50
            elif direction == "SHORT":
                return mid_price < ema_20 and ema_20 < ema_50
            return False
        except Exception: return False

    def _check_funding_alignment(self, direction: str, funding_rate: float) -> bool:
        if not FUNDING_CHECK_ENABLED: return True 
        if funding_rate is None: return False 
        try:
            if direction == "LONG" and funding_rate > FUNDING_RATE_THRESHOLD: return False
            elif direction == "SHORT" and funding_rate < -FUNDING_RATE_THRESHOLD: return False
            return True
        except Exception: return False

    def _check_volume_filter(self, tas: Dict) -> bool:
        if not VOLUME_CHECK_ENABLED: return True
        try:
            volume_data = tas.get(VOLUME_TIMEFRAME)
            if not volume_data: return False
            current_volume = volume_data.get("volume")
            volume_sma = volume_data.get(self.volume_sma_key)
            if current_volume is None or volume_sma is None or volume_sma == 0: return False
            return current_volume > (volume_sma * VOLUME_SURGE_MULTIPLIER)
        except Exception: return False

    def _check_snr_filter(self, direction: str, price: float, enriched: dict) -> bool:
        if not SNR_CHECK_ENABLED: return True
        try:
            prox_amount = price * SNR_PROXIMITY_PCT
            daily_tas = enriched.get("tas", {}).get("1d", {})
            resistance_levels = [daily_tas.get("R1"), daily_tas.get("R2"), enriched.get("PWH"), enriched.get("PMH")]
            support_levels = [daily_tas.get("S1"), daily_tas.get("S2"), enriched.get("PWL"), enriched.get("PML")]
            if direction == "LONG":
                for r in resistance_levels:
                    if r is not None and price >= (r - prox_amount): return False
            elif direction == "SHORT":
                for s in support_levels:
                    if s is not None and price <= (s + prox_amount): return False
            return True
        except Exception: return False
    
    def _check_risk_reward(self, direction: str, entry_price: float, sl_price: float, tp_price: float) -> bool:
        try:
            if direction == "LONG":
                risk = entry_price - sl_price
                reward = tp_price - entry_price
            else: # SHORT
                risk = sl_price - entry_price
                reward = entry_price - tp_price
            if risk <= 0 or reward <= 0: return False
            rr_ratio = reward / risk
            if rr_ratio >= MIN_RISK_REWARD_RATIO:
                log.info(f"✅ R/R check passed: {rr_ratio:.2f}:1")
                return True
            return False
        except Exception: return False


    def _calculate_smart_stops(self, direction: str, price: float, enriched: dict) -> Tuple[Optional[float], Optional[float]]:
        # ... (Unchanged logic for smart stops)
        try:
            atr_data = enriched.get("tas", {}).get(ATR_TIMEFRAME, {})
            atr = atr_data.get("atr")
            if atr is None or atr == 0: return None, None
            sl_price = price - (atr * SL_ATR_MULTIPLIER) if direction == "LONG" else price + (atr * SL_ATR_MULTIPLIER)

            daily_tas = enriched.get("tas", {}).get("1d", {})
            pwh = enriched.get("PWH"); pwl = enriched.get("PWL"); pmh = enriched.get("PMH"); pml = enriched.get("PML")
            tp_price = 0.0
            buffer = price * TP_BUFFER_PCT

            if direction == "LONG":
                resistance_levels = [daily_tas.get("R1"), daily_tas.get("R2"), daily_tas.get("R3"), pwh, pmh]
                valid_resistances = [r for r in resistance_levels if r is not None and r > price]
                if not valid_resistances: return None, None
                nearest_r = min(valid_resistances)
                tp_price = nearest_r - buffer
            else: # SHORT
                support_levels = [daily_tas.get("S1"), daily_tas.get("S2"), daily_tas.get("S3"), pwl, pml]
                valid_supports = [s for s in support_levels if s is not None and s < price]
                if not valid_supports: return None, None
                nearest_s = max(valid_supports)
                tp_price = nearest_s + buffer
                
            return sl_price, tp_price
        except Exception as e:
            log.error(f"Error calculating smart stops: {e}", exc_info=True)
            return None, None


    def _check_ml_filter(self, direction: str, enriched: dict) -> bool:
        # ... (Unchanged logic for ML check)
        if self.ml_model is None:
            return True 

        try:
            tas = enriched.get("tas", {})
            
            def get_ta(key, timeframe='5m'):
                # Using 5m for consistency with training data when calculating complex features
                if key == 'Vol_Ratio':
                    vol = tas.get("5m", {}).get('volume', 0)
                    sma = tas.get("5m", {}).get(f"SMA_volume_{config.get('VOLUME_SMA_PERIOD')}", 1)
                    return vol / sma if sma else 0
                if key == 'Close_vs_EMA20':
                    mid_price = enriched.get('mid_price', 0)
                    ema20 = tas.get("5m", {}).get('ema_20', mid_price)
                    return (mid_price - ema20) / mid_price * 100 if mid_price else 0
                
                return tas.get(timeframe, {}).get(key, 0)

            # Create the input vector X in the EXACT order defined in ML_FEATURE_COLS
            feature_values = [get_ta(col) for col in ML_FEATURE_COLS]
            X_live = np.array([feature_values])
            
            y_proba = self.ml_model.predict_proba(X_live)[0]

            target_class_index = 1 
            if direction == "LONG":
                target_class_index = 2
            elif direction == "SHORT":
                target_class_index = 0

            approval_probability = y_proba[target_class_index]
            
            if approval_probability >= ML_APPROVAL_THRESHOLD:
                log.info(f"🧠 ML Approval PASS: {direction} approved ({approval_probability:.2f} > {ML_APPROVAL_THRESHOLD:.2f}).")
                return True
            else:
                log.debug(f"🧠 ML Approval FAIL: {direction} rejected (Prob: {approval_probability:.2f}).")
                return False

        except Exception as e:
            log.error(f"❌ Error during ML prediction: {e}", exc_info=True)
            return False 

    # ------------------------------------------------------------------ #
    # Signal Decision Logic (Static Size)
    # ------------------------------------------------------------------ #

    def _build_signal_if_valid(self, enriched: dict) -> dict | None:
        
        symbol = enriched.get("symbol", "UNKNOWN")
        obi = enriched.get("imbalance", 0.0)
        tfi = enriched.get("tfi", 0.0)
        mid_price = enriched.get("mid_price")
        funding_rate = enriched.get("funding_rate")
        tas = enriched.get("tas", {}) 

        if mid_price is None: return None 

        microstructure_signal = self._check_microstructure(obi, tfi)
        if not microstructure_signal: return None
            
        is_trend_aligned = self._check_trend_alignment(microstructure_signal, mid_price, tas)
        is_funding_aligned = self._check_funding_alignment(microstructure_signal, funding_rate)
        is_volume_confirmed = self._check_volume_filter(tas)
        is_snr_clear = self._check_snr_filter(microstructure_signal, mid_price, enriched)

        # --- 3. Final GATES: R/R, TP/SL, and ML Check ---
        if is_trend_aligned and is_funding_aligned and is_volume_confirmed and is_snr_clear:
            
            sl_price, tp_price = self._calculate_smart_stops(microstructure_signal, mid_price, enriched)
            if sl_price is None or tp_price is None:
                log.debug(f"Signal for {symbol} blocked: Could not calculate smart TP/SL.")
                return None
            
            is_rr_valid = self._check_risk_reward(microstructure_signal, mid_price, sl_price, tp_price)
            if not is_rr_valid:
                return None # Failed R/R check

            is_ml_approved = self._check_ml_filter(microstructure_signal, enriched)
            if not is_ml_approved:
                return None # Failed ML check
            
            
            # ✅ STATIC position sizing
            contracts_final = BASE_POSITION_SIZE 
            
            
            # ✅ ALL GATES PASSED. Build the signal.
            return {
                "strategy": "heuristic_ml_v1", 
                "symbol": symbol,
                "direction": microstructure_signal,
                "confidence": SIGNAL_CONFIDENCE,
                "size_hint": contracts_final, # Static size hint
                "timestamp": enriched.get("timestamp", None),
                "trigger_price": mid_price, # This is the entry_price
                "tp_price": round(tp_price, 2), # Static rounding
                "sl_price": round(sl_price, 2)  # Static rounding
            }
        
        return None

    async def _process_enriched_message(self, enriched: dict):
        symbol = enriched.get("symbol")
        if not symbol: return

        signal_data = self._build_signal_if_valid(enriched)

        if signal_data:
            self.latest_valid_signals[symbol] = signal_data
        else:
            self.latest_valid_signals.pop(symbol, None)

    async def _decision_loop(self):
        while True:
            try:
                if MAX_CONCURRENT_TRADES > 0:
                    active_position = await self._redis.get("active_position")
                    if active_position:
                        # --- FIX: Removed .decode() ---
                        log.debug(f"Decision loop paused: Active position '{active_position}' detected.")
                        await asyncio.sleep(1) 
                        continue
                
                for symbol in self.priority_list:
                    
                    if symbol in self.latest_valid_signals:
                        signal_to_send = self.latest_valid_signals.pop(symbol) 
                        
                        # --- STATIC SIZING CHECK (Ensures trade size > 0 before firing) ---
                        if signal_to_send.get("size_hint", 0) <= 0:
                            log.warning(f"🚫 Cannot fire signal for {symbol}: size_hint is non-positive.")
                            continue
                            
                        log.info(f"🏆 Firing trade for high-priority symbol: {symbol}")
                        await self._publish_signal(signal_to_send)
                        
                        break 
                
                await asyncio.sleep(1) 

            except asyncio.CancelledError:
                log.info("Decision loop cancelled.")
                break
            except Exception as e:
                log.error(f"Error in decision loop: {e}", exc_info=True)
                await asyncio.sleep(5) 

    async def start(self, risk_manager): # ✅ UPDATED: Accepts risk_manager
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(ENRICHED_CHANNEL)
        self._risk_manager = risk_manager # Store the risk manager reference
        log.info(f"🎯 Heuristic Strategy subscribed to {ENRICHED_CHANNEL}")

        decision_task = asyncio.create_task(self._decision_loop())

        try:
            async for raw in pubsub.listen():
                if raw is None or raw.get("type") != "message":
                    continue

                try:
                    enriched = json.loads(raw.get("data"))
                    if not isinstance(enriched, dict):
                        continue
                    
                    await self._process_enriched_message(enriched)
                        
                except Exception as e:
                    log.error("Error processing enriched data: %s", e)
                    continue
        
        except asyncio.CancelledError:
            log.info("MLStrategy cancelled.")
        except Exception as e:
            log.error(f"💥 MLStrategy crashed: {e}")
        finally:
            if decision_task:
                decision_task.cancel()
                try:
                    await decision_task
                except asyncio.CancelledError:
                    pass 
            log.info("🔻 MLStrategy stopped cleanly.")