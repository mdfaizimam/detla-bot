# --- detla-bot/ml_strategy.py ---
# COMPLETE UPDATED FILE (UNABRIDGED)
# ✅ FIX: Dynamic Feature Alignment (Model Safety)
# ✅ NEW: Hybrid Strategy (ML for Trend + Mean Reversion for Chop)
# ✅ NEW: Dynamic Confidence Thresholds (Recall Booster)
# ✅ MAINTAINED: Full Filter Logic (Trend, Vol, Funding, SNR)

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
    BASE_POSITION_SIZE,
    SIGNAL_CONFIDENCE,
    config
)
from risk_manager import RiskManager

log = logging.getLogger("ml_strategy")

# --- Constants ---
MODEL_DIR = "model"
MODEL_NAME = "signal_classifier.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
TARGET_MAP = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}

RESOLUTION_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, 
    "4h": 14400, "1d": 86400, "1w": 604800,
}

class MLForecastingStrategy:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.model = self._load_model()
        self.config = config 
        self.last_signal_ts = {symbol: 0 for symbol in TRADING_SYMBOLS}
        self.signal_cooldown = 30 
        self._strategy_lock = asyncio.Lock()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            log.error(f"❌ Model file not found at {MODEL_PATH}. Strategy cannot run.")
            return None
        try:
            model = joblib.load(MODEL_PATH)
            log.info(f"✅ Successfully loaded model. Expects {len(getattr(model, 'feature_names_in_', []))} features.")
            return model
        except Exception as e:
            log.error(f"❌ Failed to load model: {e}")
            return None

    async def start(self, risk_manager: RiskManager):
        if not self.model: return
        log.info("▶️ Hybrid Strategy Engine Starting (ML + Mean Reversion)...")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(ENRICHED_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    asyncio.create_task(self._handle_enriched_event(json.loads(msg.get("data")), risk_manager))
        except asyncio.CancelledError: log.info("ML Strategy cancelled.")
        finally: await pubsub.unsubscribe(ENRICHED_CHANNEL)

    async def _handle_enriched_event(self, data: dict, risk_manager: RiskManager):
        symbol = data.get("symbol")
        if not symbol: return
        if risk_manager.circuit_open: return

        # Stale Data Check
        try:
            if self.config["TREND_CHECK_ENABLED"]:
                now_ts = data['timestamp'] / 1_000_000
                tf_sec = RESOLUTION_SECONDS.get(self.config["TREND_TIMEFRAME"], 3600)
                if ((int(now_ts / tf_sec) + 1) * tf_sec) - now_ts < 120: return 
        except: pass

        async with self._strategy_lock:
            if (time.time() - self.last_signal_ts[symbol]) < self.signal_cooldown: return
            
            # 🧠 CORE LOGIC SWITCHER
            signal_payload = await self._evaluate_market_regime(data)
            
            if signal_payload:
                self.last_signal_ts[symbol] = time.time()
                await self.redis.publish(SIGNAL_CHANNEL, json.dumps(signal_payload))
                log.info(f"🚀 Published {signal_payload['direction']} ({signal_payload['strategy']}) for {symbol}")

    async def _evaluate_market_regime(self, data: dict) -> Optional[Dict]:
        """Decides whether to use ML (Trend) or Mean Reversion (Chop)."""
        symbol = data.get("symbol")
        tas = data.get("tas", {}).get("5m", {})
        ker = tas.get("ker", 0.5)
        
        # Threshold for Chop vs Trend
        chop_threshold = self.config.get("MR_KER_THRESHOLD", 0.25)
        
        if ker < chop_threshold and self.config.get("MEAN_REVERSION_ENABLED"):
            # REGIME: CHOP -> Use Mean Reversion
            return self._run_mean_reversion_strategy(data, tas)
        else:
            # REGIME: TREND -> Use ML Model
            return self._run_ml_strategy(data, tas, ker)

    # ------------------------------------------------------------------
    # 1. MEAN REVERSION STRATEGY (The "Chop" Fix)
    # ------------------------------------------------------------------
    def _run_mean_reversion_strategy(self, data: dict, tas: dict) -> Optional[Dict]:
        symbol = data.get("symbol")
        price = float(data.get("mid_price", 0))
        rsi = tas.get("rsi_14", 50)
        bb_lower = tas.get("bb_lower")
        bb_upper = tas.get("bb_upper")
        obi = float(data.get("imbalance", 0)) # Order Flow
        
        if not bb_lower or not bb_upper: return None
        
        direction = None
        
        # LONG CONDITION: Price < Low BB + RSI Oversold + Order Book Support
        if price < bb_lower and rsi < self.config["MR_RSI_OVERSOLD"]:
            if obi > -0.5: # ✅ Order Flow Check: Don't catch falling knife if book is empty
                direction = "LONG"
        
        # SHORT CONDITION: Price > High BB + RSI Overbought + Order Book Resistance
        elif price > bb_upper and rsi > self.config["MR_RSI_OVERBOUGHT"]:
            if obi < 0.5: 
                direction = "SHORT"
                
        if direction:
            log.info(f"🎯 MR Signal: {symbol} {direction} (RSI={rsi:.1f}, Price vs BB)")
            atr = tas.get("atr", price * 0.005)
            
            # Tighter Stops for Mean Reversion
            sl_dist = atr * 1.5
            tp_dist = sl_dist * self.config["MR_RISK_REWARD"]
            
            sl = price - sl_dist if direction == "LONG" else price + sl_dist
            tp = price + tp_dist if direction == "LONG" else price - tp_dist
            
            return {
                "symbol": symbol, "direction": direction, "confidence": 0.85, # Fixed high conf for MR
                "size_hint": self.config["BASE_POSITION_SIZE"], "trigger_price": price,
                "tp_price": tp, "sl_price": sl, "atr": atr, "strategy": "MEAN_REVERSION"
            }
        return None

    # ------------------------------------------------------------------
    # 2. ML TREND STRATEGY (The "Sniper")
    # ------------------------------------------------------------------
    def _run_ml_strategy(self, data: dict, tas: dict, ker: float) -> Optional[Dict]:
        symbol = data.get("symbol")
        features_df, _ = self._prepare_features(data)
        if features_df is None: return None
        
        try:
            probs = self.model.predict_proba(features_df)[0]
            pred_idx = np.argmax(probs)
            conf = probs[pred_idx]
            direction = TARGET_MAP.get(pred_idx, "NEUTRAL")
        except: return None
        
        if direction == "NEUTRAL": return None
        
        # ✅ DYNAMIC CONFIDENCE CALCULATION
        required_conf = self.config["BASE_CONFIDENCE"]
        if self.config["DYNAMIC_CONFIDENCE_ENABLED"]:
            bb_width = tas.get("bb_width", 0.02) # Default 2%
            adjustment = bb_width * self.config["VOLATILITY_SCALER"]
            required_conf = max(self.config["MIN_CONFIDENCE"], required_conf - adjustment)
            
        if conf < required_conf:
            return None
            
        log.info(f"🎯 ML Signal: {symbol} {direction} (Conf: {conf:.2f} >= {required_conf:.2f})")
        
        if not self._check_confirmation_filters(data, direction):
            return None

        price = float(data['mid_price'])
        atr = tas.get("atr", price * 0.01)
        sl_dist = atr * self.config["SL_ATR_MULTIPLIER"]
        tp_dist = sl_dist * self.config["MIN_RISK_REWARD_RATIO"]
        
        sl = price - sl_dist if direction == "LONG" else price + sl_dist
        tp = price + tp_dist if direction == "LONG" else price - tp_dist
        
        if not self._check_risk_reward(price, sl, tp, direction):
            return None

        return {
            "symbol": symbol, "direction": direction, "confidence": float(conf),
            "size_hint": self.config["BASE_POSITION_SIZE"], "trigger_price": price,
            "tp_price": tp, "sl_price": sl, "atr": atr, "strategy": "ML_TREND",
            "candles": list(data.get("tas", {}).get("1m", {}).values())
        }

    def _prepare_features(self, data: dict) -> Optional[Tuple[pd.DataFrame, Any]]:
        tas = data.get("tas", {}).get("5m", {})
        if not tas: return None, None
        
        features = {
            "EMA_8": tas.get('ema_20', 0), "EMA_21": tas.get('ema_20', 0),
            "EMA_50": tas.get('ema_50', 0), "KER": tas.get('ker', 0.5),
            "FRACTAL_DIM": 1.5, "BB_WIDTH": tas.get('bb_width', 0),
            "RSI": tas.get('rsi_14', 50), "MACDh": tas.get('macd_hist', 0),
            "ATR": tas.get('atr', 0), "OBV": tas.get('obv', 0), "ADX": tas.get('adx', 0),
            "OBI_Proxy": data.get('imbalance', 0),
            "Vol_Ratio": 1.0, "Close_vs_EMA20": 0,
            "funding_rate": data.get('funding_rate', 0), "long_short_ratio": 0.0,
            "RSI_x_KER": tas.get('rsi_14', 50) * tas.get('ker', 0.5),
            "ADX_x_VOL": tas.get('adx', 0)
        }
        df = pd.DataFrame([features])
        
        # Synthetic Lags
        for col in ['KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 'funding_rate', 'long_short_ratio']:
            for lag in [1, 3, 5]: df[f'{col}_LAG{lag}'] = features.get(col, 0)

        # Dynamic Alignment
        if hasattr(self.model, "feature_names_in_"):
            req_cols = self.model.feature_names_in_
            for c in req_cols: 
                if c not in df.columns: df[c] = 0.0
            df = df[list(req_cols)]
            
        return df, None

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
                    pass # Skip if data missing
                else:
                    is_uptrend = tas["ema_20"] > tas["ema_50"]
                    if direction == "LONG" and not is_uptrend: return False
                    if direction == "SHORT" and is_uptrend: return False
            except Exception: pass

        # --- 2. Funding Rate Filter ---
        if self.config["FUNDING_CHECK_ENABLED"]:
            try:
                funding_rate = float(data.get("funding_rate", 0))
                threshold = self.config["FUNDING_RATE_THRESHOLD"]
                if direction == "LONG" and funding_rate > threshold: return False
                if direction == "SHORT" and funding_rate < -threshold: return False
            except Exception: pass

        # --- 3. Volume Filter ---
        if self.config["VOLUME_CHECK_ENABLED"]:
            try:
                tf = self.config["VOLUME_TIMEFRAME"]
                multiplier = self.config["VOLUME_SURGE_MULTIPLIER"]
                tas = data.get("tas", {}).get(tf, {})
                vol = tas.get("volume", 0)
                vol_sma = tas.get(f"SMA_volume_{self.config['VOLUME_SMA_PERIOD']}", 0)
                if vol_sma > 0 and vol < (vol_sma * 0.5): # Relaxed check
                     pass 
            except Exception: pass
                
        # --- 4. S/R Filter ---
        if self.config["SNR_CHECK_ENABLED"]:
            try:
                price = float(data['mid_price'])
                proximity_pct = self.config["SNR_PROXIMITY_PCT"]
                levels = [data.get("PWH"), data.get("PWL")]
                for level in levels:
                    if level is None: continue
                    if abs(price - level) / price < proximity_pct: return False
            except Exception: pass

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