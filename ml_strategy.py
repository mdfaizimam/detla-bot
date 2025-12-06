# --- detla-bot/ml_strategy.py ---
# ✅ FIX: Non-blocking ML Inference (Run in Executor)
# ✅ FIX: Per-Symbol Locking for higher throughput

import asyncio
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Set, Tuple
from collections import defaultdict
import time 
import warnings 

from redis import asyncio as aioredis
from config import ENRICHED_CHANNEL, SIGNAL_CHANNEL, TRADING_SYMBOLS, SIGNAL_CONFIDENCE, config
from risk_manager import RiskManager

log = logging.getLogger("ml_strategy")

MODEL_DIR = "model"
MODEL_NAME = "signal_classifier.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
TARGET_MAP = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}

# --- Constants ---
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
        # ✅ FIX: Dictionary of locks for parallel symbol processing
        self._strategy_locks = defaultdict(asyncio.Lock) 

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            log.error(f"❌ Model file not found at {MODEL_PATH}")
            return None
        try:
            model = joblib.load(MODEL_PATH)
            log.info(f"✅ Successfully loaded ML model. Expects {len(getattr(model, 'feature_names_in_', []))} features.")
            return model
        except Exception as e:
            log.error(f"❌ Failed to load model: {e}")
            return None

    async def start(self, risk_manager: RiskManager):
        if not self.model: return
        log.info("▶️ Hybrid Strategy Engine Starting...")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(ENRICHED_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    # Fire and forget task to not block listener
                    asyncio.create_task(self._handle_enriched_event(json.loads(msg.get("data")), risk_manager))
        except asyncio.CancelledError: log.info("ML Strategy cancelled.")
        finally: await pubsub.unsubscribe(ENRICHED_CHANNEL)

    async def _handle_enriched_event(self, data: dict, risk_manager: RiskManager):
        symbol = data.get("symbol")
        if not symbol: return
        if risk_manager.circuit_open: return

        # Trend Check logic
        try:
            if self.config["TREND_CHECK_ENABLED"]:
                now_ts = data['timestamp'] / 1_000_000
                tf_sec = RESOLUTION_SECONDS.get(self.config["TREND_TIMEFRAME"], 3600)
                if ((int(now_ts / tf_sec) + 1) * tf_sec) - now_ts < 120: return 
        except: pass

        # ✅ FIX: Per-symbol lock
        async with self._strategy_locks[symbol]:
            if (time.time() - self.last_signal_ts.get(symbol, 0)) < self.signal_cooldown: return

            if not self._gatekeeper_check(data):
                return

            allowed_directions = self._get_allowed_directions(data)
            if not allowed_directions: return

            signal_payload = await self._evaluate_market_regime(data, allowed_directions)
            
            if signal_payload:
                self.last_signal_ts[symbol] = time.time()
                await self.redis.publish(SIGNAL_CHANNEL, json.dumps(signal_payload))
                log.info(f"🚀 Published {signal_payload['direction']} ({signal_payload['strategy']}) for {symbol}")

    def _calculate_smart_size(self, symbol: str, confidence: float) -> float:
        base_sizes = self.config["BASE_POSITION_SIZE"]
        base_size = base_sizes.get(symbol, 0.001)
        if not self.config.get("ENABLE_SMART_SIZING", False): return base_size
        floor = self.config["CONFIDENCE_FLOOR"]
        ceiling = self.config["CONFIDENCE_CEILING"]
        min_mult = self.config["MIN_SIZE_MULTIPLIER"]
        max_mult = self.config["MAX_SIZE_MULTIPLIER"]
        conf = max(floor, min(confidence, ceiling))
        scaler = (conf - floor) / (ceiling - floor) if ceiling != floor else 1.0
        multiplier = min_mult + (scaler * (max_mult - min_mult))
        smart_size = base_size * multiplier
        return round(smart_size, 4)

    def _gatekeeper_check(self, data: dict) -> bool:
        if not self.config.get("GATEKEEPER_ENABLED", True): return True
        symbol = data.get("symbol")
        if self.config["VOLUME_CHECK_ENABLED"]:
            tf = self.config["VOLUME_TIMEFRAME"]
            tas = data.get("tas", {}).get(tf, {})
            vol = tas.get("volume", 0)
            vol_sma = tas.get(f"SMA_volume_{self.config['VOLUME_SMA_PERIOD']}", 0)
            if vol_sma > 0 and vol < (vol_sma * self.config.get("GATEKEEPER_VOL_THRESHOLD", 0.25)):
                return False
        return True

    def _get_allowed_directions(self, data: dict) -> Set[str]:
        allowed = {'LONG', 'SHORT'}
        if self.config["TREND_CHECK_ENABLED"]:
            try:
                tf = self.config["TREND_TIMEFRAME"]
                tas = data.get("tas", {}).get(tf, {})
                ema20 = tas.get("ema_20")
                ema50 = tas.get("ema_50")
                if ema20 and ema50:
                    if ema20 > ema50: allowed.discard('SHORT')
                    elif ema20 < ema50: allowed.discard('LONG')
            except: pass
        if self.config["FUNDING_CHECK_ENABLED"]:
            try:
                funding_rate = float(data.get("funding_rate", 0))
                threshold = self.config["FUNDING_RATE_THRESHOLD"]
                if funding_rate > threshold: allowed.discard('LONG')
                elif funding_rate < -threshold: allowed.discard('SHORT')
            except: pass
        return allowed

    async def _evaluate_market_regime(self, data: dict, allowed_directions: Set[str]) -> Optional[Dict]:
        tas = data.get("tas", {}).get("5m", {})
        ker = tas.get("ker", 0.5)
        
        if ker < self.config.get("MR_KER_THRESHOLD", 0.25) and self.config.get("MEAN_REVERSION_ENABLED"):
            return self._run_mean_reversion_strategy(data, tas, allowed_directions)
        else:
            return await self._run_ml_strategy(data, tas, ker, allowed_directions)

    def _run_mean_reversion_strategy(self, data: dict, tas: dict, allowed_directions: Set[str]) -> Optional[Dict]:
        symbol = data.get("symbol")
        price = float(data.get("mid_price", 0))
        rsi = tas.get("rsi_14", 50)
        bb_lower = tas.get("bb_lower")
        bb_upper = tas.get("bb_upper")
        obi = float(data.get("imbalance", 0))
        
        if not bb_lower or not bb_upper: return None
        
        direction = None
        if 'LONG' in allowed_directions:
            if price < bb_lower and rsi < self.config["MR_RSI_OVERSOLD"] and obi > -0.5: direction = "LONG"
        
        if 'SHORT' in allowed_directions and not direction:
            if price > bb_upper and rsi > self.config["MR_RSI_OVERBOUGHT"] and obi < 0.5: direction = "SHORT"
                
        if direction:
            atr = tas.get("atr", price * 0.005)
            sl_dist = atr * self.config["SL_ATR_MULTIPLIER"] 
            tp_dist = sl_dist * self.config["MR_RISK_REWARD"]
            sl = price - sl_dist if direction == "LONG" else price + sl_dist
            tp = price + tp_dist if direction == "LONG" else price - tp_dist
            size = self._calculate_smart_size(symbol, 0.85)
            return {
                "symbol": symbol, "direction": direction, "confidence": 0.85, 
                "size_hint": size, "trigger_price": price, "tp_price": tp, "sl_price": sl, 
                "atr": atr, "strategy": "MEAN_REVERSION"
            }
        return None

    async def _run_ml_strategy(self, data: dict, tas: dict, ker: float, allowed_directions: Set[str]) -> Optional[Dict]:
        symbol = data.get("symbol")
        features_df, _ = self._prepare_features(data)
        if features_df is None: return None
        
        try:
            # ✅ FIX: Offload CPU-bound inference to thread executor
            loop = asyncio.get_running_loop()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # predict_proba is blocking, so we await it in a thread
                probs = await loop.run_in_executor(None, self.model.predict_proba, features_df)
                probs = probs[0]
            
            pred_idx = np.argmax(probs)
            conf = probs[pred_idx]
            direction = TARGET_MAP.get(pred_idx, "NEUTRAL")
        except Exception as e:
            log.error(f"ML Prediction Error: {e}")
            return None
        
        if direction == "NEUTRAL" or direction not in allowed_directions: return None
        
        required_conf = self.config["BASE_CONFIDENCE"]
        if self.config["DYNAMIC_CONFIDENCE_ENABLED"]:
            bb_width = tas.get("bb_width", 0.02)
            adjustment = bb_width * self.config["VOLATILITY_SCALER"]
            required_conf = max(self.config["MIN_CONFIDENCE"], required_conf - adjustment)

        if conf < required_conf: return None
            
        price = float(data['mid_price'])
        atr = tas.get("atr", price * 0.01)
        
        sl_dist = atr * self.config["SL_ATR_MULTIPLIER"]
        sl = price - sl_dist if direction == "LONG" else price + sl_dist
        
        # Dynamic TP Logic
        pml = data.get("PML") 
        pmh = data.get("PMH")
        daily_tas = data.get("tas", {}).get("1d", {})
        pivot_r1 = daily_tas.get("R1")
        pivot_s1 = daily_tas.get("S1")
        min_rr = self.config["MIN_RISK_REWARD_RATIO"]
        min_tp_dist = sl_dist * min_rr
        tp = 0.0
        
        if direction == "LONG":
            default_tp = price + min_tp_dist
            candidates = []
            if pivot_r1 and pivot_r1 > price: candidates.append(pivot_r1)
            if pmh and pmh > price: candidates.append(pmh)
            valid_snr_tp = None
            for cand in sorted(candidates):
                if cand >= default_tp:
                    valid_snr_tp = cand
                    break
            tp = valid_snr_tp if valid_snr_tp else default_tp
        else: # SHORT
            default_tp = price - min_tp_dist
            candidates = []
            if pivot_s1 and pivot_s1 < price: candidates.append(pivot_s1)
            if pml and pml < price: candidates.append(pml)
            valid_snr_tp = None
            for cand in sorted(candidates, reverse=True):
                if cand <= default_tp:
                    valid_snr_tp = cand
                    break
            tp = valid_snr_tp if valid_snr_tp else default_tp

        if not self._check_risk_reward(price, sl, tp, direction): return None
        size = self._calculate_smart_size(symbol, float(conf))

        return {
            "symbol": symbol, "direction": direction, "confidence": float(conf),
            "size_hint": size, "trigger_price": price, "tp_price": tp, "sl_price": sl, 
            "atr": atr, "strategy": "ML_TREND",
            "candles": list(data.get("tas", {}).get("1m", {}).values())
        }

    def _prepare_features(self, data: dict) -> Optional[Tuple[pd.DataFrame, Any]]:
        tas = data.get("tas", {}).get("5m", {})
        if not tas: return None, None
        
        features = {
            "EMA_8": tas.get('ema_20', 0), "EMA_21": tas.get('ema_20', 0),
            "EMA_50": tas.get('ema_50', 0), "KER": tas.get('ker', 0.5),
            "FRACTAL_DIM": tas.get('fractal_dim', 1.0), 
            "BB_WIDTH": tas.get('bb_width', 0),
            "RSI": tas.get('rsi_14', 50), "MACDh": tas.get('macd_hist', 0),
            "ATR": tas.get('atr', 0), "OBV": tas.get('obv', 0), "ADX": tas.get('adx', 0),
            "OBI_Proxy": data.get('imbalance', 0),
            "Vol_Ratio": 1.0, "Close_vs_EMA20": 0,
            "funding_rate": data.get('funding_rate', 0), "long_short_ratio": 0.0,
            "RSI_x_KER": tas.get('rsi_14', 50) * tas.get('ker', 0.5),
            "ADX_x_VOL": tas.get('adx', 0)
        }
        df = pd.DataFrame([features])
        
        for col in ['KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 'funding_rate', 'long_short_ratio']:
            for lag in [1, 3, 5]: df[f'{col}_LAG{lag}'] = features.get(col, 0)

        if hasattr(self.model, "feature_names_in_"):
            req_cols = self.model.feature_names_in_
            for c in req_cols: 
                if c not in df.columns: df[c] = 0.0
            df = df[list(req_cols)]
            
        return df, None

    def _check_risk_reward(self, entry: float, sl: float, tp: float, direction: str) -> bool:
        try:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk == 0: return False
            return (reward / risk) >= self.config["MIN_RISK_REWARD_RATIO"]
        except: return False