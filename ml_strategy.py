# --- detla-bot/ml_strategy.py ---
# 🧠 BRIDGE v3: ROBUST DATAFRAME CONSTRUCTION
# ✅ FIXED: Added missing 'import time'
# ✅ FIXED: 'Incompatible indexer' error by removing .loc assignment
# ✅ UNPACKER: Extract Model, Features, and Thresholds from Joblib Package
# ✅ SMART THRESHOLDS: Uses optimized thresholds

import asyncio
import json
import logging
import os
import time  # <--- FIXED: Added this import
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict
import warnings 

from redis import asyncio as aioredis

from config import (
    ENRICHED_CHANNEL, 
    SIGNAL_CHANNEL, 
    TRADING_SYMBOLS, 
    REDIS_POSITION_LOCK_PREFIX,
    config
)
from risk_manager import RiskManager

log = logging.getLogger("ml_strategy")

# --- Constants ---
MODEL_DIR = "model"
MODEL_NAME = "signal_classifier.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

class MLForecastingStrategy:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.config = config 
        
        self.model = None
        self.model_features = []
        self.thresholds = {}
        self._load_model_package()
        
        self.last_signal_ts = {symbol: 0 for symbol in TRADING_SYMBOLS}
        self.signal_cooldown = 300  # 5 minutes
        self._strategy_lock = asyncio.Lock()

    def _load_model_package(self):
        if not os.path.exists(MODEL_PATH):
            log.error(f"❌ Model file not found at {MODEL_PATH}")
            return

        try:
            package = joblib.load(MODEL_PATH)
            if isinstance(package, dict):
                self.model = package.get('model')
                self.model_features = package.get('features', [])
                self.thresholds = package.get('prediction_thresholds', {})
                log.info(f"✅ Loaded Smart Model Package.")
            else:
                self.model = package
                self.model_features = getattr(package, "feature_names_in_", [])
                self.thresholds = {}
                log.info(f"⚠️ Loaded Legacy Raw Model.")

        except Exception as e:
            log.error(f"❌ Failed to load model: {e}")

    async def start(self, risk_manager: RiskManager):
        if not self.model: 
            log.error("🛑 No model loaded. Strategy disabled.")
            return
            
        log.info("▶️ ML Strategy Engine Active.")
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
        
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        if await self.redis.exists(lock_key): return

        async with self._strategy_lock:
            if (time.time() - self.last_signal_ts[symbol]) < self.signal_cooldown: return

            if not self._gatekeeper_check(data): return

            signal_payload = await self._generate_signal(data)
            
            if signal_payload:
                self.last_signal_ts[symbol] = time.time()
                await self.redis.publish(SIGNAL_CHANNEL, json.dumps(signal_payload))
                log.info(f"🚀 SIGNAL: {symbol} {signal_payload['direction']} (Conf: {signal_payload['confidence']:.2f})")

    def _calculate_smart_size(self, symbol: str, confidence: float) -> float:
        base_sizes = self.config["BASE_POSITION_SIZE"]
        base_size = base_sizes.get(symbol, 0.001)
        
        if not self.config.get("ENABLE_SMART_SIZING", False):
            return base_size
            
        floor = self.config["CONFIDENCE_FLOOR"]
        ceiling = self.config["CONFIDENCE_CEILING"]
        conf = max(floor, min(confidence, ceiling))
        scaler = (conf - floor) / (ceiling - floor) if ceiling > floor else 0.0
        min_mult = self.config["MIN_SIZE_MULTIPLIER"]
        max_mult = self.config["MAX_SIZE_MULTIPLIER"]
        multiplier = min_mult + (scaler * (max_mult - min_mult))
        
        return round(base_size * multiplier, 4)

    def _gatekeeper_check(self, data: dict) -> bool:
        if not self.config.get("GATEKEEPER_ENABLED", True): return True
        tas = data.get("tas", {}).get("5m", {})
        adx = tas.get("adx", 20)
        return adx >= 15

    async def _generate_signal(self, data: dict) -> Optional[Dict]:
        symbol = data.get("symbol")
        tas = data.get("tas", {}).get("5m", {})
        
        features_df = self._prepare_features(data)
        if features_df is None: return None
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                probs = self.model.predict_proba(features_df)[0]
            
            short_prob, long_prob = probs[0], probs[2] if len(probs) > 2 else 0
            
            direction, conf = "NEUTRAL", 0.0
            thresh_long = self.thresholds.get(2, self.config["BASE_CONFIDENCE"])
            thresh_short = self.thresholds.get(0, self.config["BASE_CONFIDENCE"])
            
            if long_prob > thresh_long and long_prob > short_prob:
                direction, conf = "LONG", long_prob
            elif short_prob > thresh_short and short_prob > long_prob:
                direction, conf = "SHORT", short_prob
            else:
                return None 
                
        except Exception as e:
            log.error(f"Inference Error: {e}")
            return None
        
        price = float(data['mid_price'])
        atr = tas.get("atr", price * 0.01)
        sl_dist = atr * self.config["SL_ATR_MULTIPLIER"]
        sl_price = price - sl_dist if direction == "LONG" else price + sl_dist
        tp_dist = sl_dist * self.config["MIN_RISK_REWARD_RATIO"]
        tp_price = price + tp_dist if direction == "LONG" else price - tp_dist

        if abs(price - sl_price) == 0: return None
        
        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": float(conf),
            "size_hint": self._calculate_smart_size(symbol, float(conf)),
            "trigger_price": price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "atr": atr,
            "strategy": "ML_GBM_V2"
        }

    def _prepare_features(self, data: dict) -> Optional[pd.DataFrame]:
        tas = data.get("tas", {}).get("5m", {})
        if not tas: return None
        
        ema_20 = tas.get('ema_20', 0)
        high = float(tas.get('high', 0) or 0)
        low = float(tas.get('low', 0) or 0)
        close = float(tas.get('close', 0) or 0)
        
        obi_proxy = 0.0
        if (high - low) > 0:
            clv = ((close - low) / (high - low)) 
            obi_proxy = (clv * 2) - 1

        features = {
            "ATR": tas.get('atr', 0),
            "ATR_PCT": tas.get('atr_pct', 0),
            "KER": tas.get('ker', 0.5),
            "FRACTAL_DIM": tas.get('fractal_dim', 1.0),
            "BB_WIDTH": tas.get('bb_width', 0),
            "RSI": tas.get('rsi_14', 50),
            "MACDh": tas.get('macd_hist', 0),
            "ADX": tas.get('adx', 0),
            "OBV": tas.get('obv', 0),
            "OBI_Proxy": obi_proxy, 
            "funding_rate": data.get('funding_rate', 0),
            "long_short_ratio": data.get('long_short_ratio', 1.0),
            "EMA_8": ema_20, "EMA_21": ema_20,
            "EMA_50": tas.get('ema_50', 0), "EMA_200": tas.get('ema_50', 0),
            "EMA_20": ema_20, 
            "RSI_x_KER": tas.get('rsi_14', 50) * tas.get('ker', 0.5),
            "ADX_x_VOL": tas.get('adx', 0) 
        }
        
        # Add Lags (Imputed)
        lag_cols = ['KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 'funding_rate', 'long_short_ratio']
        for col in lag_cols:
            for lag in [1, 3, 5]: 
                features[f'{col}_LAG{lag}'] = features.get(col, 0)

        # ✅ FIXED: Construct DataFrame in one go (No .loc setting)
        if self.model_features:
            # Create a dictionary with only the needed keys, defaulting to 0.0 if missing
            final_data = {col: features.get(col, 0.0) for col in self.model_features}
            df_final = pd.DataFrame([final_data], dtype=float)
            return df_final
            
        return pd.DataFrame([features])