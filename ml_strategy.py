# --- detla-bot/ml_strategy.py ---
# 🧠 BRIDGE v3: ROBUST DATAFRAME CONSTRUCTION
# ✅ FIXED: Added missing 'import time'
# ✅ FIXED: 'Incompatible indexer' error by removing .loc assignment
# ✅ UNPACKER: Extract Model, Features, and Thresholds from Joblib Package
# ✅ SMART THRESHOLDS: Uses optimized thresholds

import asyncio
import orjson
import logging
import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict
from collections import deque
import warnings 

from redis import asyncio as aioredis

from config import (
    ENRICHED_CHANNEL, 
    SIGNAL_CHANNEL, 
    TRADING_SYMBOLS, 
    REDIS_POSITION_LOCK_PREFIX,
    config,
    MIN_CONFIDENCE # ✅ Added Import
)
from risk_manager import RiskManager
from risk_manager import RiskManager
from tft_model import TFTPredictor
from rl_agent import PPOAgent
from regime_classifier import RegimeClassifier
from trade_quality_scorer import TradeQualityScorer
from confluence_engine import CouncilOfElders # ✅ GENIUS UPGRADE

log = logging.getLogger("ml_strategy")

# --- Constants ---
MODEL_DIR = "model_institutional"
MODEL_NAME = "best_sharpe_model.pth" 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

class MLForecastingStrategy:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.config = config 
        
        self.model = TFTPredictor(max_encoder_length=60, max_prediction_length=7)
        self._load_or_init_model()
        
        # RL Agent: State Dim=7 (History + Forecast + OBI), Action Dim=3 (Neutral, Long, Short)
        self.rl = PPOAgent(state_dim=7, action_dim=3)
        self.rl_active = True 
        
        # Regime Classifier
        self.regime_classifier = RegimeClassifier(model_dir=MODEL_DIR)
        if not self.regime_classifier.load():
            log.warning("⚠️ Regime Classifier not loaded (no model found). Defaulting to Low Vol.")
        
        # ✅ NEW: Trade Quality Scorer (Legacy Support)
        self.tqs = TradeQualityScorer(config)
        # ✅ NEW: Council of Elders (The Genius Brain)
        self.council = CouncilOfElders(config)
        log.info("✅ Trade Quality Scoring System initialized")
        
        # History Barriers for Sequence Input (Need 60 steps)
        self.history = {sym: deque(maxlen=60) for sym in TRADING_SYMBOLS}
        self.dxy_cache = deque(maxlen=2) 
        
        self.last_signal_ts = {symbol: 0 for symbol in TRADING_SYMBOLS}
        self.signal_cooldown = 5  # Sniper Mode: Continuous Analysis
        self._strategy_lock = asyncio.Lock()

    def _load_or_init_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load(MODEL_PATH)
                log.info(f"✅ Loaded Robust TFT Model from {MODEL_PATH}")
            except Exception as e:
                log.error(f"Failed to load TFT model: {e}")
                self.model.build_model(None)
        else:
            log.warning(f"Model {MODEL_PATH} not found. intializing empty model.")
            self.model.build_model(None)

    async def start(self, risk_manager: RiskManager):
        log.info("▶️ ML Strategy Engine Active (Robust TFT + Heuristic).")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(ENRICHED_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    asyncio.create_task(self._handle_enriched_event(orjson.loads(msg.get("data")), risk_manager))
        except asyncio.CancelledError: log.info("ML Strategy cancelled.")
        finally: await pubsub.unsubscribe(ENRICHED_CHANNEL)

    async def _handle_enriched_event(self, data: dict, risk_manager: RiskManager):
        symbol = data.get("symbol")
        if not symbol: return
        
        # 1. Update Global Macro State (First observed update wins for that tick)
        if "dxy" in data:
            val = float(data["dxy"])
            if not self.dxy_cache or val != self.dxy_cache[-1]:
                self.dxy_cache.append(val)

        # 2. Extract Features & Buffer
        try:
            tas = data.get("tas", {}).get("5m", {})
            if not tas: return

            # Calculate DXY ROC
            dxy_roc = 0.0
            if len(self.dxy_cache) == 2:
                dxy_roc = (self.dxy_cache[-1] - self.dxy_cache[0]) / (self.dxy_cache[0] + 1e-9)

            feature_row = {
                "close_log_ret": float(tas.get("close_log_ret", 0.0)),
                "vol_zscore": float(tas.get("vol_zscore", 0.0)),
                "fear_greed_norm": float(data.get("fng_norm", 0.5)),
                "dxy_roc": dxy_roc,
                "vix": float(data.get("vix", 20.0)),
                "obi": float(data.get("imbalance", 0.0)),
                "close": float(data.get("mid_price", 0.0)), 
                "log_ret": float(tas.get("close_log_ret", 0.0)),
                # ✅ Multi-Modal Features
                "dist_to_long_liq": float(data.get("dist_to_long_liq", 1.0)),
                "dist_to_short_liq": float(data.get("dist_to_short_liq", 1.0)),
                "funding_roc": float(data.get("funding_roc", 0.0)),
                # Correlations (Dynamic)
                "corr_BTCUSD": float(data.get("corr_BTCUSD", 0.0)),
                "corr_ETHUSD": float(data.get("corr_ETHUSD", 0.0))
            }
            self.history[symbol].append(feature_row)
        except Exception as e:
            log.error(f"Feature extraction failed for {symbol}: {e}")
            return

        # 3. Check Signal Conditions
        if risk_manager.circuit_open: return
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        if await self.redis.exists(lock_key): return

        async with self._strategy_lock:
            if (time.time() - self.last_signal_ts[symbol]) < self.signal_cooldown: return
            
            # Need full sequence
            if len(self.history[symbol]) < 60: return

            signal_payload = await self._generate_signal_tft(symbol, data, risk_manager)
            
            if signal_payload:
                self.last_signal_ts[symbol] = time.time()
                await self.redis.publish(SIGNAL_CHANNEL, orjson.dumps(signal_payload))
                conf_str = f"{signal_payload['confidence']:.2f}"
                log.info(f"🧠 TFT SIGNAL: {symbol} {signal_payload['direction']} (Conf: {conf_str})")

    async def _generate_signal_tft(self, symbol: str, data: dict, risk_manager: RiskManager) -> Optional[Dict]:
        df = pd.DataFrame(list(self.history[symbol]))
        
        try:
            # 1. TFT Forecast (Non-Blocking)
            # ⚡ OPTIMIZATION: Run inference in thread to avoid blocking event loop
            preds = await asyncio.to_thread(self.model.predict, df)
            avg_return_forecast = float(np.mean(preds[0, :3])) 
            
            # 2. Decision Logic
            last_row = self.history[symbol][-1]
            
            # -----------------------------------------------------------
            # ✅ NEW: MULTI-TIMEFRAME TREND FILTER (The "Big Move" Detector)
            # -----------------------------------------------------------
            if self.config.get("TREND_CHECK_ENABLED", True):
                tas_1h = data.get("tas", {}).get("1h", {})
                tas_4h = data.get("tas", {}).get("4h", {})
                
                # 1. Define Trend on Higher Timeframes (Price vs EMA50)
                trend_1h = "NEUTRAL"
                if tas_1h:
                    if tas_1h.get("close", 0) > tas_1h.get("ema_50", 0): trend_1h = "BULLISH"
                    elif tas_1h.get("close", 0) < tas_1h.get("ema_50", 0): trend_1h = "BEARISH"
                    
                trend_4h = "NEUTRAL"
                if tas_4h:
                     if tas_4h.get("close", 0) > tas_4h.get("ema_50", 0): trend_4h = "BULLISH"
                     elif tas_4h.get("close", 0) < tas_4h.get("ema_50", 0): trend_4h = "BEARISH"
    
            # --- NEW SNIPER LOGIC STARTS HERE ---
            
            direction = "NEUTRAL"
            confidence = 0.0
            
            # 1. THE BRAIN (Primary Signal)
            if self.rl_active:
                state = [
                    avg_return_forecast,      # ✅ Feature 0: Forecast (Vision)
                    last_row["close_log_ret"],
                    last_row["vol_zscore"],
                    last_row["fear_greed_norm"],
                    last_row["dxy_roc"],
                    last_row["vix"],
                    last_row["obi"]
                ]
                # Check dimensions match before select_action
                if len(state) != 7:
                     log.error(f"State dimension mismatch. Expected 7, got {len(state)}")
                     return None
                     
                action, log_prob = self.rl.select_action(state)
                if action == 1: direction = "LONG"
                elif action == 2: direction = "SHORT"
                confidence = float(np.exp(log_prob.item()))
            else:
                # Manual High-Probability Setup
                if avg_return_forecast > 0.0015: direction = "LONG" # Higher threshold (0.15% move)
                elif avg_return_forecast < -0.0015: direction = "SHORT"
                confidence = 0.7 # Base confidence for strong forecast

            if direction == "NEUTRAL": return None

            # -----------------------------------------------------------
            # ✅ NEW: SOFT SCORING INSTEAD OF HARD FILTERS
            # -----------------------------------------------------------
            
            # Calculate regime probabilities for scoring
            regime_probs = {}
            if self.config.get("TREND_CHECK_ENABLED", True):
                tas_1h = data.get("tas", {}).get("1h", {})
                tas_4h = data.get("tas", {}).get("4h", {})
                
                # Calculate trend strength scores
                if tas_1h:
                    if tas_1h.get("close", 0) > tas_1h.get("ema_50", 0):
                        regime_probs["BULLISH"] = 0.7
                        regime_probs["BEARISH"] = 0.3
                    elif tas_1h.get("close", 0) < tas_1h.get("ema_50", 0):
                        regime_probs["BEARISH"] = 0.7
                        regime_probs["BULLISH"] = 0.3
                    else:
                        regime_probs["NEUTRAL"] = 1.0
            
            # -----------------------------------------------------------
            # ✅ GENIUS UPGRADE: COUNCIL OF ELDERS EVALUATION
            # -----------------------------------------------------------
            
            # 1. Ask The Statistician (Regime) - Moved UP
            regime = "Low Vol (Calm/Trend)" 
            if self.regime_classifier.is_fitted:
                try:
                    # Use recent history (last 20 candles)
                    slice_len = min(len(df), 20)
                    subset = df.iloc[-slice_len:]
                    
                    features = self.regime_classifier.prepare_features(subset)
                    if len(features) > 0:
                        X = self.regime_classifier.scaler.transform(features)
                        # Get probabilities for the last candle
                        probs = self.regime_classifier.model.predict_proba(X)[-1]
                        
                        # Get best regime
                        best_idx = np.argmax(probs)
                        regime = self.regime_classifier.regime_map.get(best_idx, regime)
                        
                        # Add GMM probs to regime_probs for the Council
                        for i, p in enumerate(probs):
                            desc = self.regime_classifier.regime_map.get(i, f"Cluster {i}")
                            regime_probs[desc] = float(p)
                except Exception as e:
                    log.warning(f"Regime prediction failed: {e}")

            # 2. Convene The Council
            # Grab POC distance from Multimodal payload
            # We check 5m or 1h timeframe for latest calculated POC
            tas_data = data.get("tas", {})
            dist_poc = 0.0
            if "5m" in tas_data: dist_poc = float(tas_data["5m"].get("dist_to_poc", 0.0))
            elif "1h" in tas_data: dist_poc = float(tas_data["1h"].get("dist_to_poc", 0.0))
            
            liquidity_state = {
                "dist_to_long_liq": last_row.get("dist_to_long_liq", 1.0),
                "dist_to_short_liq": last_row.get("dist_to_short_liq", 1.0),
                "dist_to_poc": dist_poc
            }
            
            council_result = self.council.evaluate(
                symbol=symbol,
                tft_forecast=avg_return_forecast,
                rl_action=direction,
                rl_confidence=confidence,
                regime=regime,
                regime_probs=regime_probs, # Now contains BULLISH/BEARISH (Trend) AND Low/High Vol (GMM)
                vol_zscore=last_row["vol_zscore"],
                liquidity_state=liquidity_state
            )
            
            # 3. The Verdict
            if council_result["decision"] == "NEUTRAL":
                log.info(f"🚫 {symbol} Council Rejected: {council_result['reason']} (Score: {council_result['confluence_score']:.2f})")
                return None
                
            log.info(f"🧙 COUNCIL PASSED {symbol}: {council_result['decision']} (Conf: {council_result['confidence']:.2f}, Reason: {council_result['reason']})")
            
            # Adopt Council's High-Confluence Decision
            direction = council_result["decision"]
            confidence = council_result["confidence"]

            # 4. Dynamic Sizing via Risk Manager
            base_size = self.config["BASE_POSITION_SIZE"].get(symbol, 1.0) # Assume 1 contract default if not set
            size = risk_manager.calculate_dynamic_size(symbol, confidence, regime, base_size)

            price = float(data['mid_price'])
            atr = float(data.get("tas", {}).get("5m", {}).get("atr", price * 0.01))
            
            # ✅ GENIUS UPGRADE: Adaptive Stops
            sl_mult = risk_manager.get_adaptive_sl_multiplier(regime, base_mult=1.5)
            sl_dist = atr * sl_mult
            
            # Keep reward ratio high (e.g. 1.5R or 2R)
            tp_dist = sl_dist * 2.0 
            
            sl_price = price - sl_dist if direction == "LONG" else price + sl_dist
            tp_price = price + tp_dist if direction == "LONG" else price - tp_dist
            
            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "size_hint": size, 
                "trigger_price": price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "atr": atr,
                "strategy": "TFT_RL_REGIME_V1",
                "regime": regime,
                "reasoning": {
                    "forecast": avg_return_forecast,
                    "vol_zscore": last_row["vol_zscore"],
                    "fng": last_row["fear_greed_norm"],
                    "dxy_roc": last_row["dxy_roc"],
                    "vix": last_row["vix"]
                }
            }

        except Exception as e:
            log.error(f"TFT Inference Error: {e}")
            return None