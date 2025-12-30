# --- confluence_engine.py ---
# 🧠 WORLD GENIUS BRAIN: Council of Elders
# Implements Meta-Labeling and Confluence Filtering
# Combines: Strategist (TFT), Tactician (RL), Statistician (Regime)

import numpy as np
import logging
import joblib
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path

# Try importing sklearn, handle gracefully if missing (though it should be standard)
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

log = logging.getLogger("confluence_engine")

MODEL_DIR = Path(__file__).parent / "models" / "council"

class CouncilOfElders:
    def __init__(self, config: dict):
        self.config = config
        self.meta_model = None
        self.is_fitted = False
        self._ensure_model_dir()
        self._load_meta_model()

    def _ensure_model_dir(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _load_meta_model(self):
        model_path = MODEL_DIR / "meta_judge_rf.joblib"
        if model_path.exists() and SKLEARN_AVAILABLE:
            try:
                self.meta_model = joblib.load(model_path)
                self.is_fitted = True
                log.info(f"✅ Council Meta-Judge loaded from {model_path}")
            except Exception as e:
                log.error(f"Failed to load Meta-Judge: {e}")
                self.meta_model = None
        else:
            log.info("⚠️ No Meta-Judge model found. Running in Heuristic Mode.")

    def evaluate(
        self,
        symbol: str,
        tft_forecast: float,      # Strategist: Expected Ret
        rl_action: str,           # Tactician: LONG/SHORT/NEUTRAL
        rl_confidence: float,     # Tactician: Confidence
        regime: str,              # Statistician: Market State
        regime_probs: Dict[str, float],
        vol_zscore: float,
        liquidity_state: Dict[str, float] # dist_to_long_liq, etc.
    ) -> Dict[str, float]:
        """
        The Council Convenes to decide on a trade.
        Returns: {
            "decision": "LONG"|"SHORT"|"NEUTRAL",
            "confidence": float,
            "confluence_score": float (0-1),
            "reason": str
        }
        """
        
        # 1. Gather Expert Opinions
        consensus_score = 0.0
        reasons = []

        # --- The Strategist (Trend) ---
        # TFT forecasts log returns. > 0.001 is decent trend.
        trend_direction = "NEUTRAL"
        if tft_forecast > 0.001: trend_direction = "LONG"
        elif tft_forecast < -0.001: trend_direction = "SHORT"
        
        # --- The Tactician (Timing) ---
        # RL Agent action
        
        # --- The Statistician (Regime) ---
        # Veto power on high volatility?
        is_safe = regime != "High Vol (Crash)"
        
        # 2. Heuristic Confluence Check (The "Setup" Waiter)
        # Alignment: Trend + Tactician must agree
        alignment = False
        if trend_direction == rl_action and rl_action != "NEUTRAL":
            alignment = True
            consensus_score += 0.4
            reasons.append(f"Strategist & Tactician Agree ({rl_action})")
        
        # Regime Alignment
        # If LONG, want Bullish or Trend or Low Vol
        # If SHORT, want Bearish or Trend or High Vol
        regime_aligned = False
        if rl_action == "LONG":
            if regime in ["BULLISH", "TREND", "Low Vol (Calm/Trend)"]:
                regime_aligned = True
                consensus_score += 0.2
        elif rl_action == "SHORT":
            if regime in ["BEARISH", "TREND", "High Vol (Crash)"]: # Shorts ok in crash
                regime_aligned = True
                consensus_score += 0.2
                
        if regime_aligned:
            reasons.append(f"Regime Aligned ({regime})")
            
        # 3. Smart Money / Liquidity Check
        # Genius Bot waits for "Sweeps"
        # If LONG, we prefer price to be close to Long Liquidity (Swing Low) -> "Stop Hunt" complete?
        # Actually, if we are grabbing liquidity, price dips BELOW swing low (dist < 0) then reclaims.
        # Simple Logic: Bonus if near liquidity clusters
        liq_bonus = 0.0
        dist_long = liquidity_state.get("dist_to_long_liq", 1.0)
        dist_short = liquidity_state.get("dist_to_short_liq", 1.0)
        
        if rl_action == "LONG" and dist_long < 0.01: # Near swing low
             liq_bonus = 0.2
             reasons.append("Near Liquidity (Long)")
        elif rl_action == "SHORT" and dist_short < 0.01: # Near swing high
             liq_bonus = 0.2
             reasons.append("Near Liquidity (Short)")
             
        consensus_score += liq_bonus
        
        # 5. Magnet Check (Mean Reversion Filter)
        # If we are excessively extended from POC (> 2%), wait for pullback.
        dist_poc = liquidity_state.get("dist_to_poc", 0.0)
        if abs(dist_poc) > 0.02: 
            reasons.append(f"Overextended from POC ({dist_poc*100:.1f}%)")
            is_safe = False # Veto the trade
        
        # 4. Meta-Judge Decision (ML Overlay)
        meta_prob = 0.5
        if self.is_fitted and SKLEARN_AVAILABLE:
            # Construct feature vector for Meta-Model
            # [forecast, rl_conf, vol_z, regime_probs...]
            try:
                features = np.array([[
                    tft_forecast,
                    rl_confidence,
                    vol_zscore,
                    regime_probs.get("BULLISH", 0),
                    regime_probs.get("BEARISH", 0),
                    dist_long,
                    dist_short
                ]])
                meta_prob = self.meta_model.predict_proba(features)[0, 1] # Prob of Class 1 (Win)
                consensus_score = (consensus_score + meta_prob) / 2 # Blend heuristic + ML
                reasons.append(f"Meta-Judge Prob: {meta_prob:.2f}")
            except Exception:
                pass
        
        # Final Decision Logic
        final_decision = "NEUTRAL"
        final_conf = 0.0
        
        # Threshold: Need Alignment + Safety (or high score)
        if alignment and is_safe:
            # Base confidence from RL, boosted by Confluence
            final_conf = rl_confidence * (1 + consensus_score)
            final_conf = min(0.99, final_conf)
            
            if consensus_score > 0.5: # Hard threshold for "Genius" entry
                final_decision = rl_action
            else:
                reasons.append("Low Confluence")
                
        elif not is_safe:
            reasons.append("Unsafe Regime")
            
        elif not alignment:
            reasons.append("Experts Disagree")

        return {
            "decision": final_decision,
            "confidence": final_conf,
            "confluence_score": consensus_score,
            "reason": " | ".join(reasons)
        }

    def train_meta_judge(self, training_data: pd.DataFrame):
        """
        Train the Random Forest Meta-Judge on historical trade outcomes.
        training_data cols: [forecast, rl_conf, vol_z, bull_prob, bear_prob, dist_long, dist_short, TARGET_WIN]
        """
        if not SKLEARN_AVAILABLE:
            log.warning("Sklearn not available. Cannot train Meta-Judge.")
            return

        if len(training_data) < 100:
            log.warning("Not enough data to train Meta-Judge.")
            return

        X = training_data.iloc[:, :-1]
        y = training_data.iloc[:, -1]
        
        self.meta_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.meta_model.fit(X, y)
        self.is_fitted = True
        
        # Save
        try:
            joblib.dump(self.meta_model, MODEL_DIR / "meta_judge_rf.joblib")
            log.info("✅ Meta-Judge trained and saved.")
        except Exception as e:
            log.error(f"Failed to save Meta-Judge: {e}")
