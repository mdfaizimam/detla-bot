# --- trade_quality_scorer.py ---
# 🎯 Trade Quality Scoring System (TQS)
# Replaces hard boolean filters with soft probabilistic scoring
# Enables dynamic thresholds and adaptive trade selection

import numpy as np
from typing import Dict, Optional, List
from collections import deque
import logging

log = logging.getLogger("tqs")

class TradeQualityScorer:
    """
    Converts hard filters → soft scores
    Enables dynamic thresholding based on opportunity landscape
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Component weights (sum to 1.0)
        self.weights = {
            "ml_confidence": 0.30,      # ML model prediction confidence
            "rl_advantage": 0.25,        # RL agent Q-value advantage
            "regime_alignment": 0.20,    # Regime probability
            "volatility_score": 0.15,    # Market activity
            "risk_score": 0.10           # Risk metrics
        }
        
        # Historical score tracking for dynamic thresholds
        self.score_history = {
            "BTCUSD": deque(maxlen=500),
            "ETHUSD": deque(maxlen=500),
            "SOLUSD": deque(maxlen=500),
        }
        
        # Daily score buffer for percentile-based selection
        self.daily_scores = {symbol: [] for symbol in self.score_history.keys()}
        self.last_reset_day = None
        
    def calculate_tqs(
        self,
        symbol: str,
        ml_confidence: float,
        rl_advantage: float,
        regime_probs: Dict[str, float],
        volatility_zscore: float,
        obi: float,
        direction: str
    ) -> Dict[str, float]:
        """
        Calculate Trade Quality Score (TQS)
        Returns a dict with score breakdown
        """
        
        # 1. ML Confidence Score (already 0-1)
        ml_score = ml_confidence
        
        # 2. RL Advantage Score (normalize to 0-1)
        # Assuming RL advantage is in range [-1, 1]
        rl_score = (rl_advantage + 1.0) / 2.0
        
        # 3. Regime Alignment Score
        regime_score = self._calculate_regime_score(regime_probs, direction)
        
        # 4. Volatility Score
        volatility_score = self._calculate_volatility_score(volatility_zscore)
        
        # 5. Risk Score (from OBI and other metrics)
        risk_score = self._calculate_risk_score(obi, direction)
        
        # Calculate weighted TQS
        tqs = (
            self.weights["ml_confidence"] * ml_score +
            self.weights["rl_advantage"] * rl_score +
            self.weights["regime_alignment"] * regime_score +
            self.weights["volatility_score"] * volatility_score +
            self.weights["risk_score"] * risk_score
        )
        
        breakdown = {
            "tqs": tqs,
            "ml_score": ml_score,
            "rl_score": rl_score,
            "regime_score": regime_score,
            "volatility_score": volatility_score,
            "risk_score": risk_score
        }
        
        # Track score
        self.score_history[symbol].append(tqs)
        
        return breakdown
    
    def _calculate_regime_score(self, regime_probs: Dict[str, float], direction: str) -> float:
        """
        Score based on regime alignment
        Instead of blocking counter-regime trades, we score them lower
        """
        if not regime_probs:
            return 0.5  # Neutral if no regime data
        
        # For LONG trades, prefer bullish/trend regimes
        # For SHORT trades, prefer bearish/range regimes
        if direction == "LONG":
            score = (
                regime_probs.get("BULLISH", 0.0) * 1.0 +
                regime_probs.get("TREND", 0.0) * 0.8 +
                regime_probs.get("RANGE", 0.0) * 0.4 +
                regime_probs.get("BEARISH", 0.0) * 0.2
            )
        elif direction == "SHORT":
            score = (
                regime_probs.get("BEARISH", 0.0) * 1.0 +
                regime_probs.get("RANGE", 0.0) * 0.6 +
                regime_probs.get("TREND", 0.0) * 0.3 +
                regime_probs.get("BULLISH", 0.0) * 0.2
            )
        else:
            score = 0.5
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_volatility_score(self, vol_zscore: float) -> float:
        """
        Score volatility - prefer moderate volatility
        Too low = dead market
        Too high = choppy/risky
        """
        # Optimal volatility: -0.5 to +1.5 z-score
        if vol_zscore < -1.0:
            # Dead market
            score = 0.2
        elif vol_zscore < -0.5:
            # Low but acceptable
            score = 0.5
        elif vol_zscore <= 1.5:
            # Optimal range - linear increase
            score = 0.5 + (vol_zscore + 0.5) / 2.0 * 0.5
        else:
            # Too high - volatility risk
            score = max(0.3, 1.0 - (vol_zscore - 1.5) * 0.2)
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_risk_score(self, obi: float, direction: str) -> float:
        """
        Score based on order flow alignment
        Instead of blocking, we score counter-flow trades lower
        """
        # OBI ranges from -1 (bearish) to +1 (bullish)
        if direction == "LONG":
            # For LONG, prefer positive OBI (buying pressure)
            # But don't completely block negative OBI (value buying)
            if obi >= 0.2:
                score = 1.0
            elif obi >= 0.0:
                score = 0.7 + obi * 1.5
            elif obi >= -0.3:
                # Mild sell pressure - still tradable
                score = 0.5 + (obi + 0.3) / 0.3 * 0.2
            else:
                # Strong sell pressure
                score = 0.3
                
        elif direction == "SHORT":
            # For SHORT, prefer negative OBI (selling pressure)
            if obi <= -0.2:
                score = 1.0
            elif obi <= 0.0:
                score = 0.7 - obi * 1.5
            elif obi <= 0.3:
                score = 0.5 - (obi - 0.3) / 0.3 * 0.2
            else:
                score = 0.3
        else:
            score = 0.5
        
        return np.clip(score, 0.0, 1.0)
    
    def get_dynamic_threshold(
        self,
        symbol: str,
        mode: str = "percentile"
    ) -> float:
        """
        Calculate dynamic threshold based on recent score distribution
        
        Args:
            symbol: Trading symbol
            mode: 'percentile' (top N%) or 'adaptive' (volatility-based)
        
        Returns:
            Threshold score (0-1)
        """
        if len(self.score_history[symbol]) < 50:
            # Not enough data - use conservative threshold
            return 0.70
        
        scores = list(self.score_history[symbol])
        
        if mode == "percentile":
            # Take top 5-10% of signals
            percentile = self.config.get("TQS_PERCENTILE", 90)
            threshold = np.percentile(scores, percentile)
            
            # Clamp between 0.55 and 0.85
            # This prevents both over-trading and starvation
            threshold = np.clip(threshold, 0.55, 0.85)
            
        elif mode == "adaptive":
            # Threshold adapts to market conditions
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # In high-opportunity markets (high mean), lower threshold
            # In low-opportunity markets, raise threshold to maintain quality
            threshold = mean_score + 0.5 * std_score
            threshold = np.clip(threshold, 0.60, 0.80)
        
        else:
            threshold = 0.70
        
        return threshold
    
    def should_trade(
        self,
        symbol: str,
        tqs: float,
        mode: str = "percentile"
    ) -> tuple[bool, float, str]:
        """
        Determine if trade should execute based on dynamic threshold
        
        Returns:
            (should_trade, threshold_used, reason)
        """
        threshold = self.get_dynamic_threshold(symbol, mode)
        
        if tqs >= threshold:
            reason = f"TQS {tqs:.3f} >= threshold {threshold:.3f}"
            return True, threshold, reason
        else:
            reason = f"TQS {tqs:.3f} < threshold {threshold:.3f}"
            return False, threshold, reason
    
    
    def get_daily_top_signals(self, symbol: str, n: int = 3) -> List[float]:
        """
        Get top N signals of the day
        Useful for "only take best 3 trades per day" logic
        """
        if symbol not in self.daily_scores:
            return []
        
        scores = self.daily_scores[symbol]
        if len(scores) < n:
            return []
        
        # Sort and return top N
        sorted_scores = sorted(scores, reverse=True)
        return sorted_scores[:n]
    
    def reset_daily_buffer(self, symbol: Optional[str] = None):
        """
        Reset daily score buffer
        Call this at day rollover
        """
        if symbol:
            self.daily_scores[symbol] = []
        else:
            for sym in self.daily_scores.keys():
                self.daily_scores[sym] = []
    
    def log_score_breakdown(self, breakdown: Dict[str, float], symbol: str):
        """
        Log detailed score breakdown for analysis
        """
        log.info(
            f"📊 TQS Breakdown for {symbol}: "
            f"Total={breakdown['tqs']:.3f} | "
            f"ML={breakdown['ml_score']:.3f} | "
            f"RL={breakdown['rl_score']:.3f} | "
            f"Regime={breakdown['regime_score']:.3f} | "
            f"Vol={breakdown['volatility_score']:.3f} | "
            f"Risk={breakdown['risk_score']:.3f}"
        )
