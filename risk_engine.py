import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [RISK]: %(message)s")
log = logging.getLogger("risk_engine")

class RiskEngine:
    """
    The 'Shield': Protects capital using Math (Kelly) and Logic (Circuit Breakers).
    """
    
    def __init__(self, max_leverage=1.0):
        self.max_leverage = max_leverage
        
    def calculate_kelly_size(self, win_prob: float, win_loss_ratio: float = 1.0):
        """
        Calculates fraction of capital to bet using Kelly Criterion.
        f* = p - q/b
        """
        if win_loss_ratio <= 0: return 0.0
        
        kelly_fraction = win_prob - ((1 - win_prob) / win_loss_ratio)
        return max(0.0, kelly_fraction) # No negative sizing
        
    def get_target_size(self, capital: float, confidence: float, regime: int):
        """
        Determines position size based on Confidence (from AI) and Regime (from HMM/Stats).
        """
        
        # 1. Regime Multiplier (The Circuit Breaker)
        regime_mult = 1.0
        if regime == 1: # High Vol
            regime_mult = 0.5
            log.warning("High Volatility detected. Reducing size by 50%.")
        elif regime == 2: # Crash
            regime_mult = 0.0
            log.critical("CRASH DETECTED. HALTING TRADING (Size = 0).")
            
        if regime_mult == 0.0:
            return 0.0
            
        # 2. Kelly Sizing
        # We treat 'confidence' as win_prob. Assume 1:1 R:R for simplicity or estimate it.
        # Safe Kelly: Half-Kelly (0.5 * f*) is standard industry practice.
        raw_kelly = self.calculate_kelly_size(confidence, win_loss_ratio=1.5)
        safe_kelly = raw_kelly * 0.5 
        
        # 3. Final Calculation
        target_fraction = safe_kelly * regime_mult
        
        # Cap at max leverage
        target_fraction = min(target_fraction, self.max_leverage)
        
        position_size = capital * target_fraction
        return position_size

if __name__ == "__main__":
    risk = RiskEngine()
    
    # Scene 1: Normal Market, High Confidence
    s1 = risk.get_target_size(10000, 0.75, regime=0)
    print(f"Normal Market (75% Conf): ${s1:.2f}")
    
    # Scene 2: High Vol, High Confidence
    s2 = risk.get_target_size(10000, 0.75, regime=1)
    print(f"High Vol Market (75% Conf): ${s2:.2f}")
    
    # Scene 3: Crash, High Confidence
    s3 = risk.get_target_size(10000, 0.75, regime=2)
    print(f"Crash Market (75% Conf): ${s3:.2f}")
