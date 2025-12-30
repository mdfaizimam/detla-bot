import logging
import numpy as np
import torch
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [REGIME]: %(message)s")
log = logging.getLogger("regime_manager")

class RegimeManager:
    """
    The 'Sensors': Detects Market Regime (Bull, Bear, Crash).
    Replaces HMM with a robust Statistical/Clustering approach.
    
    States:
    0: Normal/Low Volatility
    1: High Volatility (Warning)
    2: Crash/Extreme Volatility (Circuit Breaker)
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.returns_buffer = deque(maxlen=window_size)
        self.vol_buffer = deque(maxlen=window_size)
        self.current_regime = 0
        
    def update(self, price_return: float):
        """
        Updates the regime estimate with the latest return.
        """
        self.returns_buffer.append(price_return)
        
        # Calculate Rolling Volatility (Std Dev)
        if len(self.returns_buffer) < 20: 
            return 0 # Not enough data
            
        current_vol = np.std(list(self.returns_buffer)[-20:]) # Short-term vol
        self.vol_buffer.append(current_vol)
        
        # Determine Regime based on Z-Score of Volatility
        # If current vol is > 2 sigmas above historical average -> Crash Mode
        
        hist_vol_mean = np.mean(self.vol_buffer)
        hist_vol_std = np.std(self.vol_buffer) + 1e-9
        
        vol_zscore = (current_vol - hist_vol_mean) / hist_vol_std
        
        if vol_zscore > 3.0:
            self.current_regime = 2 # Crash
        elif vol_zscore > 1.5:
            self.current_regime = 1 # High Vol
        else:
            self.current_regime = 0 # Normal
            
        return self.current_regime
        
    def get_regime_label(self):
        labels = {0: "NORMAL", 1: "HIGH_VOL", 2: "CRASH"}
        return labels.get(self.current_regime, "UNKNOWN")

if __name__ == "__main__":
    # Test
    manager = RegimeManager()
    
    # Simulate Normal Data
    data = np.random.normal(0, 0.001, 100)
    for x in data: manager.update(x)
    print(f"Normal Data Regime: {manager.get_regime_label()}")
    
    # Simulate Crash
    crash_data = np.random.normal(0, 0.02, 20) # 20x volatility
    for x in crash_data: manager.update(x)
    print(f"Crash Data Regime: {manager.get_regime_label()}")
