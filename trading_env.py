# --- detla-bot/trading_env.py ---
import gymnasium as gym
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

class CryptoTradingEnv(gym.Env):
    def __init__(self, df, continuous=True):
        self.df = df
        self.continuous = continuous
        
        # Identify feature columns (exclude timestamp and non-numeric)
        # We assume 'timestamp' is the only non-feature column usually, but let's be safe
        self.feature_cols = []
        for c in df.columns:
            if c not in ['timestamp', 'date', 'time', 'symbol', 'base_asset', 'quote_asset']:
                try:
                    if is_numeric_dtype(df[c]):
                        self.feature_cols.append(c)
                except Exception:
                    pass # Skip problematic columns
        
        # Action Space: Continuous [-1, 1] (Position Size)
        if self.continuous:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(3)
            
        # Observation Space: [Features...]
        self.state_dim = len(self.feature_cols)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_position = 0.0
        self.returns_history = []
        
        # Get first state using pre-selected columns
        state_vals = self.df.iloc[self.current_step][self.feature_cols].values
        return state_vals.astype(np.float32), {}

    def step(self, action):
        # 1. Execute Action
        # continuous action is explicitly position size (-1.0 to 1.0)
        target_position = float(action) if self.continuous else (action - 1)
        
        # 2. Calculate PnL (simplified)
        next_step = self.current_step + 1
        if next_step >= len(self.df):
            done = True
            return np.zeros(self.state_dim, dtype=np.float32), 0, True, False, {}
            
        if 'close' in self.df.columns:
            current_price = self.df.iloc[self.current_step]['close']
            next_price = self.df.iloc[next_step]['close']
            price_change = (next_price - current_price) / current_price
        else:
            # Fallback to log returns if available
            price_change = self.df.iloc[next_step].get('close_log_ret', 0.0)

        # PnL = Position * % Change - Fees
        step_return = (target_position * price_change) - (abs(target_position - self.prev_position) * 0.0005)
        
        self.returns_history.append(step_return)
        
        # 3. Sortino Reward Calculation
        # Reward = Mean Return / Downside Deviation
        window = 50 # Lookback for stats
        recent_returns = np.array(self.returns_history[-window:])
        mean_ret = np.mean(recent_returns)
        
        # Downside deviation: Only count negative returns
        negative_returns = recent_returns[recent_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.01
        
        # Sortino Ratio (Safe division)
        sortino = mean_ret / (downside_std + 1e-6)
        
        # Reward clipping for stability
        reward = np.clip(sortino, -1, 1) * 0.1 + step_return * 10 
        
        self.prev_position = target_position
        self.current_step += 1
        
        # Check termination
        done = self.current_step >= len(self.df) - 1
        
        # Next state
        next_state = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        
        return next_state, reward, done, False, {}