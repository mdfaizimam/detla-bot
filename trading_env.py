# --- detla-bot/trading_env.py ---
# 🛡️ ROBUST TRADING ENV V2
# 1. Fixes 'state_dim' crash
# 2. Includes TFT Forecast in observation
# 3. Clamps math to prevent NaNs

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trading_env")

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, fee=0.0005):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 
        # [Close, Vol, Funding, LS, OI, VIX, DXY, Forecast, Pos_Size, Unrealized_PnL]
        # Total = 10 Features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # ✅ FIX: Explicitly define state_dim for RLAgent
        self.state_dim = self.observation_space.shape[0]
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0 # Positive = Long, Negative = Short
        self.entry_price = 0.0
        self.equity_curve = [self.initial_balance]
        
        return self._next_observation(), {}

    def _next_observation(self):
        # robustly get current row
        obs = self.df.iloc[self.current_step]
        
        # ✅ Added 'feature_forecast' (The TFT Prediction)
        features = [
            obs.get('close', 0),
            obs.get('vol_zscore', 0),
            obs.get('fundingRate', 0),
            obs.get('longShortRatio', 1.0),
            obs.get('oi_pct_change', 0),
            obs.get('fear_greed_norm', 0.5),
            obs.get('dxy_roc', 0),
            obs.get('feature_forecast', 0), # <--- The AI's Vision
            self.position,
            self._get_unrealized_pnl(obs.get('close', 0))
        ]
        
        # Replace NaNs/Infs with 0.0 to protect the neural net
        clean_features = [0.0 if np.isnan(x) or np.isinf(x) else x for x in features]
        return np.array(clean_features, dtype=np.float32)

    def _get_unrealized_pnl(self, current_price):
        if self.position == 0:
            return 0.0
        
        # Avoid division by zero
        if self.entry_price == 0:
            return 0.0

        if self.position > 0: # Long
            raw_pnl = (current_price - self.entry_price) / self.entry_price
        else: # Short
            raw_pnl = (self.entry_price - current_price) / self.entry_price
            
        # 🛡️ SAFETY CLAMP: Cap PnL between -100% and +500%
        return np.clip(raw_pnl, -1.0, 5.0)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        done = False
        
        # 1. Execute Action
        if action == 1: # Buy / Long
            if self.position <= 0: # Flip or Open
                self._close_position(current_price)
                self._open_position(current_price, 1) # Long
                
        elif action == 2: # Sell / Short
            if self.position >= 0: # Flip or Open
                self._close_position(current_price)
                self._open_position(current_price, -1) # Short

        # 2. Calculate Step Reward (Change in Equity)
        pnl_pct = self._get_unrealized_pnl(current_price)
        current_equity = self.balance + (pnl_pct * self.initial_balance)
        
        # Safety check for equity
        if np.isnan(current_equity) or np.isinf(current_equity):
            current_equity = self.initial_balance
            
        prev_equity = self.equity_curve[-1]
        
        # Prevent div/0 for return calculation
        if prev_equity <= 1e-6: 
            prev_equity = 1e-6
            
        step_return = (current_equity - prev_equity) / prev_equity
        
        # 🛡️ CLAMP REWARD: Clip reward to [-1, 1]
        reward = np.clip(step_return, -1.0, 1.0)
        
        self.equity_curve.append(current_equity)
        
        # 3. Advance Step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        return self._next_observation(), reward, done, False, {}

    def _open_position(self, price, side):
        self.position = side
        self.entry_price = price
        # Deduct Fee
        self.balance -= (self.balance * self.fee)

    def _close_position(self, price):
        if self.position == 0: return
        
        pnl_pct = self._get_unrealized_pnl(price)
        pnl_cash = self.balance * pnl_pct
        
        self.balance += pnl_cash
        self.balance -= (self.balance * self.fee)
        
        self.position = 0
        self.entry_price = 0
        
        if self.balance < 10: self.balance = 10 # Prevent bankruptcy