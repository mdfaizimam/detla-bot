# --- detla-bot/trading_env.py ---
# 🛡️ ROBUST TRADING ENV V5 (REALITY CHECK MODE)
# 1. Fixed Stake: Trades $10k flat. No compounding. (Fixes $700T bug)
# 2. Cooldown: Must wait 12 steps (1 hour) between trades. (Fixes Hyperactivity)
# 3. High Threshold: Only trades if predicted move > 0.5%.

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
        
        # 🛡️ SNIPER PARAMETERS V2
        self.conviction_threshold = 0.005  # Must predict 0.5% move
        self.trade_penalty = 0.002 
        self.cooldown_steps = 12 # Force 1 hour wait between trades
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 10 Features + 1 for Cooldown tracking
        # We append cooldown_counter to observation so AI knows it's waiting
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        self.state_dim = self.observation_space.shape[0]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0 
        self.entry_price = 0.0
        self.cooldown_counter = 0 # Track steps since last trade closing
        self.equity_curve = [self.initial_balance]
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.iloc[self.current_step]
        
        # Normalize cooldown (0 to 1)
        norm_cooldown = self.cooldown_counter / self.cooldown_steps if self.cooldown_steps > 0 else 0
        
        features = [
            obs.get('close', 0), 
            obs.get('vol_zscore', 0), 
            obs.get('fundingRate', 0),
            obs.get('longShortRatio', 1.0), 
            obs.get('oi_pct_change', 0),
            obs.get('fear_greed_norm', 0.5), 
            obs.get('dxy_roc', 0),
            obs.get('feature_forecast', 0), 
            self.position,
            self._get_unrealized_pnl(obs.get('close', 0)),
            norm_cooldown # New Feature: "Are we on cooldown?"
        ]
        
        clean_features = [0.0 if np.isnan(x) or np.isinf(x) else x for x in features]
        return np.array(clean_features, dtype=np.float32)

    def _get_unrealized_pnl(self, current_price):
        if self.position == 0 or self.entry_price == 0: return 0.0
        if self.position > 0: return (current_price - self.entry_price) / self.entry_price
        else: return (self.entry_price - current_price) / self.entry_price

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        forecast = self.df.iloc[self.current_step].get('feature_forecast', 0)
        
        reward = 0
        done = False
        trade_occurred = False
        
        # Decrease cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # 🛡️ FILTERS
        # 1. Cooldown Check
        if self.cooldown_counter > 0 and action != 0 and self.position == 0:
            action = 0 # Force Hold if cooling down
            
        # 2. Conviction Check
        if action != 0 and abs(forecast) < self.conviction_threshold and self.position == 0:
            action = 0 # Force Hold if signal weak

        # Execution
        if action == 1: # Buy
            if self.position <= 0:
                self._close_position(current_price)
                self._open_position(current_price, 1)
                trade_occurred = True
        elif action == 2: # Sell
            if self.position >= 0:
                self._close_position(current_price)
                self._open_position(current_price, -1)
                trade_occurred = True

        # Reward Calculation
        # FIXED STAKE LOGIC: PnL is applied to initial_balance only.
        pnl_pct = self._get_unrealized_pnl(current_price)
        
        # Current Equity = Realized Cash + Unrealized Profit on Fixed Stake ($10k)
        # We assume position size is always $10k (or whatever initial_balance was)
        unrealized_cash = pnl_pct * self.initial_balance
        current_equity = self.balance + unrealized_cash
        
        prev_equity = self.equity_curve[-1]
        step_return = (current_equity - prev_equity) / self.initial_balance # Normalized return
        
        if trade_occurred:
            step_return -= self.trade_penalty
        
        reward = np.clip(step_return * 10, -1.0, 1.0) # Scale up small returns for PPO
        
        self.equity_curve.append(current_equity)
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        return self._next_observation(), reward, done, False, {}

    def _open_position(self, price, side):
        self.position = side
        self.entry_price = price
        # Fee applied to Fixed Stake ($10k)
        fee_cost = self.initial_balance * self.fee
        self.balance -= fee_cost

    def _close_position(self, price):
        if self.position == 0: return
        
        pnl_pct = self._get_unrealized_pnl(price)
        # Profit applied to Fixed Stake
        pnl_cash = self.initial_balance * pnl_pct
        
        # Fee applied to Fixed Stake
        fee_cost = self.initial_balance * self.fee
        
        self.balance += pnl_cash
        self.balance -= fee_cost
        
        self.position = 0
        self.entry_price = 0
        self.cooldown_counter = self.cooldown_steps # Reset Cooldown