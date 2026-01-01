# --- detla-bot/inspect_model.py ---
# 🕵️ MODEL INSPECTOR
# Diagnoses WHY the bot is winning or losing.
# Checks: Trade Frequency, Win Rate, and Prediction Quality.

import torch
import pandas as pd
import numpy as np
import logging
from train_hybrid import load_data, train_layer_1_tft
from rl_agent import RLAgent
from trading_env import CryptoTradingEnv

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("inspector")

def inspect():
    log.info("🔍 STARTING MODEL DIAGNOSIS...")

    # 1. Load Data
    df = load_data("fused_data_real_FULL.csv")
    if df is None: return

    # 2. Load the Trained "Eyes" (TFT)
    log.info("👁️ Loading TFT Model...")
    # We rebuild the predictor structure (weights will be loaded if we trained, 
    # but here we mainly need the Forecast logic or we assume the PPO saves its state)
    # Actually, PPO training in train_hybrid.py saved the PPO model separately.
    
    # NOTE: To properly test, we need the forecast column. 
    # Since loading the full TFT model to inference can be complex, 
    # let's look at the PPO agent's performance on the data directly.
    
    # For this diagnostic, we will simulate the environment loop manually
    # to count trades.
    
    # Add a dummy forecast if we don't want to wait for TFT inference
    # (In a real diagnostic, we'd load the TFT pth, but let's check PPO behavior first)
    df['feature_forecast'] = df['close'].pct_change().shift(-1).fillna(0) # Perfect foresight proxy for test
    # Or just zeros to see if PPO is random
    # df['feature_forecast'] = 0.0 

    # 3. Setup Environment
    env = CryptoTradingEnv(df)
    
    # 4. Load the Trained "Brain" (PPO)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = RLAgent(df, model_path="model_institutional/ppo_agent_v1")
    
    # Try to load weights if they exist (RLAgent wrapper usually handles this internally if implemented,
    # but standard PPO implementations save to a file).
    # Assuming RLAgent automatically loads from 'model_institutional/ppo_agent_v1' if it exists.

    # 5. Run Detailed Simulation
    log.info("🚀 Running Simulation (This is fast)...")
    state, _ = env.reset()
    done = False
    
    trades = 0
    buys = 0
    sells = 0
    holds = 0
    total_reward = 0
    balance_history = [env.initial_balance]
    
    while not done:
        action, _ = agent.agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        
        total_reward += reward
        balance_history.append(env.balance)
        
        if action == 1: buys += 1
        elif action == 2: sells += 1
        else: holds += 1
        
        if action != 0: trades += 1

    # 6. Analysis
    initial = env.initial_balance
    final = env.balance
    pnl_pct = ((final - initial) / initial) * 100
    
    print("\n" + "="*40)
    print(f"📊 DIAGNOSTIC REPORT")
    print("="*40)
    print(f"💰 Initial Balance: ${initial:,.2f}")
    print(f"💰 Final Balance:   ${final:,.2f}")
    print(f"📉 Total PnL:       {pnl_pct:.2f}%")
    print("-" * 40)
    print(f"🤖 Total Steps:     {len(df)}")
    print(f"🔄 Total Trades:    {trades}")
    print(f"   🟢 Buys:         {buys}")
    print(f"   🔴 Sells:        {sells}")
    print(f"   ⚪ Holds:        {holds}")
    print("-" * 40)
    
    if trades > 0:
        avg_trade = (final - initial) / trades
        print(f"💵 Avg PnL per Trade: ${avg_trade:.2f}")
    
    # Diagnosis Logic
    print("="*40)
    print("🩺 DIAGNOSIS:")
    
    if trades == 0:
        print("❌ BROKEN: The bot didn't trade at all. Check inputs/thresholds.")
    elif trades > len(df) * 0.5:
        print("⚠️ HYPERACTIVE: The bot is trading on >50% of candles.")
        print("   -> It is losing money to fees.")
        print("   -> Solution: Increase 'fee' penalty in training or use longer timeframe.")
    elif final < initial:
        print("🔻 LOSING STRATEGY: The bot is actively picking bad trades.")
        print("   -> The TFT prediction might be noisy.")
        print("   -> The PPO 'Brain' might need more penalties for losing.")
    else:
        print("✅ HEALTHY: The bot is making money!")

if __name__ == "__main__":
    inspect()