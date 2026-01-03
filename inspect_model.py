# --- detla-bot/inspect_model.py ---
# 🕵️ INSTITUTIONAL MODEL INSPECTOR
# Diagnoses the Brain (PPO) and Eyes (TFT) of the World Class Bot.

import logging
import pandas as pd
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("inspector")

# Import system components
try:
    from train_hybrid import DataLoader, TFTPredictor, add_rl_features
    from rl_agent import RLAgent
    from trading_env import CryptoTradingEnv
except ImportError:
    log.error("❌ Could not import bot modules. Run this from the 'detla-bot' folder.")
    sys.exit(1)

def inspect():
    print("\n" + "="*50)
    print("🔍 STARTING WORLD CLASS MODEL DIAGNOSIS")
    print("="*50)

    # 1. Load Data
    df = DataLoader.load_data()
    if df is None: return

    # Preprocess
    log.info("🧹 Preprocessing data...")
    df, _ = DataLoader._preprocess_data(df)

    # 2. Load the "Eyes" (TFT)
    log.info("👁️  Loading TFT Model & Generating Alpha...")
    tft_path = "model_institutional/best_sharpe_model.pth"
    
    if os.path.exists(tft_path):
        try:
            tft_model = TFTPredictor()
            tft_model.load(tft_path)
            
            # Enrich data
            df_rl = add_rl_features(df, tft_model)
            
            # 🕵️ DEEP CHECK: Are the eyes working?
            preds = df_rl["feature_forecast"]
            pred_std = preds.std()
            pred_mean = preds.mean()
            log.info(f"📊 Forecast Stats -> Mean: {pred_mean:.6f}, Std: {pred_std:.6f}")
            
            if pred_std < 1e-5:
                print("\n⚠️  CRITICAL WARNING: MODEL COLLAPSE DETECTED ⚠️")
                print("   The TFT model is predicting a flat line (Std ~ 0).")
                print("   The 'Eyes' are blind. The Agent will fail.")
                print("   FIX: The target (log_ret) is too small. Multiply target by 100 or 1000 during training.")
                print("-" * 50)

        except Exception as e:
            log.error(f"❌ Failed to load TFT model: {e}")
            return
    else:
        log.warning("⚠️  TFT Model not found at " + tft_path)
        return

    # 3. Load the "Brain" (PPO)
    log.info("🧠 Loading PPO Agent...")
    model_path = "model_institutional/ppo_agent_v1"
    
    try:
        agent = RLAgent(df_rl, model_path=model_path)
    except Exception as e:
        log.error(f"❌ Failed to load RL Agent: {e}")
        return

    # 4. Simulation
    log.info("🚀 Running Backtest Simulation...")
    
    env = agent.env
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    
    done = False
    total_reward = 0
    positions = [] 
    rewards = []
    
    # Speed up: Run last 20% only if data is huge
    if len(df_rl) > 50000:
        start_idx = int(len(df_rl) * 0.8)
        log.info(f"⏩ Skipping to validation set (Row {start_idx:,})...")
        env.current_step = start_idx
    
    while not done:
        action, _ = agent.agent.select_action(state)
        step_result = env.step(action)
        
        if len(step_result) == 5:
            next_state, reward, done, _, _ = step_result
        else:
            next_state, reward, done, _ = step_result
            
        state = next_state
        total_reward += reward
        positions.append(action)
        rewards.append(reward)

    # 5. Report
    positions = np.array(positions)
    avg_conviction = np.mean(np.abs(positions))
    pct_long = np.sum(positions > 0.25) / len(positions) * 100
    pct_short = np.sum(positions < -0.25) / len(positions) * 100
    
    print("\n" + "="*50)
    print(f"📊 DIAGNOSTIC RESULTS")
    print("="*50)
    print(f"💰 Total Accum Reward:  {total_reward:,.2f}")
    print(f"🌊 Avg Reward/Step:     {np.mean(rewards):.4f}")
    print("-" * 50)
    print(f"🧠 AGENT PSYCHOLOGY")
    print(f"   Avg Conviction:      {avg_conviction:.3f}")
    print(f"   🟢 Aggressive Longs: {pct_long:.1f}%")
    print(f"   🔴 Aggressive Shorts: {pct_short:.1f}%")
    print("-" * 50)

    if pred_std < 1e-5:
        print("🩺 DIAGNOSIS: BLIND AGENT")
        print("   The PPO is struggling because the TFT inputs are flat.")
        print("   ACTION: Retrain TFT with scaled targets.")
    elif total_reward > 0:
        print("🩺 DIAGNOSIS: HEALTHY ✅")
    else:
        print("🩺 DIAGNOSIS: UNDERPERFORMING 🔻")

if __name__ == "__main__":
    inspect()