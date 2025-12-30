# --- detla-bot/train_hybrid.py ---
# 🧠 HYBRID TRAINER (TFT + PPO)
# Trains the "Eyes" (TFT) to predict price, and the "Brain" (PPO) to trade it.

import logging
import argparse
import pandas as pd
import os
import torch
import numpy as np

from tft_model import TFTPredictor
from rl_agent import RLAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRAIN_HYBRID]: %(message)s")
log = logging.getLogger("train_hybrid")

def load_data(path: str = "fused_data_real.csv"):
    if not os.path.exists(path):
        # Fallback to sample if real data not found
        if os.path.exists("fused_data_sample.csv"):
           path = "fused_data_sample.csv"
           log.warning("Real data not found. Falling back to fused_data_sample.csv")
        else:
           log.error(f"Data file {path} not found. Run generate_training_data.py first.")
           return None
           
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sanitize Data for Training
    # Ensure all required columns exist (fill with defaults if missing)
    required_cols = [
        "close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", 
        "vix_close", "obi", "funding_roc", "dist_to_long_liq", 
        "dist_to_short_liq", "longShortRatio", "dist_to_poc", "oi_pct_change"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 # Default fill
        
        # Fill NaNs
        df[col] = df[col].fillna(0.0)
        # Handle Infinite values (replace with 0)
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        # Clip extreme outliers (to prevent exploding gradients)
        df[col] = df[col].clip(-10.0, 10.0)
        
    return df

def train_layer_1_tft(df: pd.DataFrame, epochs: int = 1):
    log.info("=== Phase 2.1: Training The Predictor (TFT) ===")
    
    # max_prediction_length=7 matches the ML Strategy lookahead
    predictor = TFTPredictor(max_encoder_length=60, max_prediction_length=7)
    train_ds = predictor.prepare_data(df)
    predictor.build_model(train_ds)
    
    # Train
    predictor.train(max_epochs=epochs)
    
    # Create directory if needed
    os.makedirs("model_institutional", exist_ok=True)
    predictor.save("model_institutional/best_sharpe_model.pth")
    
    return predictor

def train_layer_2_rl(df: pd.DataFrame, predictor: TFTPredictor, timesteps: int = 10000):
    log.info("=== Phase 2.2: Training The Strategist (PPO) ===")
    
    # Generate the "Vision" (Forecast) for the RL Agent
    log.info("Generating Forecasts for RL context...")
    
    # Generate predictions
    # We use the predictor to generate 'feature_forecast' for the entire history
    # This allows the RL agent to see what the TFT model *would have seen*
    try:
        if hasattr(predictor, 'predict_batch'):
            preds = predictor.predict_batch(df)
        else:
            preds = predictor.predict(df)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        # Fallback: Zero forecast
        preds = np.zeros(len(df))

    # Create a new DF for RL that includes the forecast
    df_rl = df.copy()
    
    # Process Predictions to match DataFrame length
    # TFT predictions usually start after 'max_encoder_length'
    # We need to pad the beginning or trim the end to align them.
    
    # If preds is multi-dimensional (N, 7), take the mean of the next 3 steps
    if preds.ndim > 1:
        avg_preds = np.mean(preds[:, :3], axis=1)
    else:
        avg_preds = preds

    # Alignment:
    # If we have fewer predictions than rows (normal, due to warm-up), pad the front
    if len(avg_preds) < len(df_rl):
        padding = np.zeros(len(df_rl) - len(avg_preds))
        avg_preds = np.concatenate([padding, avg_preds])
    # If we have more (rare), slice
    elif len(avg_preds) > len(df_rl):
        avg_preds = avg_preds[:len(df_rl)]
        
    df_rl["feature_forecast"] = avg_preds
    
    # Train PPO
    agent = RLAgent(df_rl, model_path="model_institutional/ppo_agent_v1")
    agent.train(total_timesteps=timesteps)
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="RL Training Steps")
    parser.add_argument("--epochs", type=int, default=1, help="TFT Training Epochs")
    args = parser.parse_args()
    
    # 1. Load Data
    df = load_data()
    if df is not None:
        # 2. Train Layer 1 (The Eyes)
        tft_model = train_layer_1_tft(df, epochs=args.epochs)
        
        # 3. Train Layer 2 (The Brain) using the Enriched Data
        rl_agent = train_layer_2_rl(df, tft_model, timesteps=args.steps)
        
        # 4. Verify
        log.info("=== Verification: Running Backtest ===")
        rl_agent.backtest()
        
        log.info("✅ Hybrid Training Complete.")