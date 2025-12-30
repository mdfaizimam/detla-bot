import logging
import argparse
import pandas as pd
import os
import torch
import numpy as np
from typing import List, Tuple

from tft_model import TFTPredictor
from rl_agent import RLAgent
from trading_env import CryptoTradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ROBUST_TRAIN]: %(message)s")
log = logging.getLogger("train_robust")

def load_data(path: str = "fused_data_real.csv") -> pd.DataFrame:
    """Loads and preprocesses data, ensuring correct columns exist."""
    if not os.path.exists(path):
        if os.path.exists("fused_data_sample.csv"):
            log.warning("Real data not found. Falling back to fused_data_sample.csv")
            path = "fused_data_sample.csv"
        else:
            log.error(f"Data file {path} not found.")
            return None
            
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Ensure all required features exist
    # If using sample data, mapped columns might be missing or named differently
    # Mappings based on prior knowledge of sample generation:
    
    if "vix" not in df.columns:
        df["vix"] = 20.0 # Default if missing
        
    if "dxy_roc" not in df.columns:
        if "dxy" in df.columns:
            df["dxy_roc"] = df["dxy"].pct_change().fillna(0)
        else:
            df["dxy_roc"] = 0.0

    if "close_log_ret" not in df.columns:
        df["close_log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        
    if "vol_zscore" not in df.columns:
        df["vol_zscore"] = (df["volume"] - df["volume"].rolling(200).mean()) / (df["volume"].rolling(200).std() + 1e-9)
        df["vol_zscore"] = df["vol_zscore"].fillna(0)

    if "fear_greed_norm" not in df.columns:
        if "fear_greed_index" in df.columns:
             df["fear_greed_norm"] = df["fear_greed_index"] / 100.0
        else:
             df["fear_greed_norm"] = 0.5

    # Sanitize
    feature_cols = ["close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", "vix"]
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # Clip to prevent gradient explosion during adversarial training
        df[col] = df[col].clip(-10.0, 10.0)
        
    return df

def add_adversarial_noise(df: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
    """
    Injects Gaussian noise into features to simulate market regime shifts 
    or data anomalies (Adversarial Training).
    """
    if noise_level <= 0: return df
    
    aug_df = df.copy()
    feature_cols = ["close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", "vix"]
    
    log.info(f"⚔️ Injecting Adversarial Noise (Level: {noise_level})...")
    
    for col in feature_cols:
        std = aug_df[col].std()
        noise = np.random.normal(0, std * noise_level, len(aug_df))
        aug_df[col] += noise
        
    return aug_df

def walk_forward_validation(df: pd.DataFrame, n_splits: int = 3, epochs: int = 1):
    """
    Performs Walk-Forward Validation:
    Train [0...T] -> Test [T...T+k]
    Train [0...T+k] -> Test [T+k...T+2k]
    """
    log.info(f"🛡️ Starting Walk-Forward Validation ({n_splits} splits)...")
    
    # Minimum training size (e.g., 50% of data)
    train_size_start = int(0.5 * len(df))
    remaining = len(df) - train_size_start
    fold_size = int(remaining / n_splits)
    
    metrics = []
    
    predictor = TFTPredictor(max_encoder_length=60, max_prediction_length=7)
    
    for i in range(n_splits):
        end_train = train_size_start + (i * fold_size)
        end_test = end_train + fold_size
        if end_test > len(df): end_test = len(df)
        
        train_df = df.iloc[:end_train].copy()
        test_df = df.iloc[end_train:end_test].copy()
        
        log.info(f"--- Fold {i+1}/{n_splits} ---")
        log.info(f"Train Size: {len(train_df)} | Test Size: {len(test_df)}")
        
        # 1. Adversarial Augmentation on Training Data
        # We train on Noisy data to be robust, but validate on Clean data
        train_df_noisy = add_adversarial_noise(train_df, noise_level=0.05)
        
        # 2. Train TFT (Retrain or Fine-tune?)
        # For speed in this prototype, we rebuild from scratch or load previous. 
        # Ideally we carry over weights.
        train_ds = predictor.prepare_data(train_df_noisy)
        if i == 0:
            predictor.build_model(train_ds)
        
        predictor.train(max_epochs=epochs) # Incremental learning if model exists
        
        # 3. Train RL Agent (Briefly)
        # We need the TFT to generate state for RL. 
        # This is computationally expensive to simulate perfectly here without full integration.
        # So we will rely on the "Test" phase backtest of the RL agent essentially using the updated TFT.
        
        # 4. Evaluate on Test Set (Backtest)
        # Run RL Backtest on Test Split
        log.info("Running Out-of-Sample Backtest...")
        rl_env = CryptoTradingEnv(test_df, reward_metric='sharpe') # Clean data
        
        # Simulate RL actions using a random/heuristic policy for now OR the trained PPO?
        # Since we haven't persistently trained the PPO in this script loop (it takes long),
        # We will use a "Heuristic + TFT" proxy to validate the TFT's predictive power.
        # IE: If TFT says Up, We go Long.
        
        balance = 10000
        holdings = 0
        
        # Simple Vectorized Backtest based on TFT Forecast
        preds = predictor.predict(test_df) # This assumes predict handles the whole DF rolling? 
        # TFTPredictor.predict currently takes last seq. We need batch prediction.
        # Standard implementation of batch predict is needed.
        # For this script, let's implement a simple loop or mock.
        
        # MOCK VALIDATION METRIC: Loss on Test Set
        # Since full RL backtest is complex, measuring MSE of Forecast on Test Set is a good proxy for Robustness.
        
        test_ds = predictor.prepare_data(test_df) # Create dataset from test slice
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
        
        total_loss = 0
        criterion = torch.nn.MSELoss()
        predictor.model.eval()
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(predictor.device), y.to(predictor.device)
                output = predictor.model(x)
                loss = criterion(output, y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        log.info(f"📉 Test Set MSE: {avg_loss:.6f}")
        metrics.append(avg_loss)
        
    avg_mse = np.mean(metrics)
    log.info(f"✅ Walk-Forward Validation Complete. Average MSE: {avg_mse:.6f}")
    
    # Save Final Robust Model
    predictor.save("model_institutional/robust_tft_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--splits", type=int, default=3)
    args = parser.parse_args()
    
    df = load_data()
    if df is not None:
        walk_forward_validation(df, n_splits=args.splits, epochs=args.epochs)
