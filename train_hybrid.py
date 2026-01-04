# --- detla-bot/train_hybrid.py ---
# 🧠 HYBRID TRAINER (TFT + PPO) - WORLD CLASS EDITION
# Includes: Feature Normalization (Z-Score), Alpha Features, and Robust Loading.

import logging
import argparse
import pandas as pd
import os
import torch
import numpy as np
import warnings
import gc
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path

from tft_model import TFTPredictor
from rl_agent import RLAgent

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [TRAIN_HYBRID]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"train_hybrid_{timestamp}.log"),
            logging.FileHandler(log_dir / "train_hybrid_latest.log", mode='w')
        ]
    )
    return logging.getLogger("train_hybrid")

log = setup_logging()

class DataLoader:
    """Enhanced data loading and preprocessing class"""
    
    REQUIRED_COLUMNS = [
        # Price and volume features
        "close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", 
        "vix_close", "obi", "funding_roc", "dist_to_long_liq", 
        "dist_to_short_liq", "longShortRatio", "dist_to_poc", "oi_pct_change",
        # Basis and MTF Features
        "basis",
        "trend_bias_1h", "volatility_1h", "rsi_1h",
        "trend_bias_4h", "volatility_4h", "rsi_4h",
        # ✅ NEW ALPHA FEATURES (CVD / Order Flow)
        "cvd_velocity", "cvd_zscore"
    ]
    
    @staticmethod
    def find_data_file(search_paths=None) -> Optional[Path]:
        if search_paths is None:
            search_paths = [
                "fused_data_real_FULL.csv",
                "fused_data_real.csv",
                "data/fused_data_real_FULL.csv",
                "data/fused_data_real.csv",
            ]
        for path in search_paths:
            p = Path(path)
            if p.exists():
                log.info(f"📂 Found data file: {p} (size: {p.stat().st_size / 1024 / 1024:.2f} MB)")
                return p
        return None
    
    @staticmethod
    def load_data(path: str = None) -> Optional[pd.DataFrame]:
        if path is None:
            data_file = DataLoader.find_data_file()
            if data_file is None:
                log.error("❌ No data file found. Please run generate_training_data.py first.")
                return None
            path = str(data_file)
        
        try:
            log.info(f"📊 Loading data from: {path}")
            df = pd.read_csv(path)
            
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                
            # Time index for TFT (CRITICAL)
            if "time_idx" not in df.columns:
                df["time_idx"] = np.arange(len(df))
            
            df["time_idx"] = df["time_idx"].astype(int)
            log.info(f"✅ Loaded {len(df):,} rows.")
            return df
            
        except Exception as e:
            log.error(f"❌ Failed to load data: {e}")
            return None
    
    @staticmethod
    def _preprocess_data(df: pd.DataFrame, train_stats: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean, Normalize (Z-Score), and Clip data.
        CRITICAL: We must normalize features so the Transformer can learn.
        """
        df_clean = df.copy()
        
        # 1. Missing Column Check
        for col in DataLoader.REQUIRED_COLUMNS:
            if col not in df_clean.columns:
                df_clean[col] = 0.0
                log.warning(f"⚠️  Missing column '{col}', filled with 0.0")
        
        # 2. Select Features for Normalization
        # We process ALL required columns EXCEPT the target (close_log_ret) 
        # because the target is handled specifically by the Loss Function logic (scaled x1000 there)
        # and used for PnL calculation (must remain real).
        features_to_norm = [c for c in DataLoader.REQUIRED_COLUMNS if c != 'close_log_ret']
        
        # Stats dictionary
        stats = {} if train_stats is None else train_stats
        
        # 3. Z-Score Normalization
        for col in features_to_norm:
            # Handle Infs first
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0.0)
            
            if train_stats is None:
                # == FIT (TRAIN) ==
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std == 0: std = 1.0 # Prevent div/0
                
                stats[col] = {'mean': mean, 'std': std}
            
            # == TRANSFORM ==
            if col in stats:
                mean = stats[col]['mean']
                std = stats[col]['std']
                df_clean[col] = (df_clean[col] - mean) / std
                
                # Clip extreme outliers (Robust Z-Score)
                df_clean[col] = df_clean[col].clip(-10, 10)

        # 4. Fill NaNs in general
        # Especially target column which wasn't normalized
        df_clean = df_clean.fillna(0.0)
        
        return df_clean, stats

def train_layer_1_tft(df_train: pd.DataFrame, df_val: pd.DataFrame, epochs: int = 1) -> Optional[TFTPredictor]:
    """Train TFT model (Phase 1)"""
    log.info("=== Phase 2.1: Training The Predictor (TFT) ===")
    
    try:
        model_dir = Path("model_institutional")
        model_dir.mkdir(exist_ok=True)
        
        predictor = TFTPredictor(
            max_encoder_length=60,
            max_prediction_length=7,
            hidden_size=64,
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=min(256, len(df_train) // 10)
        )
        
        log.info("📊 Preparing data for TFT...")
        train_ds = predictor.prepare_data(df_train)
        predictor.build_model(train_ds)
        
        log.info(f"🚀 Training TFT for {epochs} epochs...")
        predictor.train(max_epochs=epochs)
        
        model_path = model_dir / "best_sharpe_model.pth"
        predictor.save(str(model_path))
        log.info(f"💾 TFT model saved to: {model_path}")
        
        if df_val is not None:
            try:
                log.info("🧪 Running Validation...")
                val_sample = df_val.iloc[:200] if len(df_val) > 200 else df_val
                val_pred = predictor.predict(val_sample)
                log.info(f"✅ Validation predictions generated. Mean: {val_pred.mean():.6f}")
            except Exception as e:
                log.warning(f"⚠️  Validation failed: {e}")
        
        return predictor
    except Exception as e:
        log.error(f"❌ TFT training failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def generate_predictions(predictor: TFTPredictor, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from TFT model"""
    log.info("🔮 Generating forecasts for RL context...")
    try:
        chunk_size = 10000
        all_preds = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            if len(chunk) == 0: continue
            
            # Predict
            preds = predictor.predict(chunk) # Uses .predict method which handles batching internally now? 
            # Note: The TFTPredictor.predict we wrote handles small batches. 
            # If predictor.predict_batch exists use it, else predict.
            # Assuming standard .predict returns numpy array.
            
            # Weighted average if multi-step
            if preds.ndim > 1:
                n_steps = min(4, preds.shape[1])
                weights = np.exp(-np.arange(n_steps) * 0.5)
                weights = weights / weights.sum()
                avg_preds = np.average(preds[:, :n_steps], axis=1, weights=weights)
            else:
                avg_preds = preds
            all_preds.append(avg_preds)
            
            if i % (chunk_size * 5) == 0 and i > 0:
                log.info(f"📊 Processed {i:,} rows...")
        
        final_preds = np.concatenate(all_preds) if all_preds else np.array([])
        
        # Nan Safety
        if np.isnan(final_preds).any():
            final_preds = np.nan_to_num(final_preds, nan=0.0)

        # Padding/Trimming to match DF length
        if len(final_preds) < len(df):
            pad_len = len(df) - len(final_preds)
            padding = np.zeros(pad_len)
            final_preds = np.concatenate([padding, final_preds])
        elif len(final_preds) > len(df):
            final_preds = final_preds[:len(df)]
            
        return final_preds
    except Exception as e:
        log.error(f"❌ Prediction failed: {e}")
        return np.zeros(len(df))

def add_rl_features(df: pd.DataFrame, predictor: TFTPredictor) -> pd.DataFrame:
    """Enriches dataframe with TFT forecasts and derived stats"""
    try:
        forecasts = generate_predictions(predictor, df)
        
        df_rl = df.copy()
        df_rl["feature_forecast"] = forecasts
        
        # Derived stats
        df_rl["forecast_ma"] = df_rl["feature_forecast"].rolling(20, min_periods=1).mean()
        df_rl["forecast_std"] = df_rl["feature_forecast"].rolling(20, min_periods=1).std()
        df_rl["forecast_change"] = df_rl["feature_forecast"].diff().fillna(0.0)
        
        # Normalize the new forecast features (Z-Score)
        for col in ["feature_forecast", "forecast_ma", "forecast_std"]:
            mean = df_rl[col].mean()
            std = df_rl[col].std()
            if std != 0:
                df_rl[col] = (df_rl[col] - mean) / std
            df_rl[col] = df_rl[col].clip(-5, 5)
            
        # ✅ FIX: Fill NaNs only in numeric columns
        numeric_cols = df_rl.select_dtypes(include=[np.number]).columns
        df_rl[numeric_cols] = df_rl[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        return df_rl
    except Exception as e:
        log.error(f"❌ Failed to add RL features: {e}")
        return df

def train_layer_2_rl(df: pd.DataFrame, predictor: TFTPredictor, timesteps: int = 10000) -> Optional[RLAgent]:
    """Train RL agent (Phase 2)"""
    log.info("=== Phase 2.2: Training The Strategist (PPO) ===")
    try:
        df_rl = add_rl_features(df, predictor)
        model_path = "model_institutional/ppo_agent_v1"
        agent = RLAgent(df_rl, model_path=model_path)
        
        log.info(f"🎮 Training PPO agent for {timesteps:,} timesteps...")
        agent.train(total_timesteps=timesteps)
        return agent
    except Exception as e:
        log.error(f"❌ RL training failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--skip-tft", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    if args.quick:
        args.steps = max(1000, args.steps // 10)
        args.epochs = max(1, args.epochs // 2)
    
    log.info("🚀 Starting Hybrid Training Pipeline")
    
    df_raw = DataLoader.load_data(args.data)
    if df_raw is None: return
    
    # Split
    split_idx = int(len(df_raw) * 0.8)
    train_raw = df_raw.iloc[:split_idx].copy()
    val_raw = df_raw.iloc[split_idx:].copy()
    
    # Preprocess with Normalization
    train_df, train_stats = DataLoader._preprocess_data(train_raw)
    val_df, _ = DataLoader._preprocess_data(val_raw, train_stats=train_stats)
    log.info("✅ Data Normalized (Z-Score applied)")
    
    # Train TFT
    tft_model = None
    if not args.skip_tft:
        tft_model = train_layer_1_tft(train_df, val_df, epochs=args.epochs)
    else:
        try:
            tft_model = TFTPredictor()
            tft_model.load("model_institutional/best_sharpe_model.pth")
            log.info("✅ Loaded existing TFT model")
        except: pass
        
    # Train RL
    rl_agent = None
    if not args.skip_rl:
        if tft_model is None:
             log.warning("⚠️ No TFT model. RL training will be ineffective.")
        rl_agent = train_layer_2_rl(train_df, tft_model, timesteps=args.steps)
    
    # Backtest on Validation
    if rl_agent and not args.skip_rl:
        log.info("=== Final Validation Backtest ===")
        val_df_rl = add_rl_features(val_df, tft_model)
        res = rl_agent.backtest(df=val_df_rl)
        log.info(f"📊 Validation Results: {res}")
    
    log.info("✅ DONE")

if __name__ == "__main__":
    main()