# --- detla-bot/train_hybrid.py ---
# 🧠 HYBRID TRAINER (TFT + PPO) - WORLD CLASS EDITION
# Trains the "Eyes" (TFT) to predict price, and the "Brain" (PPO) to trade it.
# Features: Order Flow (CVD), Multi-Timeframe Context, Robust Data Loading.

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
    
    OPTIONAL_COLUMNS = [
        "price", "close", "high", "low", "volume",  # Price data if available
        "signal_strength", "market_regime"          # Additional features
    ]
    
    @staticmethod
    def find_data_file(search_paths=None) -> Optional[Path]:
        """Find the most appropriate data file"""
        if search_paths is None:
            search_paths = [
                "fused_data_real_FULL.csv",
                "fused_data_real.csv",
                "fused_data_sample.csv",
                "data/fused_data_real_FULL.csv",
                "data/fused_data_real.csv",
                "data/processed/fused_data_real_FULL.csv",
                "dataset/fused_data_real_FULL.csv"
            ]
        
        for path in search_paths:
            p = Path(path)
            if p.exists():
                log.info(f"📂 Found data file: {p} (size: {p.stat().st_size / 1024 / 1024:.2f} MB)")
                return p
        
        return None
    
    @staticmethod
    def load_data(path: str = None) -> Optional[pd.DataFrame]:
        """Load and preprocess data with fallback options"""
        
        # Auto-detect data file if not specified
        if path is None:
            data_file = DataLoader.find_data_file()
            if data_file is None:
                log.error("❌ No data file found. Please run generate_training_data.py first.")
                log.info("💡 To create training data, run: python generate_training_data.py")
                return None
            path = str(data_file)
        else:
            data_file = Path(path)
            if not data_file.exists():
                log.error(f"❌ Specified data file not found: {path}")
                alt_file = DataLoader.find_data_file()
                if alt_file:
                    log.info(f"💡 Using alternative file: {alt_file}")
                    path = str(alt_file)
                else:
                    return None
        
        try:
            log.info(f"📊 Loading data from: {path}")
            
            # Try to load with memory optimization
            try:
                # First, check file size and columns
                sample_df = pd.read_csv(path, nrows=1000)
                columns = sample_df.columns.tolist()
                
                # Determine dtypes for memory efficiency
                dtype_dict = {}
                for col in columns:
                    if sample_df[col].dtype == 'object':
                        try:
                            pd.to_datetime(sample_df[col].head())
                            dtype_dict[col] = 'str' 
                        except:
                            dtype_dict[col] = 'category'
                    elif sample_df[col].dtype == 'float64':
                        dtype_dict[col] = 'float32'
                    elif sample_df[col].dtype == 'int64':
                        dtype_dict[col] = 'int32'
                
                df = pd.read_csv(path, dtype=dtype_dict)
                log.info("✅ Loaded with optimized dtypes for memory efficiency")
                
            except MemoryError:
                log.warning("⚠️  Memory error, loading without dtype optimization")
                df = pd.read_csv(path)
            
            # Parse timestamp if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                log.info(f"📈 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                log.info(f"📊 Total duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
            
            # Validate and preprocess
            # Time index for TFT (CRITICAL) - Generate Globally to ensure continuity
            if "time_idx" not in df.columns:
                if "timestamp" in df.columns:
                    df["time_idx"] = np.arange(len(df))
                else:
                    df["time_idx"] = np.arange(len(df))
                    df["timestamp"] = pd.date_range(start='2023-01-01', periods=len(df), freq='5min')
                    log.warning("⚠️  No timestamp found, created synthetic timestamps")
            
            df["time_idx"] = df["time_idx"].astype(int)
            
            log.info(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
            log.info(f"📋 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return df
            
        except Exception as e:
            log.error(f"❌ Failed to load data: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None
    
    @staticmethod
    def _preprocess_data(df: pd.DataFrame, train_stats: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """Clean and prepare data for training."""
        
        df_clean = df.copy()
        
        # Ensure all required columns exist
        for col in DataLoader.REQUIRED_COLUMNS:
            if col not in df_clean.columns:
                df_clean[col] = 0.0
                log.warning(f"⚠️  Missing required column '{col}', filled with zeros")
        
        # Fill NaN values (FFILL is Causal)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(0.0)
        
        # Stats dictionary
        stats = {} if train_stats is None else train_stats
        
        # Handle infinite values & Clipping
        for col in numeric_cols:
            inf_mask = np.isinf(df_clean[col])
            if inf_mask.any():
                df_clean.loc[inf_mask, col] = 0.0
            
            # Clip extreme values (Robust logic)
            if col in df_clean.columns:
                if train_stats is None:
                    # == FIT MODE (TRAIN) ==
                    if df_clean[col].std() > 0 and len(df_clean[col].unique()) > 10:
                        Q1 = df_clean[col].quantile(0.05)
                        Q3 = df_clean[col].quantile(0.95)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        stats[col] = {
                            'lower': lower_bound,
                            'upper': upper_bound
                        }
                
                # == TRANSFORM MODE (APPLY) ==
                if col in stats:
                     lower = stats[col]['lower']
                     upper = stats[col]['upper']
                     if not (np.isinf(lower) or np.isinf(upper)):
                        clip_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                        if clip_mask.any():
                            df_clean[col] = df_clean[col].clip(lower, upper)
        
        # Final Sweep: Force fill ANY remaining NaNs or Infs
        nan_count = df_clean.isna().sum().sum()
        if nan_count > 0:
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(0.0)
            
        inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            df_clean = df_clean.replace([np.inf, -np.inf], 0.0)
        
        return df_clean, stats

def train_layer_1_tft(df_train: pd.DataFrame, df_val: pd.DataFrame, epochs: int = 1) -> Optional[TFTPredictor]:
    """Train TFT model (Phase 1) with Validation"""
    log.info("=== Phase 2.1: Training The Predictor (TFT) ===")
    
    try:
        model_dir = Path("model_institutional")
        model_dir.mkdir(exist_ok=True)
        
        min_samples = 1000
        if len(df_train) < min_samples:
            log.warning(f"⚠️  Limited data: only {len(df_train)} samples (minimum recommended: {min_samples})")
        
        # Initialize predictor
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
        
        if hasattr(train_ds, '__len__'):
            log.info(f"📦 Training dataset size: {len(train_ds):,} samples")
        
        log.info("🏗️ Building TFT model...")
        predictor.build_model(train_ds)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("🧹 Cleared GPU memory before training")
        
        log.info(f"🚀 Training TFT for {epochs} epochs...")
        predictor.train(max_epochs=epochs)
        
        model_path = model_dir / "best_sharpe_model.pth"
        predictor.save(str(model_path))
        log.info(f"💾 TFT model saved to: {model_path}")
        
        # Validation Evaluation
        if df_val is not None:
            try:
                log.info("🧪 Running Validation...")
                val_sample = df_val.iloc[:200] if len(df_val) > 200 else df_val
                val_pred = predictor.predict(val_sample)
                log.info(f"✅ Validation predictions generated. Shape: {val_pred.shape}")
            except Exception as e:
                log.warning(f"⚠️  Validation failed: {e}")
        
        return predictor
        
    except Exception as e:
        log.error(f"❌ TFT training failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def generate_predictions(predictor: TFTPredictor, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from TFT model with proper alignment"""
    log.info("🔮 Generating forecasts for RL context...")
    
    try:
        if predictor is None:
            raise ValueError("Predictor is None")
        
        chunk_size = 10000
        all_preds = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            if len(chunk) == 0: continue
                
            if hasattr(predictor, 'predict_batch') and callable(predictor.predict_batch):
                preds = predictor.predict_batch(chunk)
            elif hasattr(predictor, 'predict') and callable(predictor.predict):
                preds = predictor.predict(chunk)
            else:
                raise AttributeError("Predictor has no valid prediction method")
            
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
        
        if np.isnan(final_preds).any():
            nan_count = np.isnan(final_preds).sum()
            log.warning(f"⚠️  Predictions contain {nan_count} NaNs! Replaced with 0.0.")
            final_preds = np.nan_to_num(final_preds, nan=0.0)

        if len(final_preds) < len(df):
            pad_length = len(df) - len(final_preds)
            padding = np.full(pad_length, final_preds[0] if len(final_preds) > 0 else 0.0) if len(final_preds) > 0 else np.zeros(pad_length)
            final_preds = np.concatenate([padding, final_preds])
            log.info(f"📏 Padded predictions: added {pad_length} values")
        elif len(final_preds) > len(df):
            final_preds = final_preds[:len(df)]
            log.info(f"📏 Trimmed predictions to match data length")
        
        if len(final_preds) > 10:
            try:
                from scipy.ndimage import gaussian_filter1d
                final_preds = gaussian_filter1d(final_preds, sigma=1.0)
                log.debug("🔁 Applied light smoothing to predictions")
            except ImportError:
                log.warning("⚠️  scipy not found, skipping smoothing")
        
        log.info(f"📊 Prediction stats - Mean: {final_preds.mean():.6f}, Std: {final_preds.std():.6f}")
        return final_preds
        
    except Exception as e:
        log.error(f"❌ Prediction generation failed: {e}")
        return np.zeros(len(df))

# ✅ NEW HELPER FUNCTION TO FIX THE CRASH
def add_rl_features(df: pd.DataFrame, predictor: TFTPredictor) -> pd.DataFrame:
    """Enriches a dataframe with TFT forecasts and derived stats for the RL Agent"""
    try:
        forecasts = generate_predictions(predictor, df)
        
        df_rl = df.copy()
        df_rl["feature_forecast"] = forecasts
        df_rl["forecast_ma"] = df_rl["feature_forecast"].rolling(20, min_periods=1).mean()
        df_rl["forecast_std"] = df_rl["feature_forecast"].rolling(20, min_periods=1).std()
        df_rl["forecast_change"] = df_rl["feature_forecast"].diff().fillna(0.0)
        
        norm_window = 1000
        for col in ["feature_forecast", "forecast_ma", "forecast_std"]:
            if df_rl[col].std() > 0:
                roll_mean = df_rl[col].rolling(norm_window, min_periods=1).mean()
                roll_std = df_rl[col].rolling(norm_window, min_periods=1).std().replace(0, 1)
                df_rl[col] = (df_rl[col] - roll_mean) / roll_std
                df_rl[col] = df_rl[col].clip(-5, 5)
        
        # ✅ FIX: Only fill NaNs in NUMERIC columns to avoid Categorical crash
        numeric_cols = df_rl.select_dtypes(include=[np.number]).columns
        df_rl[numeric_cols] = df_rl[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        return df_rl
    except Exception as e:
        log.error(f"❌ Failed to add RL features: {e}")
        return df # Return original if failed

def train_layer_2_rl(df: pd.DataFrame, predictor: TFTPredictor, timesteps: int = 10000) -> Optional[RLAgent]:
    """Train RL agent (Phase 2)"""
    log.info("=== Phase 2.2: Training The Strategist (PPO) ===")
    
    try:
        # ✅ Use the new helper function
        df_rl = add_rl_features(df, predictor)
        log.info(f"📊 Enhanced RL dataset with {len(df_rl.columns)} columns")
        
        model_dir = Path("model_institutional")
        model_path = model_dir / "ppo_agent_v1"
        
        agent = RLAgent(df_rl, model_path=str(model_path))
        
        log.info(f"🎮 Training PPO agent for {timesteps:,} timesteps...")
        agent.train(total_timesteps=timesteps)
        
        log.info(f"💾 RL agent saved to: {model_path}")
        return agent
        
    except Exception as e:
        log.error(f"❌ RL training failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def run_backtest(agent: RLAgent, val_df: pd.DataFrame = None):
    """Run backtest with error handling"""
    log.info("=== Verification: Running Backtest ===")
    try:
        # If val_df is provided, we tell the agent to switch env (if supported)
        # Note: RLAgent.backtest() in your provided file accepts a 'df' argument!
        results = agent.backtest(df=val_df)
        
        if results is not None and isinstance(results, dict):
            log.info("📈 ===== BACKTEST RESULTS =====")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    log.info(f"  {key}: {value:.4f}")
                else:
                    log.info(f"  {key}: {value}")
            log.info("=" * 35)
        else:
            log.info(f"📊 Backtest results: {results}")
        log.info("✅ Backtest completed successfully")
    except Exception as e:
        log.error(f"❌ Backtest failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Hybrid TFT+PPO Training Pipeline")
    parser.add_argument("--steps", type=int, default=10000, help="RL Training Steps")
    parser.add_argument("--epochs", type=int, default=1, help="TFT Training Epochs")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--skip-tft", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick:
        args.steps = max(1000, args.steps // 10)
        args.epochs = max(1, args.epochs // 2)
        log.info(f"⚡ Quick mode enabled: TFT epochs={args.epochs}, RL steps={args.steps}")
    
    Path("logs").mkdir(exist_ok=True)
    Path("model_institutional").mkdir(exist_ok=True)
    
    log.info("🚀 Starting Hybrid Training Pipeline")
    
    if args.validate_only:
        df = DataLoader.load_data(args.data)
        if df is not None:
            log.info(f"✅ Data Valid. Shape: {df.shape}")
        return
    
    df_raw = DataLoader.load_data(args.data)
    if df_raw is None: return
    
    # === DATA SPLITTING & PREPROCESSING ===
    split_idx = int(len(df_raw) * 0.8)
    train_raw = df_raw.iloc[:split_idx].copy()
    val_raw = df_raw.iloc[split_idx:].copy()
    
    log.info(f"✂️  Splitting Data: Train={len(train_raw):,} rows, Val={len(val_raw):,} rows")
    
    # 1. Preprocess TRAIN (Fit stats)
    train_df, train_stats = DataLoader._preprocess_data(train_raw)
    
    # 2. Preprocess VAL (Apply stats)
    val_df, _ = DataLoader._preprocess_data(val_raw, train_stats=train_stats)
    
    log.info("✅ Preprocessing Complete (Leakage-Free)")
    
    tft_model = None
    if not args.skip_tft:
        tft_model = train_layer_1_tft(train_df, val_df, epochs=args.epochs)
    else:
        log.info("⏭️  Skipping TFT training as requested")
        model_path = Path("model_institutional/best_sharpe_model.pth")
        if model_path.exists():
            try:
                tft_model = TFTPredictor()
                tft_model.load(str(model_path))
                log.info(f"✅ Loaded existing TFT model from {model_path}")
            except Exception as e:
                log.warning(f"⚠️  Could not load existing TFT model: {e}")
    
    rl_agent = None
    if not args.skip_rl:
        if tft_model is None:
             try:
                 tft_model = TFTPredictor()
                 tft_model.load("model_institutional/best_sharpe_model.pth")
                 log.info("✅ Loaded existing TFT model for RL training")
             except:
                 log.warning("⚠️ No TFT model found. Training RL on raw features (less effective).")
        
        # Use TRAIN data for RL training
        rl_agent = train_layer_2_rl(train_df, tft_model, timesteps=args.steps)
    else:
        log.info("⏭️  Skipping RL training as requested")
    
    if rl_agent:
        # ✅ World Class Validation: Use VAL data for backtest
        if tft_model is not None and val_df is not None:
             log.info("🧪 Enhancing Validation Data for Backtest...")
             val_df_rl = add_rl_features(val_df, tft_model)
             run_backtest(rl_agent, val_df=val_df_rl)
        else:
             # Fallback
             run_backtest(rl_agent)
    
    log.info("✅ HYBRID TRAINING COMPLETE!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("⚠️ Training interrupted by user")
    except Exception as e:
        log.error(f"❌ Fatal error: {e}")