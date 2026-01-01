# --- detla-bot/train_hybrid.py ---
# 🧠 HYBRID TRAINER (TFT + PPO) - FIXED
# Trains the "Eyes" (TFT) to predict price, and the "Brain" (PPO) to trade it.

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
        "trend_bias_4h", "volatility_4h", "rsi_4h"
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
                # Suggest how to create data
                log.info("💡 To create training data, run: python generate_training_data.py")
                return None
            path = str(data_file)
        else:
            data_file = Path(path)
            if not data_file.exists():
                log.error(f"❌ Specified data file not found: {path}")
                # Try to find alternative
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
                        # Check if it's actually a datetime
                        try:
                            pd.to_datetime(sample_df[col].head())
                            dtype_dict[col] = 'str'  # Will parse as datetime later
                        except:
                            dtype_dict[col] = 'category'
                    elif sample_df[col].dtype == 'float64':
                        dtype_dict[col] = 'float32'
                    elif sample_df[col].dtype == 'int64':
                        dtype_dict[col] = 'int32'
                
                # Load full dataset with optimized dtypes
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
            df = DataLoader._preprocess_data(df)
            
            log.info(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
            log.info(f"📋 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return df
            
        except Exception as e:
            log.error(f"❌ Failed to load data: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None
    
    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for training"""
        
        df_clean = df.copy()
        
        # Store original column count for logging
        original_cols = len(df_clean.columns)
        
        # Ensure all required columns exist
        missing_cols = []
        for col in DataLoader.REQUIRED_COLUMNS:
            if col not in df_clean.columns:
                missing_cols.append(col)
                df_clean[col] = 0.0
                log.warning(f"⚠️  Missing required column '{col}', filled with zeros")
        
        # Add optional columns if they exist
        for col in DataLoader.OPTIONAL_COLUMNS:
            if col in df_clean.columns and col not in DataLoader.REQUIRED_COLUMNS:
                log.debug(f"➕ Optional column '{col}' found")
        
        if missing_cols:
            log.warning(f"⚠️  Total missing required columns: {len(missing_cols)}")
        
        # Fill NaN values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(0.0)
        
        # Handle infinite values
        for col in numeric_cols:
            # Replace infinities
            inf_mask = np.isinf(df_clean[col])
            if inf_mask.any():
                df_clean.loc[inf_mask, col] = 0.0
                log.info(f"🔄 Replaced {inf_mask.sum()} infinite values in column '{col}'")
            
            # Clip extreme values (CAUSAL VERSION)
            if df_clean[col].std() > 0:
                # Use Rolling Z-Score to detect outliers without future leakage
                # Window = 288 (24h of 5m data) or larger
                window = 1000
                roll_mean = df_clean[col].rolling(window, min_periods=1).mean()
                roll_std = df_clean[col].rolling(window, min_periods=1).std().replace(0, 1)
                
                # Calculate Z-Scores
                z_score = (df_clean[col] - roll_mean) / roll_std
                
                # Clip Z-Scores to +/- 4 and reconstruct
                # This ensures we only use PAST data to decide if current point is outlier
                clipped_z = z_score.clip(-4, 4)
                df_clean[col] = roll_mean + (clipped_z * roll_std)
                
                # Log if changes happened (approximate check)
                n_clipped = (z_score.abs() > 4).sum()
                if n_clipped > 0:
                     log.debug(f"📏 Rolled-Clipped {n_clipped} outliers in column '{col}'")
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_len:
            log.info(f"🧹 Removed {initial_len - len(df_clean):,} duplicate rows")
        
        # Time index for TFT (CRITICAL)
        if "time_idx" not in df_clean.columns:
            if "timestamp" in df_clean.columns:
                # Create monotonic time index
                df_clean["time_idx"] = np.arange(len(df_clean))
                log.info("ℹ️  Generated 'time_idx' for TFT model")
            else:
                # Create synthetic time index
                df_clean["time_idx"] = np.arange(len(df_clean))
                df_clean["timestamp"] = pd.date_range(
                    start='2023-01-01', 
                    periods=len(df_clean), 
                    freq='5min'
                )
                log.warning("⚠️  No timestamp found, created synthetic timestamps")
        
        # Ensure time_idx is integer
        df_clean["time_idx"] = df_clean["time_idx"].astype(int)
        
        # Final Sweep: Force fill ANY remaining NaNs or Infs
        nan_count = df_clean.isna().sum().sum()
        if nan_count > 0:
            log.warning(f"⚠️  Data still contains {nan_count} NaN values. Force filling numeric cols with 0.")
            # Only fill numeric columns to avoid Categorical errors
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(0.0)
            
        inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            log.warning(f"⚠️  Data still contains {inf_count} infinite values. Force filling with 0.")
            df_clean = df_clean.replace([np.inf, -np.inf], 0.0)
        
        log.info(f"📊 Preprocessing complete: {original_cols} → {len(df_clean.columns)} columns")
        
        return df_clean

def train_layer_1_tft(df: pd.DataFrame, epochs: int = 1) -> Optional[TFTPredictor]:
    """Train TFT model (Phase 1)"""
    
    log.info("=== Phase 2.1: Training The Predictor (TFT) ===")
    
    try:
        # Create model directory
        model_dir = Path("model_institutional")
        model_dir.mkdir(exist_ok=True)
        
        # Check if we have enough data
        min_samples = 1000  # Minimum samples for meaningful training
        if len(df) < min_samples:
            log.warning(f"⚠️  Limited data: only {len(df)} samples (minimum recommended: {min_samples})")
        
        # Initialize predictor with optimized parameters
        # Note: Parameters are hardcoded in TFTPredictor/TimeSeriesTransformer for now
        predictor = TFTPredictor(
            max_encoder_length=60,
            max_prediction_length=7
        )
        
        # Prepare data
        log.info("📊 Preparing data for TFT...")
        train_ds = predictor.prepare_data(df)
        
        # Log dataset info
        if hasattr(train_ds, '__len__'):
            log.info(f"📦 Training dataset size: {len(train_ds):,} samples")
        
        # Build model
        log.info("🏗️ Building TFT model...")
        predictor.build_model(train_ds)
        
        # Clear memory before training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("🧹 Cleared GPU memory before training")
        
        # Train with progress tracking
        log.info(f"🚀 Training TFT for {epochs} epochs...")
        predictor.train(max_epochs=epochs)
        
        # Save model
        model_path = model_dir / "best_sharpe_model.pth"
        predictor.save(str(model_path))
        log.info(f"💾 TFT model saved to: {model_path}")
        
        # Test prediction on a small sample
        try:
            test_sample = df.iloc[-100:]  # Last 100 samples
            test_pred = predictor.predict(test_sample)
            log.info(f"🧪 Test prediction shape: {test_pred.shape}")
        except Exception as e:
            log.warning(f"⚠️  Test prediction failed: {e}")
        
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
        # Check if predictor is valid
        if predictor is None:
            raise ValueError("Predictor is None")
        
        # Generate predictions in chunks to avoid memory issues
        chunk_size = 10000
        all_preds = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            if len(chunk) == 0:
                continue
                
            if hasattr(predictor, 'predict_batch') and callable(predictor.predict_batch):
                preds = predictor.predict_batch(chunk)
            elif hasattr(predictor, 'predict') and callable(predictor.predict):
                preds = predictor.predict(chunk)
            else:
                raise AttributeError("Predictor has no valid prediction method")
            
            # Process predictions
            if preds.ndim > 1:
                # Weighted average with emphasis on near-term predictions
                n_steps = min(4, preds.shape[1])
                weights = np.exp(-np.arange(n_steps) * 0.5)  # Exponential decay
                weights = weights / weights.sum()
                avg_preds = np.average(preds[:, :n_steps], axis=1, weights=weights)
            else:
                avg_preds = preds
            
            all_preds.append(avg_preds)
            
            if i % (chunk_size * 5) == 0 and i > 0:
                log.info(f"📊 Processed {i:,} rows...")
        
        # Combine all predictions
        final_preds = np.concatenate(all_preds) if all_preds else np.array([])
        
        # SAFETY: Replace NaNs with 0 (Fixes the crash from untrained models)
        if np.isnan(final_preds).any():
            nan_count = np.isnan(final_preds).sum()
            log.warning(f"⚠️  Predictions contain {nan_count} NaNs! Replaced with 0.0.")
            final_preds = np.nan_to_num(final_preds, nan=0.0)

        # Handle alignment
        if len(final_preds) < len(df):
            # TFT predictions typically start after encoder length
            pad_length = len(df) - len(final_preds)
            # Use forward fill for padding (last available prediction)
            if len(final_preds) > 0:
                padding = np.full(pad_length, final_preds[0] if len(final_preds) > 0 else 0.0)
            else:
                padding = np.zeros(pad_length)
            final_preds = np.concatenate([padding, final_preds])
            log.info(f"📏 Padded predictions: added {pad_length} values")
        elif len(final_preds) > len(df):
            final_preds = final_preds[:len(df)]
            log.info(f"📏 Trimmed predictions to match data length")
        
        # Smooth predictions slightly
        if len(final_preds) > 10:
            try:
                from scipy.ndimage import gaussian_filter1d
                final_preds = gaussian_filter1d(final_preds, sigma=1.0)
                log.debug("🔁 Applied light smoothing to predictions")
            except ImportError:
                log.warning("⚠️  scipy not found, skipping smoothing")
        
        # Log prediction statistics
        log.info(f"📊 Prediction stats - Mean: {final_preds.mean():.6f}, "
                 f"Std: {final_preds.std():.6f}, "
                 f"Range: [{final_preds.min():.6f}, {final_preds.max():.6f}]")
        
        return final_preds
        
    except Exception as e:
        log.error(f"❌ Prediction generation failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        # Return zero predictions as fallback
        return np.zeros(len(df))

def train_layer_2_rl(df: pd.DataFrame, predictor: TFTPredictor, timesteps: int = 10000) -> Optional[RLAgent]:
    """Train RL agent (Phase 2)"""
    
    log.info("=== Phase 2.2: Training The Strategist (PPO) ===")
    
    try:
        # Generate forecasts
        forecasts = generate_predictions(predictor, df)
        
        # Create enhanced dataframe for RL
        df_rl = df.copy()
        df_rl["feature_forecast"] = forecasts
        
        # Add derived features from forecast
        df_rl["forecast_ma"] = df_rl["feature_forecast"].rolling(20, min_periods=1).mean()
        df_rl["forecast_std"] = df_rl["feature_forecast"].rolling(20, min_periods=1).std()
        df_rl["forecast_change"] = df_rl["feature_forecast"].diff().fillna(0.0)
        
        # Normalize forecast features (CAUSAL VERSION)
        # Use rolling window to ensure we don't leak future distribution stats
        norm_window = 1000 # Align with preprocessing window
        for col in ["feature_forecast", "forecast_ma", "forecast_std"]:
            if df_rl[col].std() > 0:
                roll_mean = df_rl[col].rolling(norm_window, min_periods=1).mean()
                roll_std = df_rl[col].rolling(norm_window, min_periods=1).std().replace(0, 1)
                df_rl[col] = (df_rl[col] - roll_mean) / roll_std
                
                # Clip to reasonable range again (safe now as it's local z-score)
                df_rl[col] = df_rl[col].clip(-5, 5)
        
        log.info(f"📊 Enhanced RL dataset with {len(df_rl.columns)} columns")
        
        # Initialize and train RL agent
        model_dir = Path("model_institutional")
        model_path = model_dir / "ppo_agent_v1"
        
        agent = RLAgent(df_rl, model_path=str(model_path))
        
        log.info(f"🎮 Training PPO agent for {timesteps:,} timesteps...")
        
        # ✅ FIX: Removed callback argument that was causing crash
        agent.train(total_timesteps=timesteps)
        
        log.info(f"💾 RL agent saved to: {model_path}")
        return agent
        
    except Exception as e:
        log.error(f"❌ RL training failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def run_backtest(agent: RLAgent):
    """Run backtest with error handling"""
    
    log.info("=== Verification: Running Backtest ===")
    
    try:
        results = agent.backtest()
        
        if results is not None:
            if isinstance(results, dict):
                log.info("📈 ===== BACKTEST RESULTS =====")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        # Format numbers nicely
                        if abs(value) >= 1e6:
                            log.info(f"  {key}: {value:,.2f}")
                        elif abs(value) < 0.01:
                            log.info(f"  {key}: {value:.6f}")
                        else:
                            log.info(f"  {key}: {value:.4f}")
                    else:
                        log.info(f"  {key}: {value}")
                log.info("=" * 35)
            else:
                log.info(f"📊 Backtest results: {results}")
        
        log.info("✅ Backtest completed successfully")
        
    except Exception as e:
        log.error(f"❌ Backtest failed: {e}")
        import traceback
        log.error(traceback.format_exc())

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description="Hybrid TFT+PPO Training Pipeline")
    parser.add_argument("--steps", type=int, default=10000, help="RL Training Steps")
    parser.add_argument("--epochs", type=int, default=1, help="TFT Training Epochs")
    parser.add_argument("--data", type=str, default=None, 
                        help="Path to training data (auto-detects if not specified)")
    parser.add_argument("--skip-tft", action="store_true", help="Skip TFT training")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training")
    parser.add_argument("--quick", action="store_true", help="Quick training mode (reduced steps)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't train")
    
    args = parser.parse_args()
    
    # Apply quick mode
    if args.quick:
        args.steps = max(1000, args.steps // 10)
        args.epochs = max(1, args.epochs // 2)
        log.info(f"⚡ Quick mode enabled: TFT epochs={args.epochs}, RL steps={args.steps}")
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("model_institutional").mkdir(exist_ok=True)
    
    log.info("🚀 Starting Hybrid Training Pipeline")
    log.info(f"⚙️  Configuration: TFT epochs={args.epochs}, RL steps={args.steps}")
    log.info(f"📂 Working directory: {Path.cwd()}")
    
    # Validate-only mode
    if args.validate_only:
        log.info("🔍 Running in validation-only mode...")
        df = DataLoader.load_data(args.data)
        if df is not None:
            log.info("✅ Data validation passed")
            log.info(f"📊 Data shape: {df.shape}")
            log.info(f"📈 Columns: {list(df.columns)}")
        return
    
    # 1. Load Data
    log.info("📥 Step 1: Loading data...")
    df = DataLoader.load_data(args.data)
    if df is None:
        log.error("❌ Failed to load data. Exiting.")
        return
    
    # 2. Train Layer 1 (TFT)
    tft_model = None
    if not args.skip_tft:
        tft_model = train_layer_1_tft(df, epochs=args.epochs)
        if tft_model is None:
            log.warning("⚠️  TFT training failed, but continuing with RL training...")
    else:
        log.info("⏭️  Skipping TFT training as requested")
        # Try to load existing TFT model
        model_path = Path("model_institutional/best_sharpe_model.pth")
        if model_path.exists():
            try:
                # Try different loading strategies
                if hasattr(TFTPredictor, 'load_classmethod'):
                    tft_model = TFTPredictor.load(str(model_path))
                else:
                    # Create new instance and load weights
                    tft_model = TFTPredictor()
                    tft_model.load(str(model_path))
                log.info(f"✅ Loaded existing TFT model from {model_path}")
            except Exception as e:
                log.warning(f"⚠️  Could not load existing TFT model: {e}")
        else:
            log.warning(f"⚠️  No existing TFT model found at {model_path}")
    
    # 3. Train Layer 2 (RL)
    rl_agent = None
    if not args.skip_rl:
        if tft_model is not None:
            rl_agent = train_layer_2_rl(df, tft_model, timesteps=args.steps)
        else:
            log.warning("⚠️  No TFT model available, training RL with raw features only")
            # Create RL agent without forecasts
            df_rl = df.copy()
            df_rl["feature_forecast"] = 0.0
            agent = RLAgent(df_rl, model_path="model_institutional/ppo_agent_v1")
            agent.train(total_timesteps=args.steps)
            rl_agent = agent
    else:
        log.info("⏭️  Skipping RL training as requested")
    
    # 4. Run backtest if agent was trained
    if rl_agent is not None:
        run_backtest(rl_agent)
    
    # Summary
    log.info("=" * 50)
    log.info("✅ HYBRID TRAINING COMPLETE!")
    log.info("=" * 50)
    
    # Check what was created
    model_dir = Path("model_institutional")
    if model_dir.exists():
        model_files = list(model_dir.glob("*"))
        log.info(f"📁 Models created ({len(model_files)}):")
        for f in model_files:
            size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
            log.info(f"  • {f.name} ({size_mb:.2f} MB)")
    
    log.info("💡 Next steps:")
    log.info("   1. Test the model: python run_inference.py")
    log.info("   2. Deploy trading bot: python deploy_bot.py")
    log.info("   3. Monitor performance: python monitor_performance.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("⚠️  Training interrupted by user")
    except Exception as e:
        log.error(f"❌ Fatal error in main: {e}")
        import traceback
        log.error(traceback.format_exc())