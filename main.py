# --- detla-bot/main.py ---
import time
import logging
import pandas as pd
import numpy as np
import torch
import json
import os
import sys
from pathlib import Path

# Custom Modules
# Ensure current directory is in path
sys.path.append(os.getcwd())

from historical_data_fetcher import HistoricalDataFetcher # Reuse to get live snapshot
from feature_engine import FeatureEngine
from executor import Executor
from tft_model import TFTPredictor
from rl_agent import RLAgent
from train_hybrid import generate_predictions, DataLoader # Use the generator from training script

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MAIN]: %(message)s")
log = logging.getLogger("live_trader")

# --- CONFIGURATION ---
SYMBOL = "BTCUSD" 
TIMEFRAME = 5 # minutes
BUFFER_SIZE = 2000 # Need enough for 1H/4H resampling and rolling windows

class LiveTrader:
    def __init__(self):
        self.fetcher = HistoricalDataFetcher() # Utilizing your existing fetcher logic
        self.engine = FeatureEngine()
        self.executor = Executor()
        
        # Load Models
        self.load_models()
        
    def load_models(self):
        log.info("🧠 Loading AI Models...")
        try:
            # 1. Load Normalization Stats
            stats_path = "model_institutional/normalization_stats.json"
            if not os.path.exists(stats_path):
                 raise FileNotFoundError(f"{stats_path} not found. Run export_stats.py first.")
                 
            with open(stats_path, "r") as f:
                self.norm_stats = json.load(f)
            log.info("✅ Normalization stats loaded.")

            # 2. Load TFT (Eyes)
            self.tft = TFTPredictor()
            self.tft.load("model_institutional/best_sharpe_model.pth")
            
            # 3. Load PPO (Brain)
            # We only need the actor network for inference
            dummy_df = pd.DataFrame(columns=DataLoader.REQUIRED_COLUMNS) # Schema dummy
            
            # ✅ FIX: Add Forecast Columns that RLAgent expects (added during training via add_rl_features)
            for fc in ["feature_forecast", "forecast_ma", "forecast_std", "forecast_change"]:
                dummy_df[fc] = 0.0
                
            # ✅ FIX: Add 1 row of zeros to prevent Env init crash (IndexError)
            dummy_df.loc[0] = 0.0
            
            self.ppo_agent = RLAgent(dummy_df, model_path="model_institutional/ppo_agent_v1")
            
            log.info("✅ World Class Models Loaded Successfully.")
            
        except Exception as e:
            log.error(f"❌ Failed to load models: {e}")
            raise e

    def prepare_live_data(self):
        """
        World Class Data Fetcher:
        1. Gets Spot Data from Binance (The 'Alpha' source)
        2. Gets Futures Data from Delta (The 'Execution' source)
        3. Fuses them exactly like training data.
        """
        log.info("🌍 Fetching LIVE Hybrid Data (Binance Spot + Delta Futures)...")
        
        try:
            # --- 1. Fetch Binance Spot Data (The "Eyes") ---
            # We need the last N candles to calculate rolling features (CVD, Vol Z-Score)
            spot_df = self.fetcher.fetch_binance_candles_sync(
                symbol="BTCUSDT", 
                interval=str(TIMEFRAME)+"m", 
                limit=BUFFER_SIZE
            )
            
            if spot_df.empty:
                log.warning("⚠️ Binance Spot Data Empty.")
                return None

            # Rename columns to match training schema: 'spot_close', 'spot_volume', 'taker_buy_vol'
            spot_df = spot_df.rename(columns={
                'close': 'spot_close',
                'volume': 'spot_volume',
                # 'taker_buy_vol' is already renamed in fetcher
            })
            
            # --- 2. Fetch Delta Futures Data (The "Target") ---
            futures_df = self.fetcher.fetch_delta_candles_sync(
                symbol="BTCUSD",
                interval=str(TIMEFRAME)+"m",
                limit=BUFFER_SIZE
            )
            
            if futures_df.empty:
                 log.warning("⚠️ Delta Futures Data Empty.")
                 return None
            
            # --- 3. Fuse Data (The "Bridge") ---
            # Merge on timestamp (nearest)
            spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
            futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
            
            # Use merge_asof just like training
            df = pd.merge_asof(
                futures_df.sort_values('timestamp'), 
                spot_df[['timestamp', 'spot_close', 'spot_volume', 'taker_buy_vol']].sort_values('timestamp'),
                on='timestamp', 
                direction='nearest',
                tolerance=pd.Timedelta("1m") # Tight tolerance for live
            )
            
            # --- 4. Add Missing Context (Funding/Macro) ---
            current_funding = self.fetcher.get_current_funding_rate_sync("BTCUSD")
            df['funding_rate'] = float(current_funding)
            
            # Macro is slow. Use constant or fetch daily.
            df['vix_close'] = 15.0 # Placeholder or fetch from API
            df['dxy_close'] = 104.0 # Placeholder
            
            # --- 5. Feature Engineering ---
            # Now we have a DF that looks EXACTLY like 'fused_data_real_FULL.csv'
            df = self.engine.add_features(df)
            
            # === SAFETY: Ensure ALL Required Feature Columns Exist ===
            # If input dim mismatch occurs (e.g. 60x0 error), it's because we have fewer cols than Training.
            for req_col in DataLoader.REQUIRED_COLUMNS:
                if req_col not in df.columns:
                    log.warning(f"⚠️ Missing feature '{req_col}', filling with 0.0")
                    df[req_col] = 0.0
            
            # === NORMALIZATION (CRITICAL) ===
            # Apply saved Mean/Std provided they exist in stats
            for col, stats in self.norm_stats.items():
                if col in df.columns:
                    mean = stats['mean']
                    std = stats['std']
                    # Z-Score
                    df[col] = (df[col] - mean) / std
                    # Clip
                    df[col] = df[col].clip(-10, 10)
            
            # Ensure "close_log_ret" exists for Dataset Exclusion logic (even if 0)
            if "close_log_ret" not in df.columns:
                 df["close_log_ret"] = 0.0

            # Debug: Log schema
            # log.info(f"Live DF Dtypes: \n{df.dtypes}")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            log.info(f"✅ Live Data Prepared: {len(df)} rows. {len(numeric_cols)} numeric columns.")
            # log.info(f"Numeric Columns: {numeric_cols}")
            
            return df
            
        except Exception as e:
            log.error(f"❌ Error preparing live data: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None

    def run_cycle(self):
        """Main decision loop"""
        log.info(f"🚀 Starting Live Trading Cycle for {SYMBOL}")
        while True:
            try:
                log.info(f"⏳ Waiting for {TIMEFRAME}m candle close...")
                
                # 1. Get Data
                df_live = self.prepare_live_data()
                if df_live is None: 
                    time.sleep(10)
                    continue
                
                # 2. Generate Forecast (The Eyes)
                log.info("👁️ TFT Analyzing...")
                
                # We need historical forecasts to calculate Z-Score for PPO
                # Since tft.predict only gives the current forecast, we simulate a rolling buffer
                # by predicting on the last 20 time steps.
                
                recent_forecasts = []
                lookback_window = 30 # Enough to get valid rolling 20 mean/std
                
                # Create rolling windows
                for i in range(lookback_window):
                    # Slice DF to end at t-(lookback-i)
                    # e.g. if len=2000, lookback=30.
                    # i=0 -> end at -30. i=29 -> end at -1 (current).
                    idx = (len(df_live) - (lookback_window - 1) + i)
                    if idx < 60: continue # Need at least seq_len
                    
                    sub_df = df_live.iloc[:idx]
                    
                    try:
                        raw_preds = self.tft.predict(sub_df)
                        # raw_preds shape is (1, 7)
                        
                        # Weighted Average Logic (Same as train_hybrid)
                        if raw_preds.ndim > 1:
                            n_steps = min(4, raw_preds.shape[1])
                            weights = np.exp(-np.arange(n_steps) * 0.5)
                            weights = weights / weights.sum()
                            val = np.average(raw_preds[:, :n_steps], axis=1, weights=weights)[0]
                        else:
                            val = raw_preds[0]
                            
                        recent_forecasts.append(val)
                    except Exception:
                        recent_forecasts.append(0.0)
                        
                if not recent_forecasts:
                     log.warning("⚠️ TFT returned no forecasts. Skipping cycle.")
                     time.sleep(10)
                     continue

                # Current Forecast is the last one
                current_forecast = recent_forecasts[-1]
                
                # Calculate Statistics
                forecast_series = pd.Series(recent_forecasts)
                f_mean = forecast_series.rolling(20, min_periods=1).mean().iloc[-1]
                f_std = forecast_series.rolling(20, min_periods=1).std().iloc[-1]
                
                # Normalize forecast features (Local Z-Score)
                if f_std == 0 or np.isnan(f_std): f_std = 1.0
                norm_forecast = (current_forecast - f_mean) / f_std
                
                log.info(f"🔮 Forecast: {current_forecast:.4f} (Z: {norm_forecast:.2f})")
                
                # 3. Construct State Vector for PPO
                # Get last row of features
                # last_row = df_live.iloc[-1]
                
                # Combine standard features + Forecast features
                # Must match RLAgent observation space order!
                
                # Hack: Create a 1-row DataFrame with all columns
                state_df = df_live.iloc[[-1]].copy()
                state_df['feature_forecast'] = norm_forecast
                state_df['forecast_ma'] = 0 # simplified
                state_df['forecast_std'] = 1 # simplified
                state_df['forecast_change'] = 0 # simplified
                
                # 3. Construct State Vector for PPO
                # Get last row of features
                # last_row = df_live.iloc[-1]
                
                # Combine standard features + Forecast features
                # Must match RLAgent observation space order!
                
                # Hack: Create a 1-row DataFrame with all columns
                state_df = df_live.iloc[[-1]].copy()
                state_df['feature_forecast'] = norm_forecast
                state_df['forecast_ma'] = 0 # simplified
                state_df['forecast_std'] = 1 # simplified
                state_df['forecast_change'] = 0 # simplified
                
                # Get State Vector
                # We need to extract exactly the features the agent was trained on
                feature_cols = self.ppo_agent.env.feature_cols
                
                # Check for missing columns in state_df just in case
                missing = [c for c in feature_cols if c not in state_df.columns]
                if missing:
                    log.warning(f"⚠️ State missing cols: {missing}. Filling 0.")
                    for c in missing: state_df[c] = 0.0
                
                # Construct observation
                obs = state_df[feature_cols].iloc[0].values.astype(np.float32)
                
                # 4. PPO Decision (The Brain)
                action, _ = self.ppo_agent.agent.select_action(obs)
                target_position_size = np.clip(action, -1.0, 1.0)
                
                # --- SNIPER GATING LOGIC ---
                confidence = abs(target_position_size)
                SNIPER_THRESHOLD = 0.75
                
                direction = "LONG" if target_position_size > 0 else "SHORT"
                log.info(f"🧠 Analysis: Signal={target_position_size:.4f} ({direction}) | Confidence={confidence:.2%}")
                
                if confidence < SNIPER_THRESHOLD:
                    log.info(f"💤 Confidence {confidence:.2%} < {SNIPER_THRESHOLD:.0%}. Sniper Holding Fire.")
                    # Stay Flat
                    self.executor.sync_position(SYMBOL, 0.0)
                else:
                    log.info(f"🎯 SNIPER TRIGGER! Executing {direction}")
                    # Execute full size (which is cast to 1 contract in executor)
                    self.executor.sync_position(SYMBOL, target_position_size)
                
                # Wait for next cycle
                time.sleep(60 * TIMEFRAME) 
                
            except KeyboardInterrupt:
                log.info("🛑 Stopping Live Trader...")
                break
            except Exception as e:
                log.error(f"❌ Cycle Error: {e}")
                import traceback
                log.error(traceback.format_exc())
                time.sleep(60)

if __name__ == "__main__":
    bot = LiveTrader()
    bot.run_cycle()