# --- detla-bot/generate_training_data.py ---
# 🧠 INSTITUTIONAL DATA FUSER (MULTI-TIMEFRAME EDITION)
# Merges Candles, Spot, Funding, and Macro.
# Generates 1H and 4H context features from 5m data.

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [DATA_GEN]: %(message)s")
log = logging.getLogger("data_gen")

DATA_DIR = Path("data")
OUTPUT_FILE = "fused_data_real_FULL.csv"

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        log.error(f"❌ Missing file: {path}")
        return None
    
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --------------------------------------------------------------------------------
# 🛠️ HELPER: RESAMPLING ENGINE
# --------------------------------------------------------------------------------
def calculate_mtf_features(df_5m, interval, suffix):
    """
    Resamples 5m data to 'interval' (e.g., '1h', '4h'), calculates trends,
    and returns a DataFrame ready to be merged back.
    """
    # 1. Resample
    df_resampled = df_5m.set_index('timestamp').resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # 2. Calculate Indicators on this timeframe
    # A. Trend (EMA 20 vs EMA 50)
    ema_fast = df_resampled['close'].ewm(span=20, adjust=False).mean()
    ema_slow = df_resampled['close'].ewm(span=50, adjust=False).mean()
    df_resampled[f'trend_bias_{suffix}'] = (ema_fast - ema_slow) / ema_slow # Normalized Trend
    
    # B. Volatility (ATR Normalized)
    tr = np.maximum(df_resampled['high'] - df_resampled['low'], 
                    np.abs(df_resampled['high'] - df_resampled['close'].shift(1)))
    atr = tr.rolling(14).mean()
    df_resampled[f'volatility_{suffix}'] = atr / df_resampled['close']
    
    # C. Momentum (RSI)
    delta = df_resampled['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_resampled[f'rsi_{suffix}'] = 100 - (100 / (1 + rs))
    
    # Reset index for merging
    df_resampled = df_resampled.reset_index()
    
    # Keep only the timestamp and the new features
    return df_resampled[['timestamp', f'trend_bias_{suffix}', f'volatility_{suffix}', f'rsi_{suffix}']]

# --------------------------------------------------------------------------------
# 2. CORE PROCESSING
# --------------------------------------------------------------------------------
def process_symbol_group(base_asset, candles, spot, funding, macro):
    log.info(f"🔹 Processing {base_asset}...")
    
    # 1. Filter Futures Candles (Target)
    c_df = candles[candles['base_asset'] == base_asset].copy()
    if c_df.empty: return pd.DataFrame()
    c_df = c_df.sort_values('timestamp')

    # 2. Filter Spot Candles (For Basis)
    s_df = spot[spot['base_asset'] == base_asset].copy()
    s_df = s_df.sort_values('timestamp')
    
    # 3. Filter Funding (Regime)
    f_df = funding[funding['base_asset'] == base_asset].copy()
    f_df = f_df.sort_values('timestamp')

    # ---------------------------------------------------------
    # 🕒 MULTI-TIMEFRAME GENERATION (The Genius Part)
    # ---------------------------------------------------------
    # Generate 1-Hour Context
    mtf_1h = calculate_mtf_features(c_df, '1h', '1h')
    
    # Generate 4-Hour Context
    mtf_4h = calculate_mtf_features(c_df, '4h', '4h')

    # ---------------------------------------------------------
    # 🔗 FUSION (Smart Merge)
    # ---------------------------------------------------------
    
    # Merge Spot (Nearest match within 5m)
    df = pd.merge_asof(c_df, s_df[['timestamp', 'spot_close', 'spot_volume']],
                       on='timestamp', direction='nearest', tolerance=pd.Timedelta("5m"))
    
    # Merge Funding (Backward: Last known funding rate)
    df = pd.merge_asof(df, f_df[['timestamp', 'funding_rate']],
                       on='timestamp', direction='backward', tolerance=pd.Timedelta("8h"))
    
    # Merge Macro (Broadcast daily/5m macro)
    df = pd.merge_asof(df, macro[['timestamp', 'vix_close', 'dxy_close']],
                       on='timestamp', direction='backward', tolerance=pd.Timedelta("1d"))

    # ✅ Merge MTF 1H (Backward: e.g. 10:15 sees 10:00 candle data)
    df = pd.merge_asof(df, mtf_1h, on='timestamp', direction='backward')

    # ✅ Merge MTF 4H (Backward)
    df = pd.merge_asof(df, mtf_4h, on='timestamp', direction='backward')

    # ---------------------------------------------------------
    # 🧮 FEATURE CALCULATION
    # ---------------------------------------------------------
    
    # A. Basis
    df['basis'] = (df['close'] - df['spot_close']) / df['spot_close']
    
    # B. Funding Regime
    df['funding_roc'] = df['funding_rate'].diff().fillna(0) * 1000
    
    # C. Price Action
    df['close_log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # D. Volume Z-Score
    roll_mean = df['volume'].rolling(288).mean() # 24h
    roll_std = df['volume'].rolling(288).std().replace(0, 1)
    df['vol_zscore'] = (df['volume'] - roll_mean) / roll_std
    
    # E. Order Book Imbalance Proxy
    df['obi'] = np.sign(df['basis']) * np.log1p(df['volume']) / 10.0
    
    # F. Liquidity Levels
    window_7d = 2016 
    roll_max = df['high'].rolling(window_7d).max()
    roll_min = df['low'].rolling(window_7d).min()
    df['dist_to_long_liq'] = ((df['close'] - roll_min) / df['close'])
    df['dist_to_short_liq'] = ((roll_max - df['close']) / df['close'])
    
    # G. Macro Features
    df['dxy_roc'] = df['dxy_close'].pct_change().fillna(0) * 100
    df['fear_greed_norm'] = (df['vix_close'] - 10) / 20.0
    
    # H. Market Structure
    df['oi_pct_change'] = df['volume'].pct_change(288).fillna(0)
    
    # I. Dummy Placeholders
    df['longShortRatio'] = 1.0 
    df['dist_to_poc'] = 0.0 
    
    # Cleanup NaNs (caused by rolling windows and MTF resampling lag)
    df = df.dropna()
    
    # Sanitize Infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    log.info(f"   ✅ {base_asset}: Generated {len(df)} MTF-aligned rows.")
    return df

# --------------------------------------------------------------------------------
# 3. MAIN RUNNER
# --------------------------------------------------------------------------------
def generate_real_data():
    log.info("🚀 Starting Institutional Data Fusion (MTF Edition)...")
    
    # Load Source Files
    candles = load_csv("historical_candles.csv")
    spot = load_csv("historical_spot_candles.csv")
    funding = load_csv("historical_funding_rates.csv")
    macro = load_csv("historical_macro.csv")
    
    if any(x is None for x in [candles, spot, funding, macro]):
        log.error("❌ Aborting. Missing input files.")
        return

    # Forward Fill Macro
    macro = macro.sort_values('timestamp').set_index('timestamp').resample('5min').ffill().reset_index()

    fused_dfs = []
    
    assets = candles['base_asset'].unique()
    for asset in assets:
        df = process_symbol_group(asset, candles, spot, funding, macro)
        if not df.empty:
            fused_dfs.append(df)
            
    if not fused_dfs:
        log.error("❌ No valid data generated.")
        return

    final_df = pd.concat(fused_dfs).sort_values('timestamp')
    
    log.info("🔍 Final Validation:")
    log.info(f"   Total Rows: {len(final_df)}")
    log.info(f"   Features: {list(final_df.columns)}")
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"✅ SUCCESS! MTF Training data saved to: {Path(OUTPUT_FILE).resolve()}")

if __name__ == "__main__":
    generate_real_data()