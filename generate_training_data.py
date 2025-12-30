# --- detla-bot/generate_training_data.py ---
# 🧠 GENIUS DATA FUSER (FIXED PATHS & ALIGNMENT)
# Fixes 0.0 values by correctly aligning jagged date ranges
# and robustly finding data files on Windows/Linux.

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [DATA_GEN]: %(message)s")
log = logging.getLogger("data_gen")

# --------------------------------------------------------------------------------
# 1. ROBUST PATH DETECTION
# --------------------------------------------------------------------------------
def get_data_dir():
    # Check common locations for the data folder
    candidates = [
        Path("data"),                   # Running from inside detla-bot/
        Path("detla-bot/data"),         # Running from project root
        Path("../data"),                # Running from nested folder
        Path("C:/deltaBot/Bot/detla-bot/data") # Absolute fallback
    ]
    
    for path in candidates:
        if path.exists() and path.is_dir():
            log.info(f"📂 Found data directory at: {path.resolve()}")
            return path
            
    log.error("❌ Could not find 'data' directory. Please ensure 'data' folder exists.")
    return Path("data") # Default fallback

DATA_DIR = get_data_dir()
OUTPUT_FILE = "fused_data_real.csv"

# --------------------------------------------------------------------------------
# 2. SYMBOL MAPPING
# --------------------------------------------------------------------------------
def map_symbol(delta_symbol, target_df):
    """
    Tries to find the matching symbol in the target dataframe (e.g., BTCUSD -> BTCUSDT)
    """
    targets = target_df['symbol'].unique()
    
    # 1. Exact Match
    if delta_symbol in targets:
        return delta_symbol
    
    # 2. Try appending 'T' (BTCUSD -> BTCUSDT)
    if delta_symbol + "T" in targets:
        return delta_symbol + "T"
        
    # 3. Try removing 'T' (BTCUSDT -> BTCUSD)
    if delta_symbol.endswith("T") and delta_symbol[:-1] in targets:
        return delta_symbol[:-1]

    # 4. Try replace variants
    usdt_var = delta_symbol.replace("USD", "USDT")
    if usdt_var in targets: return usdt_var
    
    usd_var = delta_symbol.replace("USDT", "USD")
    if usd_var in targets: return usd_var
        
    return None

def process_single_symbol(symbol, candles, funding, ls_ratio):
    log.info(f"  Processing {symbol}...")
    
    # 1. Base Data: Candles (Keep all history)
    df = candles[candles['symbol'] == symbol].copy()
    if df.empty:
        log.warning(f"  ⚠️ No candle data for {symbol}")
        return pd.DataFrame()
    
    # Sort by time to be safe
    df = df.sort_values('time')

    # 2. Prepare Ancillary Data (Funding)
    f_sym = pd.DataFrame()
    if not funding.empty:
        mapped_sym = map_symbol(symbol, funding)
        if mapped_sym:
            f_sym = funding[funding['symbol'] == mapped_sym].copy()
            f_sym = f_sym.sort_values('fundingTime')
            # Log the date range overlap
            f_start = f_sym['fundingTime'].min()
            c_start = df['time'].min()
            if f_start > c_start:
                log.warning(f"    ⚠️ Funding data starts later ({f_start}) than candles ({c_start}). Early rows will have 0.0 funding.")

    # 3. Prepare Ancillary Data (L/S Ratio)
    l_sym = pd.DataFrame()
    if not ls_ratio.empty:
        mapped_sym = map_symbol(symbol, ls_ratio)
        if mapped_sym:
            l_sym = ls_ratio[ls_ratio['symbol'] == mapped_sym].copy()
            l_sym = l_sym.sort_values('timestamp')

    # 4. Smart Merge (Left Join on Candles)
    # Use merge_asof to find the most recent known value ('backward') or nearest
    
    # Merge Funding
    if not f_sym.empty:
        df = pd.merge_asof(df, f_sym[['fundingTime', 'fundingRate']], 
                           left_on='time', right_on='fundingTime', 
                           direction='backward', tolerance=pd.Timedelta("8h"))
        df['fundingRate'] = df['fundingRate'].fillna(0.0)
        df.drop(columns=['fundingTime'], inplace=True)
    else:
        df['fundingRate'] = 0.0

    # Merge L/S Ratio
    if not l_sym.empty:
        df = pd.merge_asof(df, l_sym[['timestamp', 'longShortRatio']], 
                           left_on='time', right_on='timestamp', 
                           direction='nearest', tolerance=pd.Timedelta("30min"))
        # Default L/S to 1.0 (Neutral) if missing
        df['longShortRatio'] = df['longShortRatio'].fillna(1.0)
        df.drop(columns=['timestamp'], inplace=True)
    else:
        df['longShortRatio'] = 1.0

    # ---------------------------------------------------------
    # 🧠 FEATURE ENGINEERING
    # ---------------------------------------------------------
    
    # 1. Advanced Sentinel Features
    df['funding_roc'] = df['fundingRate'].diff().fillna(0.0) * 1000 
    
    # 2. Price Action / Volatility
    df['close_log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Robust Volume Z-Score (Avoid div by zero)
    roll_mean = df['volume'].rolling(200).mean()
    roll_std = df['volume'].rolling(200).std().replace(0, 1) # Prevent div/0
    df['vol_zscore'] = (df['volume'] - roll_mean) / roll_std
    
    # 3. Liquidity Clusters (Swing Highs/Lows)
    # Rolling 7 days (2016 * 5m candles)
    roll_max = df['high'].rolling(2016).max()
    roll_min = df['low'].rolling(2016).min()
    
    # Handle NaN at start of rolling window
    df['dist_to_short_liq'] = ((roll_max - df['close']) / df['close']).fillna(0.0)
    df['dist_to_long_liq'] = ((df['close'] - roll_min) / df['close']).fillna(0.0)
    
    # 4. Magnet (POC / VWAP)
    # Simple Volume Weighted Average Price over 24h (288 * 5m)
    roll_pv = (df['close'] * df['volume']).rolling(288).sum()
    roll_v = df['volume'].rolling(288).sum().replace(0, 1)
    df['vwap_24h'] = roll_pv / roll_v
    df['dist_to_poc'] = ((df['close'] - df['vwap_24h']) / df['vwap_24h']).fillna(0.0)
    
    # 5. Order Book Imbalance (Proxy)
    df['obi'] = np.sign(df['close_log_ret']) * np.log1p(df['volume']) / 10.0
    
    # 6. Global Macro Defaults
    df['vix'] = 20.0
    df['dxy_roc'] = 0.0
    df['fear_greed_norm'] = 0.5
    
    # Cleanup all NaNs just in case
    df = df.replace([np.inf, -np.inf], 0).fillna(0.0)
    
    return df

def generate_real_data():
    log.info("📂 Loading Real Historical Data...")
    
    # Load Candles
    c_path = DATA_DIR / "historical_candles.csv"
    f_path = DATA_DIR / "historical_funding_rates.csv"
    l_path = DATA_DIR / "historical_long_short_ratio.csv"

    if not c_path.exists():
        log.error(f"❌ No candles found at {c_path.resolve()}! Did you run the fetcher?")
        return

    try:
        all_candles = pd.read_csv(c_path)
        # Handle various time column names
        t_col = 'time' if 'time' in all_candles.columns else 'timestamp'
        all_candles.rename(columns={t_col: 'time'}, inplace=True)
        all_candles['time'] = pd.to_datetime(all_candles['time'])
        all_candles = all_candles.sort_values(['symbol', 'time'])
        log.info(f"✅ Loaded {len(all_candles)} candles.")
    except Exception as e:
        log.error(f"Failed to load candles: {e}")
        return

    # Load Funding
    all_funding = pd.DataFrame()
    if f_path.exists():
        all_funding = pd.read_csv(f_path)
        if 'fundingTime' in all_funding.columns:
            all_funding['fundingTime'] = pd.to_datetime(all_funding['fundingTime'], unit='ms')
        log.info(f"✅ Loaded {len(all_funding)} funding rates.")

    # Load LS
    all_ls = pd.DataFrame()
    if l_path.exists():
        all_ls = pd.read_csv(l_path)
        if 'timestamp' in all_ls.columns:
            all_ls['timestamp'] = pd.to_datetime(all_ls['timestamp'], unit='ms')
        log.info(f"✅ Loaded {len(all_ls)} L/S ratios.")

    # Process
    fused_dfs = []
    symbols = all_candles['symbol'].unique()
    
    for sym in symbols:
        df = process_single_symbol(sym, all_candles, all_funding, all_ls)
        if not df.empty:
            fused_dfs.append(df)
            
    if not fused_dfs:
        log.error("❌ No data processed.")
        return

    final_df = pd.concat(fused_dfs).sort_values('time')
    final_df.rename(columns={'time': 'timestamp'}, inplace=True)
    
    # Stats
    log.info("🔍 Data Validation:")
    log.info(f"   Rows: {len(final_df)}")
    n_funding = (final_df['fundingRate'] != 0).sum()
    log.info(f"   Funding Non-Zero: {n_funding} / {len(final_df)} ({n_funding/len(final_df):.1%})")
    
    # Save to current directory (not data dir) for the trainer to find easily
    final_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"✅ Generated training data: {Path(OUTPUT_FILE).resolve()}")

if __name__ == "__main__":
    generate_real_data()