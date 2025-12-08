# --- detla-bot/historical_data_fetcher.py ---
# FIXED: Reduced L/S fetch window to 28 days to prevent HTTP 400 errors.
# FIXED: Skips Candle fetching if file already exists (saves time).
# FIXED: Improved error handling for Binance API.

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import shutil
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import aiofiles
from io import StringIO
import os

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
try:
    from config import DELTA_BASE_URL, BINANCE_FUTURES_URL, USER_AGENT
except ImportError:
    DELTA_BASE_URL = "https://api.delta.exchange"
    BINANCE_FUTURES_URL = "https://fapi.binance.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

DEFAULT_DAYS_TO_FETCH = 365
TRAINING_TIMEFRAME = "5m"
FETCH_CHUNK_DAYS = 6
DEFAULT_DATA_DIR = Path("data")
DEFAULT_TRADING_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
SYMBOL_MAPPING = {
    "BTCUSD": {"delta": "BTCUSD", "binance": "BTCUSDT"},
    "ETHUSD": {"delta": "ETHUSD", "binance": "ETHUSDT"},
    "SOLUSD": {"delta": "SOLUSD", "binance": "SOLUSDT"}
}
HTTP_TIMEOUT = 15

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "data_fetcher.log", encoding="utf-8")
    ]
)
log = logging.getLogger("data_fetcher")

# ----------------------------------------------------------------------
# Retry Logic
# ----------------------------------------------------------------------
def async_retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        log.error(f"❌ {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise e
                    log.warning(f"⚠️ {func.__name__} attempt {attempt+1} failed: {e}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= 2
        return wrapper
    return decorator

# ----------------------------------------------------------------------
# Data Fetching Logic
# ----------------------------------------------------------------------
@async_retry()
async def fetch_candle_chunk(session, symbol, start, end):
    params = {
        "symbol": symbol, "resolution": TRAINING_TIMEFRAME, 
        "start": str(start), "end": str(end), "limit": "2000"
    }
    async with session.get(f"{DELTA_BASE_URL}/v2/history/candles", params=params, headers={"User-Agent": USER_AGENT}) as resp:
        if resp.status != 200: return []
        data = await resp.json()
        return data.get("result", [])

@async_retry()
async def fetch_binance_ls_paginated(session, symbol, start_ts, end_ts):
    """
    Fetches Long/Short Ratio using strict pagination (500 limit).
    """
    binance_symbol = SYMBOL_MAPPING.get(symbol, {}).get("binance", f"{symbol[:-3]}USDT")
    all_data = []
    current_start = start_ts
    
    # Loop until we reach end_ts
    while current_start < end_ts:
        params = {
            "symbol": binance_symbol, 
            "period": "5m", 
            "limit": 500,
            "startTime": current_start,
            "endTime": end_ts
        }
        
        try:
            url = f"{BINANCE_FUTURES_URL}/futures/data/globalLongShortAccountRatio"
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    log.warning(f"Binance Rate Limit! Sleeping 5s...")
                    await asyncio.sleep(5)
                    continue
                
                if resp.status != 200:
                    text = await resp.text()
                    log.warning(f"{symbol}: L/S fetch failed HTTP {resp.status} - {text}")
                    break
                
                data = await resp.json()
                if not isinstance(data, list) or not data:
                    break # No more data
                
                all_data.extend(data)
                
                # Update pagination cursor
                last_ts = data[-1]['timestamp']
                # If we aren't moving forward, force a jump (prevent infinite loop)
                if last_ts <= current_start:
                    current_start += (500 * 5 * 60 * 1000) # 500 candles * 5 mins
                else:
                    current_start = last_ts + 300000 # +5 mins
                
                await asyncio.sleep(0.1) # Be nice to API
                    
        except Exception as e:
            log.error(f"{symbol}: Pagination error: {e}")
            break
            
    return all_data

# ----------------------------------------------------------------------
# Orchestrators
# ----------------------------------------------------------------------
async def fetch_candles_orchestrator(session, symbol, days):
    log.info(f"🕯️ Fetching candles for {symbol}...")
    end = int(time.time())
    start = int((datetime.now() - timedelta(days=days)).timestamp())
    
    chunks = []
    current = end
    step = 6 * 86400 # 6 day chunks
    
    while current > start:
        chunk_start = max(start, current - step)
        data = await fetch_candle_chunk(session, symbol, chunk_start, current)
        if data: chunks.extend(data)
        current = chunk_start
        await asyncio.sleep(0.1)

    if not chunks: return pd.DataFrame()
    
    df = pd.DataFrame(chunks)
    tcol = "time" if "time" in df.columns else "timestamp"
    df["time"] = pd.to_datetime(df[tcol], unit="s")
    df = df.sort_values("time").drop_duplicates(subset=["time"])
    
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["symbol"] = symbol
    return df

async def fetch_sentiment_orchestrator(session, symbols, days):
    log.info("🧠 Fetching Sentiment Data (Funding + L/S Ratio)...")
    
    funding_list = []
    ls_list = []
    
    end_ts = int(time.time() * 1000)
    
    # ✅ FIX: Reduced to 28 days to prevent HTTP 400 (Binance strict limit)
    ls_start_ts = int((datetime.now() - timedelta(days=28)).timestamp() * 1000)
    
    fund_start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    for symbol in symbols:
        binance_sym = SYMBOL_MAPPING[symbol]["binance"]
        
        # 1. Funding Rates
        try:
            url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
            params = {"symbol": binance_sym, "startTime": fund_start_ts, "endTime": end_ts, "limit": 1000}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        for d in data: d["symbol"] = symbol
                        funding_list.extend(data)
                        log.info(f"  ✅ {symbol}: Fetched {len(data)} funding records.")
        except Exception as e: log.error(f"Funding fetch error {symbol}: {e}")

        # 2. Long/Short Ratio
        log.info(f"  ⏳ {symbol}: Fetching L/S Ratio (Last 28 Days)...")
        ls_data = await fetch_binance_ls_paginated(session, symbol, ls_start_ts, end_ts)
        if ls_data:
            for d in ls_data: d["symbol"] = symbol
            ls_list.extend(ls_data)
            log.info(f"  ✅ {symbol}: Fetched {len(ls_data)} L/S records.")
        else:
            log.warning(f"  ⚠️ {symbol}: No L/S data found.")

    return pd.DataFrame(funding_list), pd.DataFrame(ls_list)

async def save_df(df, path):
    if df.empty: return
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w") as f:
        await f.write(df.to_csv(index=False))
    log.info(f"💾 Saved {len(df)} rows to {path.name}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def run_fetcher(symbols=DEFAULT_TRADING_SYMBOLS, days=365, auto_start=False, output_dir=DEFAULT_DATA_DIR):
    # Check if candles already exist to save time
    candle_path = Path(output_dir) / "historical_candles.csv"
    skip_candles = False
    
    if candle_path.exists():
        if auto_start:
            skip_candles = True
            log.info("⏩ Found existing candles. Skipping candle fetch (Sentiments Only).")
        else:
            print("\nFound existing candle data. Skip candle fetch? (y/n)")
            if input().lower() == 'y': 
                skip_candles = True

    async with aiohttp.ClientSession() as session:
        # 1. Fetch Candles (If needed)
        if not skip_candles:
            all_candles = []
            for sym in symbols:
                df = await fetch_candles_orchestrator(session, sym, days)
                if not df.empty: all_candles.append(df)
            
            if all_candles:
                final_candles = pd.concat(all_candles).drop_duplicates(subset=["time", "symbol"]).sort_values("time")
                await save_df(final_candles, candle_path)

        # 2. Fetch Sentiment (Always run this to fix L/S)
        funding_df, ls_df = await fetch_sentiment_orchestrator(session, symbols, days)
        
        await save_df(funding_df, Path(output_dir) / "historical_funding_rates.csv")
        
        if not ls_df.empty:
            ls_path = Path(output_dir) / "historical_long_short_ratio.csv"
            # Standardize columns
            if "longShortRatio" in ls_df.columns:
                ls_df = ls_df[["symbol", "timestamp", "longShortRatio", "longAccount", "shortAccount"]]
            await save_df(ls_df, ls_path)
        else:
            log.error("❌ Still no L/S Data. Check Binance API availability.")
            # Create dummy to prevent training crash
            dummy = pd.DataFrame(columns=["symbol", "timestamp", "longShortRatio", "longAccount", "shortAccount"])
            dummy.to_csv(Path(output_dir) / "historical_long_short_ratio.csv", index=False)

    print("\n✅ Data Fetch Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", default="no")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    
    asyncio.run(run_fetcher(days=args.days, auto_start=(args.auto=="yes")))