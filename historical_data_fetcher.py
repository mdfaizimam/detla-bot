# --- historical_data_fetcher.py ---
# Complete Updated File (Optimized for 2000 candle limit and larger chunks)

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

from config import DELTA_BASE_URL, USER_AGENT, TRADING_SYMBOLS

# --- Configuration ---
DAYS_TO_FETCH = 365        # The total historical range we want (1 year for better S/R)
TRAINING_TIMEFRAME = "5m"  # The FINAL candle size for our training data
FETCH_CHUNK_DAYS = 30      # Fetch 30 days at a time 
# --------------------

DATA_DIR = "data"        
OUTPUT_FILE = os.path.join(DATA_DIR, "historical_candles.csv")
API_LIMIT = 2000 # ✅ UPDATED: Max API limit per request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
log = logging.getLogger("data_fetcher")

async def fetch_candle_chunk(session, symbol, resolution, start_time, end_time, limit=API_LIMIT):
    """Fetches a single chunk of candle data from Delta Exchange."""
    path = "/v2/history/candles"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": str(start_time),
        "end": str(end_time),
        "limit": str(limit)
    }
    url = f"{DELTA_BASE_URL}{path}"
    
    try:
        async with session.get(url, params=params, headers={'User-Agent': USER_AGENT}) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("result", [])
            else:
                log.error(f"Failed to fetch data for {symbol}: HTTP {resp.status} {await resp.text()}")
            return []
    except Exception as e:
        log.error(f"Error during fetch for {symbol}: {e}")
        return []

async def fetch_all_data_for_symbol(session, symbol):
    """Continuously fetches 5m data backwards in chunks for a single symbol."""
    log.info(f"Starting download for {symbol}...")
    all_candles = []
    
    end_date = datetime.now()
    fetch_start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    current_end_date = end_date

    while current_end_date > fetch_start_date:
        
        # Calculate the chunk's start date
        chunk_start_date = current_end_date - timedelta(days=FETCH_CHUNK_DAYS)
        if chunk_start_date < fetch_start_date:
            chunk_start_date = fetch_start_date
        
        start_ts = int(chunk_start_date.timestamp())
        end_ts = int(current_end_date.timestamp())

        log.info(f"Fetching {symbol} chunk: {chunk_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
        
        # Fetching backward in time (up to current_end_date)
        candles = await fetch_candle_chunk(
            session, 
            symbol, 
            TRAINING_TIMEFRAME, 
            start_ts, 
            end_ts, 
            limit=API_LIMIT
        )
        
        if not candles:
            log.warning(f"No candles returned for {symbol} in chunk ending {current_end_date.strftime('%Y-%m-%d')}. Stopping.")
            # Break if no data is returned, assuming we hit the start of history
            break 
            
        all_candles.extend(candles)
        
        # Move the end date for the next chunk backward to the start of the current chunk
        current_end_date = chunk_start_date
        
        await asyncio.sleep(0.5) 

    log.info(f"✅ Download complete for {symbol}. Total candles: {len(all_candles)}")
    return all_candles

async def main():
    """Main function to orchestrate the data fetching and saving."""
    
    os.makedirs(DATA_DIR, exist_ok=True)
    all_dataframes = []

    async with aiohttp.ClientSession() as session:
        
        tasks = [fetch_all_data_for_symbol(session, symbol) for symbol in TRADING_SYMBOLS]
        results = await asyncio.gather(*tasks)

        for symbol_index, candles_list in enumerate(results):
            symbol = TRADING_SYMBOLS[symbol_index]
            if candles_list:
                df = pd.DataFrame(candles_list)
                df['symbol'] = symbol
                all_dataframes.append(df)
            else:
                log.warning(f"Final data check: No rows available for {symbol}.")

    if not all_dataframes:
        log.error("No data was downloaded for any symbol. Exiting.")
        return

    log.info("Combining and cleaning all data...")
    final_df = pd.concat(all_dataframes)

    # Clean up data
    df_numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in df_numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col])

    # ✅ FIX: Convert the 'time' column to a proper Python datetime object
    final_df['time'] = pd.to_datetime(final_df['time'], unit='s')
    
    final_df = final_df.drop_duplicates(subset=['symbol', 'time'])
    final_df = final_df.sort_values(by=['symbol', 'time'])
    
    # ✅ FIX: Explicitly format the timestamp back to integer seconds before saving to CSV
    final_df['time'] = final_df['time'].astype(np.int64) // 10**9

    # Save to CSV
    final_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"✅ All data saved successfully to {OUTPUT_FILE}")
    log.info(f"Total rows: {len(final_df)}")
    log.info(f"Date range: {pd.to_datetime(final_df['time'].min(), unit='s')} to {pd.to_datetime(final_df['time'].max(), unit='s')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Data fetching cancelled by user.")