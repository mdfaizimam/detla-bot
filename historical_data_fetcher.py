
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path
import aiofiles
import yfinance as yf 

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DELTA_BASE_URL = "https://api.delta.exchange"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# 3 Years of Data
DAYS_TO_FETCH = 1095 
DATA_DIR = Path("data")
TARGET_ASSETS = ["BTC", "ETH", "SOL"]

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [FETCHER]: %(message)s")
log = logging.getLogger("data_fetcher")

class HistoricalDataFetcher:
    def __init__(self):
        self.session = None
        self.valid_symbols = {} # Map 'BTC' -> 'BTCUSD'
        DATA_DIR.mkdir(exist_ok=True)

    async def _init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})

    async def close(self):
        if self.session:
            await self.session.close()

    async def validate_delta_symbols(self):
        """Validates which symbols are active on Delta Exchange using robust string matching."""
        log.info("🔍 Validating symbols with Delta Exchange...")
        await self._init_session()
        
        try:
            async with self.session.get(f"{DELTA_BASE_URL}/v2/products") as resp:
                if resp.status != 200:
                    log.error(f"❌ Failed to fetch product list. Status: {resp.status}")
                    return False
                    
                data = await resp.json()
                products = data.get("result", [])
                all_symbols = set(p.get("symbol") for p in products)
                
                self.valid_symbols = {}
                for asset in TARGET_ASSETS:
                    target_usd = f"{asset}USD"
                    target_usdt = f"{asset}USDT"
                    
                    if target_usd in all_symbols:
                        self.valid_symbols[asset] = target_usd
                        log.info(f"✅ Found valid symbol for {asset}: {target_usd}")
                    elif target_usdt in all_symbols:
                        self.valid_symbols[asset] = target_usdt
                        log.info(f"✅ Found valid symbol for {asset}: {target_usdt}")
                    else:
                        log.warning(f"⚠️ Could not find {target_usd} or {target_usdt} on Delta.")
                
                return bool(self.valid_symbols)
        except Exception as e:
            log.error(f"Symbol validation crashed: {e}")
            return False

    async def fetch_macro_data(self):
        """Fetches VIX and DXY using yfinance (Daily resolution)"""
        log.info("🌍 Fetching Macro Data (VIX, DXY)...")
        try:
            # Run blocking call in thread
            await asyncio.to_thread(self._fetch_macro_sync)
        except Exception as e:
            log.error(f"❌ Macro data fetch failed: {e}")

    def _fetch_macro_sync(self):
        start_date = (datetime.now() - timedelta(days=DAYS_TO_FETCH)).strftime('%Y-%m-%d')
        tickers = ["^VIX", "DX-Y.NYB"]
        data = yf.download(tickers, start=start_date, interval="1d", progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].reset_index()
        else:
            df = data.reset_index()
            
        df.columns = [str(col).replace("DX-Y.NYB", "dxy_close").replace("^VIX", "vix_close") for col in df.columns]
        
        # Normalize date column
        for col in ['Datetime', 'Date']:
            if col in df.columns: 
                df.rename(columns={col: 'timestamp'}, inplace=True)
                break
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            
        # Save
        path = DATA_DIR / "historical_macro.csv"
        df.to_csv(path, index=False)
        log.info(f"💾 Saved {len(df)} macro rows to {path.resolve()}")

    async def fetch_candles(self):
        """Fetches 3 years of 5m candles for all valid symbols concurrently."""
        tasks = [self._fetch_single_symbol_candles(sym) for sym in self.valid_symbols.values()]
        results = await asyncio.gather(*tasks)
        
        valid = [df for df in results if not df.empty]
        if valid:
            final_df = pd.concat(valid)
            path = DATA_DIR / "historical_candles.csv"
            await self._save_df(final_df, path)
        else:
            log.error("❌ No candles fetched!")

    async def _fetch_single_symbol_candles(self, symbol):
        log.info(f"🕯️ Fetching Candles for {symbol}...")
        end = int(time.time())
        start = int((datetime.now() - timedelta(days=DAYS_TO_FETCH)).timestamp())
        
        current = end
        step = 518400 # 6 days
        chunks = []
        
        while current > start:
            chunk_start = max(start, current - step)
            params = {
                "symbol": symbol, "resolution": "5m", 
                "start": str(chunk_start), "end": str(current), "limit": "2000"
            }
            try:
                async with self.session.get(f"{DELTA_BASE_URL}/v2/history/candles", params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("result", [])
                        if result: chunks.extend(result)
            except Exception as e:
                log.warning(f"⚠️ Candle chunk failed: {e}")
            
            current = chunk_start
            await asyncio.sleep(0.05)
            
        if not chunks: return pd.DataFrame()
        
        df = pd.DataFrame(chunks)
        tcol = "time" if "time" in df.columns else "timestamp"
        df["time"] = pd.to_datetime(df[tcol], unit="s")
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["symbol"] = symbol
        return df

    async def fetch_binance_metrics(self):
        """Fetches L/S Ratio and Open Interest using Backwards Pagination."""
        # Need to map Delta Symbols (BTCUSD) to Binance (BTCUSDT)
        binance_symbols = [f"{asset}USDT" for asset in self.valid_symbols.keys()]
        
        # 1. L/S Ratio
        await self._fetch_metric_group(binance_symbols, "ls", "historical_long_short_ratio.csv")
        # 2. Open Interest
        await self._fetch_metric_group(binance_symbols, "oi", "historical_open_interest.csv")

    async def _fetch_metric_group(self, symbols, metric_type, filename):
        tasks = [self._fetch_single_metric_backwards(sym, metric_type) for sym in symbols]
        results = await asyncio.gather(*tasks)
        
        valid = [df for df in results if not df.empty]
        if valid:
            final_df = pd.concat(valid)
            path = DATA_DIR / filename
            await self._save_df(final_df, path)

    async def _fetch_single_metric_backwards(self, symbol, metric_type):
        meta = {
            "ls": ("/futures/data/globalLongShortAccountRatio", "⚖️ L/S Ratio"),
            "oi": ("/futures/data/openInterestHist", "📊 Open Interest")
        }
        endpoint, label = meta[metric_type]
        log.info(f"{label} Fetching history for {symbol}...")
        
        limit_ts = int((datetime.now() - timedelta(days=DAYS_TO_FETCH)).timestamp() * 1000)
        current_end = int(time.time() * 1000)
        all_data = []
        
        while current_end > limit_ts:
            url = f"{BINANCE_FUTURES_URL}{endpoint}"
            params = {"symbol": symbol, "period": "5m", "limit": 500, "endTime": current_end}
            
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 429:
                        log.warning("🔥 Rate Limit! Sleeping 5s...")
                        await asyncio.sleep(5)
                        continue
                    if resp.status != 200:
                        break # Stop on error (likely end of data)
                    
                    data = await resp.json()
                    if not data: break
                    
                    all_data.extend(data)
                    oldest_ts = data[0]['timestamp']
                    if oldest_ts >= current_end: break # Prevent infinite loop
                    current_end = oldest_ts - 1
                    
                    await asyncio.sleep(0.05)
            except Exception:
                break

        if not all_data: return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        # Normalize to internal format: BTCUSDT -> BTCUSD
        df['symbol'] = symbol.replace("USDT", "USD")
        
        # Clean columns
        if metric_type == "ls" and 'longShortRatio' in df.columns:
            df = df[['symbol', 'timestamp', 'longShortRatio', 'longAccount', 'shortAccount']]
        elif metric_type == "oi" and 'sumOpenInterest' in df.columns:
            df = df[['symbol', 'timestamp', 'sumOpenInterest', 'sumOpenInterestValue']]
            
        return df

    async def fetch_funding(self):
        """Fetches Funding Rates efficiently using robust backwards windows."""
        binance_symbols = [f"{asset}USDT" for asset in self.valid_symbols.keys()]
        tasks = [self._fetch_single_funding(sym) for sym in binance_symbols]
        results = await asyncio.gather(*tasks)
        
        valid = [df for df in results if not df.empty]
        if valid:
            final_df = pd.concat(valid)
            # Dedup based on time/symbol
            final_df = final_df.drop_duplicates(subset=['symbol', 'fundingTime'])
            path = DATA_DIR / "historical_funding_rates.csv"
            await self._save_df(final_df, path)

    async def _fetch_single_funding(self, symbol):
        log.info(f"💸 Fetching Funding for {symbol}...")
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
        all_data = []
        
        # Strategy: 
        # First request: Don't specify endTime (get latest real-world data).
        # Subsequent requests: Use the oldest timestamp from previous batch.
        # This handles cases where System Time > Real Time (Simulated Env).
        
        current_end_time = None 
        
        for i in range(5): 
            params = {"symbol": symbol, "limit": 1000}
            if current_end_time:
                params["endTime"] = current_end_time
                
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data: 
                            log.warning(f"   No data for {symbol} (Iter {i})")
                            break
                        
                        all_data.extend(data)
                        
                        # Debug Log
                        first_ts = data[0]['fundingTime']
                        last_ts = data[-1]['fundingTime']
                        log.info(f"   {symbol} Chunk {i}: {len(data)} rows. Range: {datetime.fromtimestamp(first_ts/1000)} -> {datetime.fromtimestamp(last_ts/1000)}")
                        
                        # Move cursor back
                        oldest_ts = min(d['fundingTime'] for d in data)
                        current_end_time = oldest_ts - 1
                        
                        await asyncio.sleep(0.1)
                    else:
                        log.warning(f"   Failed {symbol} funding fetch: {resp.status}")
                        break
            except Exception as e:
                log.error(f"   Error fetching funding: {e}")
                break
                
        if not all_data: return pd.DataFrame()
        df = pd.DataFrame(all_data)
        df['symbol'] = symbol.replace("USDT", "USD")
        return df

    async def _save_df(self, df, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use simple synchronous save or thread for simplicity in "God Mode" runner
        # But keeping asyncio logic for consistency
        async with aiofiles.open(path, "w") as f:
            await f.write(df.to_csv(index=False))
        log.info(f"💾 Saved {len(df)} rows to {path.resolve()}")

    async def run(self):
        await self._init_session()
        
        # 1. Macro
        await self.fetch_macro_data()
        
        # 2. Validate
        if not await self.validate_delta_symbols():
            await self.close()
            return
            
        # 3. Candles
        await self.fetch_candles()
        
        # 4. Metrics (L/S, OI)
        await self.fetch_binance_metrics()
        
        # 5. Funding
        await self.fetch_funding()
        
        await self.close()
        print("\n✅ GOD MODE DATA FETCH COMPLETE.")

if __name__ == "__main__":
    fetcher = HistoricalDataFetcher()
    asyncio.run(fetcher.run())