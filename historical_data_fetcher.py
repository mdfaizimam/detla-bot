# --- detla-bot/historical_data_fetcher.py ---
# 🧠 INSTITUTIONAL DATA FETCHER (World Class Upgrade)
# Fetches Candles, Funding, Spot Data + ORDER FLOW (CVD)

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import time
from pathlib import Path
import aiofiles
import yfinance as yf
import random
import sys

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Toggle based on your region
DELTA_BASE_URL = "https://api.india.delta.exchange"
# DELTA_BASE_URL = "https://api.delta.exchange"

BINANCE_SPOT_URL = "https://api.binance.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

DAYS_TO_FETCH = 1095  # 3 Years
DATA_DIR = Path("data")
TARGET_ASSETS = ["BTC", "ETH", "SOL"]

# ----------------------------------------------------------------------
# Logging & Robustness
# ----------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [FETCHER]: %(message)s")
log = logging.getLogger("data_fetcher")

def async_retry(retries=3, backoff_factor=1.5):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = 1
            for i in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status in [429, 500, 502, 503, 504]:
                        reset = int(e.headers.get("Retry-After", delay)) if e.headers else delay
                        log.warning(f"⚠️ API Status {e.status}. Sleeping {reset}s...")
                        await asyncio.sleep(reset)
                        delay *= backoff_factor
                    else:
                        raise e
                except Exception as e:
                    if i == retries:
                        log.error(f"❌ Critical Failure in {func.__name__}: {str(e)}")
                        raise e
                    log.warning(f"⚠️ Transient Error: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

class HistoricalDataFetcher:
    def __init__(self):
        self.session = None
        self.symbol_map = {} 
        self.server_time_s = None # Source of Truth
        DATA_DIR.mkdir(exist_ok=True)

    async def _init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}, raise_for_status=False)

    async def close(self):
        if self.session:
            await self.session.close()

    @async_retry(retries=3)
    async def fetch_json(self, url, params=None):
        async with self.session.get(url, params=params) as resp:
            if resp.status == 429:
                raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=429, headers=resp.headers)
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # 1. Synchronization (Zero Leakage)
    # ------------------------------------------------------------------
    async def get_delta_server_time(self):
        """CRITICAL: Get server time to prevent future-leakage."""
        log.info("⏳ Synchronizing with Delta Server Time...")
        try:
            try:
                data = await self.fetch_json(f"{DELTA_BASE_URL}/v2/time")
                ts = int(data.get("server_time", 0))
                if ts > 0: return ts // 1000 
            except: pass

            async with self.session.get(f"{DELTA_BASE_URL}/v2/tickers") as resp:
                if "Date" in resp.headers:
                    http_date = resp.headers["Date"]
                    dt = datetime.strptime(http_date, "%a, %d %b %Y %H:%M:%S GMT")
                    return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception as e:
            log.error(f"❌ Server Time Sync Failed: {e}")
            raise e
            
        raise Exception("Could not verify Exchange Server Time. Aborting to prevent leakage.")

    # ------------------------------------------------------------------
    # 2. Validation & Metadata
    # ------------------------------------------------------------------
    async def validate_delta_symbols(self):
        log.info("🔍 Validating symbols via Metadata...")
        try:
            data = await self.fetch_json(f"{DELTA_BASE_URL}/v2/products")
            products = data.get("result", [])
            
            self.symbol_map = {}
            for asset in TARGET_ASSETS:
                candidates = []
                for p in products:
                    if p.get("contract_type") != "perpetual_futures" or p.get("state") != "live":
                        continue
                        
                    p_base = p.get("base_asset_symbol")
                    if not p_base:
                        ua = p.get("underlying_asset")
                        if ua and isinstance(ua, dict):
                            p_base = ua.get("symbol")
                    
                    p_symbol = p.get("symbol", "")
                    
                    if (p_base == asset) or (not p_base and p_symbol.startswith(asset)):
                         candidates.append(p)

                # Prioritize USD Inverse
                target_exact = f"{asset}USD"
                candidates.sort(key=lambda x: 0 if x.get("symbol") == target_exact else 1)
                
                if candidates:
                    chosen = candidates[0]
                    delta_sym = chosen.get("symbol")
                    if not delta_sym:
                         continue

                    self.symbol_map[asset] = {
                        "delta": delta_sym,
                        "binance_spot": f"{asset}USDT" 
                    }
                    log.info(f"✅ Locked {asset}: Delta=[{delta_sym}] (Settlement: {chosen.get('settlement_asset_symbol')})")
                else:
                    log.warning(f"⚠️ No live perpetuals found for {asset} on Delta.")
            
            return bool(self.symbol_map)
        except Exception as e:
            log.error(f"Symbol validation crashed: {e}")
            return False

    # ------------------------------------------------------------------
    # 3. Delta Futures Candles (Target)
    # ------------------------------------------------------------------
    async def fetch_delta_candles(self):
        tasks = [self._fetch_single_delta_asset(asset, meta['delta']) 
                 for asset, meta in self.symbol_map.items()]
        results = await asyncio.gather(*tasks)
        self._concat_and_save(results, "historical_candles.csv")

    async def _fetch_single_delta_asset(self, asset, symbol):
        log.info(f"🕯️ Fetching Delta Futures: {symbol}...")
        
        end_ts = self.server_time_s
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=DAYS_TO_FETCH)).timestamp())
        
        current = end_ts
        chunk_step_fallback = 600000 
        all_data = []
        
        while current > start_ts:
            chunk_start = max(start_ts, current - chunk_step_fallback)
            params = {
                "symbol": symbol, "resolution": "5m", 
                "start": str(chunk_start), "end": str(current), "limit": "4000"
            }
            try:
                data = await self.fetch_json(f"{DELTA_BASE_URL}/v2/history/candles", params=params)
                result = data.get("result", [])
                
                if result:
                    all_data.extend(result)
                    timestamps = [r.get("time", 0) for r in result]
                    oldest = min(timestamps)
                    current = oldest - 1 
                else:
                    current = chunk_start - 1
            except Exception as e:
                log.warning(f"⚠️ Chunk failed for {symbol}: {e}")
                current = chunk_start - 1
            
            await asyncio.sleep(0.1) 
            
        if not all_data: return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        tcol = "time" if "time" in df.columns else "timestamp"
        df["timestamp"] = pd.to_datetime(df[tcol], unit="s")
        if tcol != "timestamp": df.drop(columns=[tcol], inplace=True)
        
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
        df["symbol"] = symbol 
        df["base_asset"] = asset
        df = df[df['volume'] > 0]
        
        return df.sort_values("timestamp")

    # ------------------------------------------------------------------
    # 4. Delta Funding Rates
    # ------------------------------------------------------------------
    async def fetch_delta_funding(self):
        tasks = [self._fetch_single_delta_funding(asset, meta['delta']) 
                 for asset, meta in self.symbol_map.items()]
        results = await asyncio.gather(*tasks)
        self._concat_and_save(results, "historical_funding_rates.csv")

    async def _fetch_single_delta_funding(self, asset, symbol):
        funding_symbol = f"FUNDING:{symbol}"
        log.info(f"💸 Fetching Delta Funding: {funding_symbol}...")
        
        end_ts = self.server_time_s
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=DAYS_TO_FETCH)).timestamp())
        
        current = end_ts
        chunk_step_fallback = 2592000 
        all_data = []
        
        while current > start_ts:
            chunk_start = max(start_ts, current - chunk_step_fallback)
            params = {
                "symbol": funding_symbol, 
                "resolution": "1h", 
                "start": str(chunk_start), 
                "end": str(current), 
                "limit": "4000"
            }
            try:
                data = await self.fetch_json(f"{DELTA_BASE_URL}/v2/history/candles", params=params)
                result = data.get("result", [])
                if result:
                    all_data.extend(result)
                    timestamps = [r.get("time", 0) for r in result]
                    oldest = min(timestamps)
                    current = oldest - 1
                else:
                    current = chunk_start - 1
                    
            except Exception as e:
                current = chunk_start - 1
            
            await asyncio.sleep(0.1)

        if not all_data: 
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        tcol = "time" if "time" in df.columns else "timestamp"
        df["timestamp"] = pd.to_datetime(df[tcol], unit="s")
        if tcol != "timestamp": df.drop(columns=[tcol], inplace=True)
        
        df["funding_rate"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[["timestamp", "funding_rate"]]
        df["symbol"] = symbol
        df["base_asset"] = asset
        return df.sort_values("timestamp")

    # ------------------------------------------------------------------
    # 5. Binance Spot (Basis + CVD Orderflow)
    # ------------------------------------------------------------------
    async def fetch_binance_spot(self):
        tasks = [self._fetch_single_binance_spot(asset, meta['binance_spot']) 
                 for asset, meta in self.symbol_map.items()]
        results = await asyncio.gather(*tasks)
        self._concat_and_save(results, "historical_spot_candles.csv")

    async def _fetch_single_binance_spot(self, asset, symbol):
        log.info(f"🪙 Fetching Binance Spot (with Orderflow): {symbol}...")
        url = f"{BINANCE_SPOT_URL}/api/v3/klines"
        
        limit_ts = int((datetime.now(timezone.utc) - timedelta(days=DAYS_TO_FETCH)).timestamp() * 1000)
        current_end = self.server_time_s * 1000
        
        all_data = []
        
        while current_end > limit_ts:
            params = {"symbol": symbol, "interval": "5m", "limit": 1000, "endTime": current_end}
            try:
                data = await self.fetch_json(url, params)
                if not data: break
                
                all_data.extend(data)
                
                # Safer Min check
                oldest_ts = min(row[0] for row in data)
                if oldest_ts <= limit_ts: break
                current_end = oldest_ts - 1
                
                await asyncio.sleep(0.1)
            except Exception:
                break
                
        if not all_data: return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # ✅ KEY UPGRADE: Fetch Index 9 (Taker Buy Volume) for CVD
        # Format: [Time, Open, High, Low, Close, Volume, ..., TakerBuyBaseVol]
        # Indexes: 0, 1, 2, 3, 4, 5, ..., 9
        df = df.iloc[:, [0, 1, 2, 3, 4, 5, 9]] 
        df.columns = ["timestamp", "spot_open", "spot_high", "spot_low", "spot_close", "spot_volume", "taker_buy_vol"]
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        for c in df.columns:
            if c != "timestamp": df[c] = pd.to_numeric(df[c], errors="coerce")
            
        df["symbol"] = symbol 
        df["base_asset"] = asset
        return df.sort_values("timestamp")

    # ------------------------------------------------------------------
    # 6. Macro
    # ------------------------------------------------------------------
    async def fetch_macro(self):
        log.info("🌍 Fetching Macro Data...")
        await asyncio.to_thread(self._fetch_macro_sync)

    def _fetch_macro_sync(self):
        start = (datetime.now() - timedelta(days=DAYS_TO_FETCH)).strftime('%Y-%m-%d')
        data = yf.download(["^VIX", "DX-Y.NYB"], start=start, interval="1d", progress=False)
        
        if data.empty: return
        
        if isinstance(data.columns, pd.MultiIndex):
            try: df = data['Close'].reset_index()
            except: df = data.reset_index()
        else:
            df = data.reset_index()
            
        cols_map = {}
        for c in df.columns:
            s = str(c)
            if "VIX" in s: cols_map[c] = "vix_close"
            elif "DX-Y" in s: cols_map[c] = "dxy_close"
            elif s.lower() in ["date", "datetime"]: cols_map[c] = "timestamp"
        df.rename(columns=cols_map, inplace=True)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            
        df = df.set_index("timestamp").resample("5min").ffill().reset_index()
        self._save_df_sync(df, DATA_DIR / "historical_macro.csv")

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _validate_dataframe(self, df, name):
        if df.empty: return False
        if not df["timestamp"].is_monotonic_increasing:
            log.warning(f"⚠️ {name} is not sorted monotonically. Sorting now.")
            df.sort_values("timestamp", inplace=True)
        if df["timestamp"].isna().any():
            log.error(f"❌ {name} contains NaT timestamps. Dropping bad rows.")
            df.dropna(subset=["timestamp"], inplace=True)
        return True

    def _concat_and_save(self, results, filename):
        valid = [df for df in results if not df.empty]
        if valid:
            final = pd.concat(valid)
            subset = ["symbol", "timestamp"] if "symbol" in final.columns else ["timestamp"]
            final = final.sort_values("timestamp").drop_duplicates(subset=subset)
            
            server_dt = datetime.fromtimestamp(self.server_time_s, tz=timezone.utc).replace(tzinfo=None)
            final = final[final["timestamp"] <= server_dt]

            if self._validate_dataframe(final, filename):
                self._save_df_sync(final, DATA_DIR / filename)
        else:
            log.warning(f"⚠️ No data for {filename}")

    def _save_df_sync(self, df, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"💾 Saved {len(df)} rows to {path.name}")

    async def run(self):
        await self._init_session()
        try:
            # 1. Sync Time
            self.server_time_s = await self.get_delta_server_time()
            log.info(f"🕒 Server Time: {datetime.fromtimestamp(self.server_time_s, tz=timezone.utc)}")
            
            # 2. Validate
            if not await self.validate_delta_symbols():
                return
            
            # 3. Parallel Fetch
            await asyncio.gather(
                self.fetch_delta_candles(),
                self.fetch_delta_funding(),
                self.fetch_binance_spot(),
                self.fetch_macro()
            )
        finally:
            await self.close()
            print("\n✅ INSTITUTIONAL FETCH COMPLETE.")

    # ------------------------------------------------------------------
    # 7. Live Trading Methods (Synchronous)
    # ------------------------------------------------------------------
    def fetch_binance_candles_sync(self, symbol="BTCUSDT", interval="5m", limit=1000):
        """
        Synchronous fetch for live bot.
        """
        import requests
        try:
            url = f"{BINANCE_SPOT_URL}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not isinstance(data, list):
                log.error(f"Binance fetch failed: {data}")
                return pd.DataFrame()
                
            cols = [
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_av", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            df = pd.DataFrame(data, columns=cols)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            # Select and rename for continuity
            df = df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]]
            df = df.rename(columns={"taker_buy_base": "taker_buy_vol"})
            return df
        except Exception as e:
            log.error(f"Live Binance Fetch Error: {e}")
            return pd.DataFrame()

    def fetch_delta_candles_sync(self, symbol="BTCUSD", interval="5m", limit=1000):
        """
        Synchronous fetch for live bot (Delta).
        """
        import requests
        try:
            # Helper to map standard intervals to Delta
            # Delta Res: 1m, 5m, 15m, 1h, 4h, 1d
            url = f"{DELTA_BASE_URL}/v2/history/candles"
            
            # Calculate start/end
            # limit * interval_seconds
            # Approximating start time usually fine for 'limit' based APIS, 
            # but Delta requires start/end/limit.
            
            end_ts = int(time.time())
            # interval map
            mins = 5
            if interval == "1m": mins = 1
            elif interval == "1h": mins = 60
            elif interval == "4h": mins = 240
            
            start_ts = end_ts - (limit * mins * 60)
            
            params = {
                "symbol": symbol,
                "resolution": interval,
                "start": start_ts,
                "end": end_ts,
                "limit": limit
            }
            
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not data.get("success"):
                log.error(f"Delta fetch failed: {data}")
                return pd.DataFrame()
                
            result = data.get("result", [])
            if not result: return pd.DataFrame()
            
            df = pd.DataFrame(result)
            # Delta returns: time (unix), open, high, low, close, volume
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df = df.drop(columns=["time"])
            
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                
            return df.sort_values("timestamp")
            
        except Exception as e:
            log.error(f"Live Delta Fetch Error: {e}")
            return pd.DataFrame()

    def get_current_funding_rate_sync(self, symbol="BTCUSD"):
        import requests
        try:
             # Ticker often has funding info or 'mark_price' endpoint
             url = f"{DELTA_BASE_URL}/v2/products/{symbol}"
             resp = requests.get(url, timeout=5)
             data = resp.json()
             if data.get("success"):
                 # Product details might have funding properties or use /v2/tickers
                 # Let's check Ticker
                 pass
             
             # Better: /v2/tickers/{symbol}
             url2 = f"{DELTA_BASE_URL}/v2/tickers/{symbol}"
             resp2 = requests.get(url2, timeout=5)
             data2 = resp2.json()
             if data2.get("success"):
                 # Ticker usually contains 'mark_price', 'funding_rate'??
                 # Delta documentation: Ticker has 'mark_price'. 
                 # Funding rate is often in "contract_type": "perpetual_futures" metadata or separate endpoint.
                 # Let's estimate or return 0 if not found easily.
                 # Wait, fetcher uses /v2/history/candles for funding history.
                 # For live, maybe just use 0 or last known.
                 # Actually, let's use the most recent candle from FUNDING:symbol
                 
                 funding_sym = f"FUNDING:{symbol}"
                 end_ts = int(time.time())
                 start_ts = end_ts - 3600*8
                 
                 params = {
                     "symbol": funding_sym,
                     "resolution": "1h",
                     "start": start_ts,
                     "end": end_ts,
                     "limit": 5
                 }
                 url_hist = f"{DELTA_BASE_URL}/v2/history/candles"
                 resp_h = requests.get(url_hist, params=params)
                 data_h = resp_h.json()
                 if data_h.get("result"):
                     # Last close is current funding rate
                     return float(data_h["result"][-1]["close"])
                     
             return 0.0
        except Exception:
            return 0.0

if __name__ == "__main__":
    if sys.platform == 'win32':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    fetcher = HistoricalDataFetcher()
    asyncio.run(fetcher.run())