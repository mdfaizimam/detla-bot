# --- historical_data_fetcher.py (FIXED VERSION) ---

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
    "BTCUSD": {"delta": "BTCUSD", "coincodex": "Bitcoin", "binance": "BTCUSDT", "output": "BTCUSD"},
    "ETHUSD": {"delta": "ETHUSD", "coincodex": "Ethereum", "binance": "ETHUSDT", "output": "ETHUSD"},
    "SOLUSD": {"delta": "SOLUSD", "coincodex": "Solana", "binance": "SOLUSDT", "output": "SOLUSD"}
}
API_LIMIT_DELTA = 2000
HTTP_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.5
OUTLIER_MULTIPLIER = 15
MAX_CONCURRENT_SYMBOLS = 2
MAX_FAILED_CHUNKS = 5

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
ERROR_LOG_DIR = LOG_DIR / "errors"
ERROR_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "main.log", encoding="utf-8")
    ]
)
log = logging.getLogger("data_fetcher")

# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------
class APIRateLimitError(Exception): ...
class APINotAvailableError(Exception): ...

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.start_time: Optional[datetime] = None
        self.chunk_timings: List[float] = []
        self._lock: Optional[asyncio.Lock] = None

    async def _ensure_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def record_request(self, latency: float, success: bool = True):
        await self._ensure_lock()
        async with self._lock:
            self.total_requests += 1
            self.total_latency += float(latency or 0.0)
            self.chunk_timings.append(float(latency or 0.0))
            if not success:
                self.failed_requests += 1

    def get_avg_latency(self) -> float:
        return (self.total_latency / self.total_requests) if self.total_requests > 0 else 0.0

    def get_success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.failed_requests) / self.total_requests

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical Data Fetcher")
    parser.add_argument("--auto", choices=["yes", "no"], default="no",
                        help="Auto-start without prompt (yes/no)")
    parser.add_argument("--no-input", action="store_true",
                        help="Non-interactive mode (equivalent to --auto yes)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help=f"Symbols to fetch (default: {DEFAULT_TRADING_SYMBOLS})")
    parser.add_argument("--days", type=int, default=None,
                        help=f"Days of historical data to fetch (default: {DEFAULT_DAYS_TO_FETCH})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory for data files (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--test", action="store_true",
                        help="Run API verification only (no data fetching)")
    return parser.parse_args()

# ----------------------------------------------------------------------
# Retry Decorator
# ----------------------------------------------------------------------
def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
                retry_exceptions=(aiohttp.ClientError, asyncio.TimeoutError, APIRateLimitError)):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        wait = min(current_delay * 2, 60) if isinstance(e, APIRateLimitError) else current_delay
                        log.warning(f"{func.__name__}: attempt {attempt+1}/{max_attempts} failed ({e}); sleeping {wait}s")
                        await asyncio.sleep(wait)
                        current_delay *= backoff
                    else:
                        log.error(f"{func.__name__}: exhausted retries; last error: {e}")
                        raise last_exc
            raise last_exc if last_exc else RuntimeError(f"Retry failed unexpectedly for {func.__name__}")
        return wrapper
    return decorator

# ----------------------------------------------------------------------
# API Health Check
# ----------------------------------------------------------------------
async def test_endpoint(session: aiohttp.ClientSession, name: str, url: str, 
                        params=None, metrics: Optional[Metrics]=None) -> bool:
    t0 = time.time()
    success = False
    try:
        async with session.get(url, params=params) as resp:
            success = (resp.status == 200)
            if resp.status == 429:
                log.warning(f"[{name}] Rate limited")
            elif not success:
                log.warning(f"[{name}] HTTP {resp.status}")
            return success
    except Exception as e:
        log.warning(f"[{name}] error: {e}")
        return False
    finally:
        if metrics:
            latency = time.time() - t0
            await metrics.record_request(latency, success=success)

async def verify_api_sources(session: aiohttp.ClientSession, metrics: Metrics):
    log.info("Testing API endpoints...")
    tests = [
        test_endpoint(session, "Delta", f"{DELTA_BASE_URL}/v2/tickers", metrics=metrics),
        test_endpoint(session, "CoinCodex", "https://coincodex.com/api/coincodex/get_coin_history_by_name/Bitcoin/", metrics=metrics),
        test_endpoint(session, "Binance", "https://fapi.binance.com/fapi/v1/fundingRate",
                      params={"symbol": "BTCUSDT", "limit": 1}, metrics=metrics),
    ]
    results = await asyncio.gather(*tests)
    providers = ["Delta", "CoinCodex", "Binance"]
    status = dict(zip(providers, results))

    print("\nAPI STATUS")
    print("-" * 40)
    for p, ok in status.items():
        status_text = "Working" if ok else "Unavailable"
        print(f"{p:<12} -> {status_text}")
    print("-" * 40)

    if not any(status.values()):
        raise APINotAvailableError("No APIs available")
    return [p for p, ok in status.items() if ok], status

# ----------------------------------------------------------------------
# Delta Fetcher - CANDLES
# ----------------------------------------------------------------------
@async_retry()
async def fetch_candle_chunk(session: aiohttp.ClientSession, symbol: str, resolution: str,
                             start_time: int, end_time: int, limit: int = API_LIMIT_DELTA) -> List[Dict]:
    params = {
        "symbol": symbol, "resolution": resolution,
        "start": str(start_time), "end": str(end_time), "limit": str(limit)
    }
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    async with session.get(url, params=params, headers={"User-Agent": USER_AGENT}) as resp:
        if resp.status == 429:
            raise APIRateLimitError(f"Rate limited: {symbol}")
        if resp.status != 200:
            text = (await resp.text())[:200]
            log.error(f"Delta {symbol} HTTP {resp.status}: {text}")
            return []
        try:
            data = await resp.json(content_type=None)
        except (aiohttp.ContentTypeError, json.JSONDecodeError):
            text = (await resp.text())[:200]
            log.error(f"Delta {symbol}: invalid JSON: {text}")
            return []
        result = data.get("result", [])
        if result:
            df = pd.DataFrame(result)
            tcol = "time" if "time" in df.columns else "timestamp"
            if tcol in df.columns:
                df = df.drop_duplicates(subset=[tcol])
            result = df.to_dict("records")
        return result

# ----------------------------------------------------------------------
# Binance Fetcher - FUNDING RATES
# ----------------------------------------------------------------------
@async_retry()
async def fetch_funding_rates(session: aiohttp.ClientSession, symbol: str, 
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None,
                              limit: int = 1000) -> List[Dict]:
    binance_symbol = SYMBOL_MAPPING.get(symbol, {}).get("binance", f"{symbol[:-3]}USDT")
    
    params = {"symbol": binance_symbol, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
    async with session.get(url, params=params) as resp:
        if resp.status == 429:
            raise APIRateLimitError(f"Binance rate limited: {symbol}")
        if resp.status != 200:
            text = (await resp.text())[:200]
            log.error(f"Binance funding {symbol} HTTP {resp.status}: {text}")
            return []
        try:
            data = await resp.json()
            if isinstance(data, list):
                return data
            return []
        except (aiohttp.ContentTypeError, json.JSONDecodeError):
            text = (await resp.text())[:200]
            log.error(f"Binance funding {symbol}: invalid JSON: {text}")
            return []

# ----------------------------------------------------------------------
# Binance Fetcher - LONG/SHORT RATIO (FIXED - 30-day limit handling)
# ----------------------------------------------------------------------
@async_retry()
async def fetch_long_short_ratio(session: aiohttp.ClientSession, symbol: str,
                                 period: str = "5m",
                                 start_time: Optional[int] = None,
                                 end_time: Optional[int] = None,
                                 limit: int = 500) -> List[Dict]:
    """
    Fetch long/short ratio from Binance with proper 30-day limit handling.
    Binance only provides max 30 days of historical L/S ratio data.
    """
    binance_symbol = SYMBOL_MAPPING.get(symbol, {}).get("binance", f"{symbol[:-3]}USDT")
    
    # Binance restriction: cannot fetch data older than 30 days
    max_history_days = 30
    if start_time:
        # Convert to datetime for comparison
        start_dt = datetime.fromtimestamp(start_time / 1000)
        thirty_days_ago = datetime.now() - timedelta(days=max_history_days)
        
        # If requesting data older than 30 days, adjust start_time
        if start_dt < thirty_days_ago:
            log.warning(f"{symbol}: Binance L/S ratio limited to {max_history_days} days. Adjusting timeframe.")
            start_time = int(thirty_days_ago.timestamp() * 1000)
    
    params = {"symbol": binance_symbol, "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    url = f"{BINANCE_FUTURES_URL}/futures/data/globalLongShortAccountRatio"
    async with session.get(url, params=params) as resp:
        if resp.status == 429:
            raise APIRateLimitError(f"Binance L/S ratio rate limited: {symbol}")
        if resp.status == 400:
            # Handle "Invalid parameter" for dates beyond 30 days
            error_text = await resp.text()
            if "Invalid parameter" in error_text or "startTime" in error_text:
                log.warning(f"{symbol}: Binance L/S ratio - data older than {max_history_days} days not available")
                return []
            else:
                log.error(f"Binance L/S ratio {symbol} HTTP 400: {error_text[:200]}")
                return []
        if resp.status != 200:
            text = (await resp.text())[:200]
            log.error(f"Binance L/S ratio {symbol} HTTP {resp.status}: {text}")
            return []
        try:
            data = await resp.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'msg' in data:
                log.warning(f"Binance L/S ratio {symbol}: {data.get('msg')}")
                return []
            return []
        except (aiohttp.ContentTypeError, json.JSONDecodeError):
            text = (await resp.text())[:200]
            log.error(f"Binance L/S ratio {symbol}: invalid JSON: {text}")
            return []

# ----------------------------------------------------------------------
# CoinCodex Fetcher (Fallback)
# ----------------------------------------------------------------------
class CoinCodexFetcher:
    BASE_URL = "https://coincodex.com/api/coincodex/get_coin_history_by_name/"

    @async_retry()
    async def fetch_historical(self, session: aiohttp.ClientSession, coin_name="Bitcoin",
                               output_symbol="BTCUSD", resample_to: Optional[str]=None) -> pd.DataFrame:
        async with session.get(f"{self.BASE_URL}{coin_name}/") as resp:
            if resp.status != 200:
                log.error(f"CoinCodex HTTP {resp.status}")
                return pd.DataFrame()
            try:
                data = await resp.json(content_type=None)
            except (aiohttp.ContentTypeError, json.JSONDecodeError):
                text = (await resp.text())[:200]
                log.error(f"CoinCodex {coin_name}: invalid JSON: {text}")
                return pd.DataFrame()

            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if not isinstance(data, list):
                return pd.DataFrame()

            df = pd.DataFrame(data)
            colmap = {"timestamp": "time", "high": "high", "low": "low", "open": "open", "close": "close", "volume": "volume"}
            for src, dst in colmap.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]

            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")

            df = df.dropna(subset=["time"]).sort_values("time")
            df["symbol"] = output_symbol

            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if len(df) > 1 and resample_to:
                df = self._resample(df, resample_to)
            return df

    def _resample(self, df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        try:
            dfr = df.set_index("time").resample(resolution).agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "symbol": "first"
            }).ffill().dropna(subset=["open", "high", "low", "close"])
            return dfr.reset_index()
        except Exception as e:
            log.error(f"Resample failed: {e}")
            return df.reset_index() if "time" not in df.columns else df

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_ohlc_consistency(df: pd.DataFrame) -> pd.Series:
    req = {"open", "high", "low", "close"}
    if not req.issubset(df.columns):
        return pd.Series(True, index=df.index)
    return ((df["low"] <= df["open"]) & (df["low"] <= df["close"]) &
            (df["high"] >= df["open"]) & (df["high"] >= df["close"]))

def validate_and_clean_candle_data(candles: List[Dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)

    tcol = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if not tcol:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df[tcol], unit="s", errors="coerce")
    df = df.dropna(subset=["time"]).drop_duplicates("time")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pricecols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for col in pricecols:
        series = df[col]
        if series.notna().any():
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lb, ub = Q1 - OUTLIER_MULTIPLIER * IQR, Q3 + OUTLIER_MULTIPLIER * IQR
                mask &= series.between(lb, ub)

    df = df[mask]
    df = df[validate_ohlc_consistency(df)]
    return df.sort_values("time")

# ----------------------------------------------------------------------
# Merge with Fallback
# ----------------------------------------------------------------------
async def merge_delta_and_fallback(session: aiohttp.ClientSession, delta_candles: List[Dict],
                                     symbol: str = "BTCUSD") -> pd.DataFrame:
    delta_df = validate_and_clean_candle_data(delta_candles)
    if delta_df.empty or len(delta_df) < 100:
        log.warning(f"{symbol}: Delta data insufficient ({len(delta_df)} rows), trying fallback...")
        cfg = SYMBOL_MAPPING.get(symbol, {})
        fb = await CoinCodexFetcher().fetch_historical(
            session, cfg.get("coincodex", "Bitcoin"), cfg.get("output", symbol), resample_to=TRAINING_TIMEFRAME
        )
        return fb if not fb.empty else delta_df
    return delta_df

# ----------------------------------------------------------------------
# Fetch All Candles for Symbol
# ----------------------------------------------------------------------
async def fetch_all_candles_for_symbol(session: aiohttp.ClientSession, symbol: str,
                                         data_dir: Path, total_days: int,
                                         metrics: Optional[Metrics] = None) -> pd.DataFrame:
    log.info(f"Starting candle fetch for {symbol}...")
    tdir = data_dir / "temp" / symbol
    tdir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []

    end_time = datetime.now()
    start_time = end_time - timedelta(days=total_days)
    current_time = end_time
    total_chunks = max(1, int(np.ceil((end_time - start_time).days / FETCH_CHUNK_DAYS)))
    failed_chunks = 0
    symbol_start = datetime.now()
    chunk_num = 0

    error_logger = logging.getLogger(f"error.{symbol}")
    error_file = ERROR_LOG_DIR / f"{symbol}_errors.log"
    
    handler_exists = False
    for handler in error_logger.handlers:
        if (isinstance(handler, logging.FileHandler) and 
            hasattr(handler, 'baseFilename') and 
            str(error_file) in handler.baseFilename):
            handler_exists = True
            break
    
    if not handler_exists:
        fh = logging.FileHandler(error_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
        error_logger.addHandler(fh)
        error_logger.setLevel(logging.ERROR)

    try:
        while current_time > start_time and failed_chunks < MAX_FAILED_CHUNKS:
            chunk_num += 1
            chunk_start = max(start_time, current_time - timedelta(days=FETCH_CHUNK_DAYS))
            st_ts, end_ts = int(chunk_start.timestamp()), int(current_time.timestamp())

            try:
                t0 = time.time()
                candles = await fetch_candle_chunk(session, symbol, TRAINING_TIMEFRAME, st_ts, end_ts)
                latency = time.time() - t0
                if metrics:
                    await metrics.record_request(latency, success=bool(candles))

                if candles:
                    df = validate_and_clean_candle_data(candles)
                    if not df.empty:
                        fpath = tdir / f"chunk_{st_ts}_{end_ts}.csv"
                        df.to_csv(fpath, index=False)
                        files.append(fpath)

                        progress_pct = ((end_time - current_time).days / max(1, total_days)) * 100
                        elapsed = (datetime.now() - symbol_start).total_seconds()
                        eta_str = "calculating..."
                        if progress_pct > 0:
                            eta_sec = (elapsed / progress_pct) * (100 - progress_pct)
                            eta_str = f"{eta_sec/60:.1f}m"
                        log.info(f"{symbol}: Chunk {chunk_num}/{total_chunks} ({progress_pct:.1f}%) - {len(df)} rows - ETA: {eta_str}")

                current_time = chunk_start
                await asyncio.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                if metrics:
                    await metrics.record_request(0, success=False)
                log.error(f"{symbol}: chunk {chunk_num} failed: {e}")
                error_logger.error(f"Chunk {chunk_num} ({chunk_start.date()} → {current_time.date()}): {e}")
                failed_chunks += 1
                await asyncio.sleep(2)

        if failed_chunks >= MAX_FAILED_CHUNKS and files:
            log.warning(f"{symbol}: circuit breaker after {failed_chunks} failures; using {len(files)} collected chunks")

        dfs: List[pd.DataFrame] = []
        for f in sorted(files, key=lambda p: p.name):
            try:
                chunk_df = pd.read_csv(f, parse_dates=["time"])
                if not chunk_df.empty:
                    dfs.append(chunk_df)
            except Exception as e:
                log.warning(f"{symbol}: failed to read {f.name}: {e}")

        if not dfs:
            log.warning(f"{symbol}: no data collected")
            return pd.DataFrame()

        try:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates("time").sort_values("time")
        except ValueError as e:
            log.error(f"{symbol}: failed to merge chunks: {e}")
            return pd.DataFrame()

        log.info(f"{symbol}: merged {len(combined)} rows from {len(dfs)} chunks")

        if len(combined) > 1:
            expected = pd.Timedelta(minutes=5)
            max_gap = combined["time"].diff().max()
            if pd.notna(max_gap) and max_gap > expected * 2:
                log.warning(f"{symbol}: large time gap detected: {max_gap}")

        return await merge_delta_and_fallback(session, combined.to_dict("records"), symbol)

    finally:
        try:
            shutil.rmtree(tdir, ignore_errors=True)
        except Exception as e:
            log.warning(f"{symbol}: temp cleanup failed: {e}")
        for handler in error_logger.handlers[:]:
            handler.close()
            error_logger.removeHandler(handler)

# ----------------------------------------------------------------------
# Fetch Sentiment Data (UPDATED with L/S ratio limitations)
# ----------------------------------------------------------------------
async def fetch_sentiment_data(session: aiohttp.ClientSession, symbols: List[str],
                               total_days: int, metrics: Metrics) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Starting sentiment data fetch...")
    
    all_funding_rates = []
    all_long_short_ratios = []
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=total_days)
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    # For L/S ratio: Binance only provides 30 days max
    ls_start_time = datetime.now() - timedelta(days=30)
    ls_start_ts = int(ls_start_time.timestamp() * 1000)
    
    for symbol in symbols:
        log.info(f"Fetching sentiment data for {symbol}...")
        
        # Fetch funding rates (full history available)
        try:
            t0 = time.time()
            funding_data = await fetch_funding_rates(session, symbol, start_ts, end_ts, limit=1000)
            latency = time.time() - t0
            if metrics:
                await metrics.record_request(latency, success=bool(funding_data))
                
            if funding_data:
                for item in funding_data:
                    item["symbol"] = symbol
                    item["fundingTime"] = pd.to_datetime(item["fundingTime"], unit="ms")
                    item["fundingRate"] = float(item["fundingRate"])
                all_funding_rates.extend(funding_data)
                log.info(f"{symbol}: fetched {len(funding_data)} funding rates")
            await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            log.error(f"{symbol}: funding rate fetch failed: {e}")
            if metrics:
                await metrics.record_request(0, success=False)
        
        # Fetch long/short ratio (limited to 30 days)
        try:
            t0 = time.time()
            ls_data = await fetch_long_short_ratio(session, symbol, "5m", ls_start_ts, end_ts, limit=500)
            latency = time.time() - t0
            if metrics:
                await metrics.record_request(latency, success=bool(ls_data))
                
            if ls_data:
                for item in ls_data:
                    item["symbol"] = symbol
                    item["timestamp"] = pd.to_datetime(item["timestamp"], unit="ms")
                    item["longShortRatio"] = float(item["longShortRatio"])
                    item["longAccount"] = float(item["longAccount"])
                    item["shortAccount"] = float(item["shortAccount"])
                all_long_short_ratios.extend(ls_data)
                log.info(f"{symbol}: fetched {len(ls_data)} long/short ratios (last 30 days only)")
            else:
                log.warning(f"{symbol}: no long/short ratio data available (Binance 30-day limit)")
            await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            log.error(f"{symbol}: long/short ratio fetch failed: {e}")
            if metrics:
                await metrics.record_request(0, success=False)
    
    # Create DataFrames
    funding_df = pd.DataFrame(all_funding_rates) if all_funding_rates else pd.DataFrame()
    ls_df = pd.DataFrame(all_long_short_ratios) if all_long_short_ratios else pd.DataFrame()
    
    if not funding_df.empty:
        funding_df = funding_df.sort_values(["symbol", "fundingTime"]).reset_index(drop=True)
        log.info(f"Total funding rates: {len(funding_df)} rows")
    
    if not ls_df.empty:
        ls_df = ls_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        # Check date range of L/S data
        if not ls_df.empty:
            min_date = ls_df["timestamp"].min()
            max_date = ls_df["timestamp"].max()
            log.info(f"L/S ratio date range: {min_date} to {max_date} ({len(ls_df)} rows)")
    else:
        log.warning("No long/short ratio data collected (Binance API limitation)")
    
    return funding_df, ls_df

# ----------------------------------------------------------------------
# Save Functions
# ----------------------------------------------------------------------
async def clean_and_save_candles_async(df: pd.DataFrame, data_dir: Path) -> bool:
    if df.empty:
        log.warning("No candle data to save")
        return False

    df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    if df.empty:
        log.error("No valid candle data after cleaning")
        return False

    output_file = data_dir / "historical_candles.csv"
    try:
        buf = StringIO()
        df.to_csv(buf, index=False)
        csv_content = buf.getvalue()
        buf.close()

        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(csv_content)

        log.info(f"✅ Saved {len(df)} candle rows → {output_file}")
        return True
    except Exception as e:
        log.error(f"Candle save failed: {e}")
        return False

async def save_sentiment_data_async(funding_df: pd.DataFrame, ls_df: pd.DataFrame, data_dir: Path) -> bool:
    success = True
    
    # Save funding rates
    if not funding_df.empty:
        output_file = data_dir / "historical_funding_rates.csv"
        try:
            buf = StringIO()
            funding_df.to_csv(buf, index=False)
            csv_content = buf.getvalue()
            buf.close()

            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(csv_content)
            log.info(f"✅ Saved {len(funding_df)} funding rate rows → {output_file}")
        except Exception as e:
            log.error(f"Funding rate save failed: {e}")
            success = False
    else:
        log.warning("No funding rate data to save")
    
    # Save long/short ratios with limitation notice
    if not ls_df.empty:
        output_file = data_dir / "historical_long_short_ratio.csv"
        try:
            buf = StringIO()
            ls_df.to_csv(buf, index=False)
            csv_content = buf.getvalue()
            buf.close()

            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(csv_content)
            
            # Add warning about data limitations
            min_date = ls_df["timestamp"].min()
            max_date = ls_df["timestamp"].max()
            log.info(f"✅ Saved {len(ls_df)} long/short ratio rows → {output_file}")
            log.warning(f"📅 L/S Ratio Data Range: {min_date.date()} to {max_date.date()} (Binance 30-day limit)")
            
        except Exception as e:
            log.error(f"Long/short ratio save failed: {e}")
            success = False
    else:
        log.warning("❌ No long/short ratio data available (Binance API limitation)")
        # Create empty file with header to avoid errors in train_model.py
        try:
            empty_df = pd.DataFrame(columns=["symbol", "timestamp", "longShortRatio", "longAccount", "shortAccount"])
            empty_df.to_csv(data_dir / "historical_long_short_ratio.csv", index=False)
            log.info("Created empty L/S ratio file to maintain compatibility")
        except Exception as e:
            log.error(f"Failed to create empty L/S ratio file: {e}")
    
    return success

# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
async def fetch_single_symbol(session: aiohttp.ClientSession, symbol: str, sem: asyncio.Semaphore,
                              data_dir: Path, days_to_fetch: int, metrics: Metrics) -> Tuple[str, pd.DataFrame]:
    async with sem:
        logger = logging.getLogger(f"fetcher.{symbol}")
        logger.propagate = False
        fh = logging.FileHandler(LOG_DIR / f"{symbol}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"))
        logger.addHandler(fh)
        try:
            df = await fetch_all_candles_for_symbol(session, symbol, data_dir, days_to_fetch, metrics)
            if not df.empty:
                df["symbol"] = symbol # <-- THIS IS THE FIX
            return symbol, df
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return symbol, pd.DataFrame()
        finally:
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)

async def fetch_all_data(session: aiohttp.ClientSession, symbols: List[str],
                         data_dir: Path, days_to_fetch: int, metrics: Metrics) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Fetch candles concurrently
    sem = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
    candle_tasks = [fetch_single_symbol(session, s, sem, data_dir, days_to_fetch, metrics) for s in symbols]
    candle_results = await asyncio.gather(*candle_tasks, return_exceptions=True)
    
    # Combine all candle data
    all_candles = []
    for result in candle_results:
        if isinstance(result, Exception):
            log.error(f"Symbol task failed: {result}")
            continue
        sym, df = result
        if not df.empty:
            all_candles.append(df)
    
    combined_candles = pd.concat(all_candles, ignore_index=True) if all_candles else pd.DataFrame()
    
    # Fetch sentiment data
    funding_df, ls_df = await fetch_sentiment_data(session, symbols, days_to_fetch, metrics)
    
    return combined_candles, funding_df, ls_df

# ----------------------------------------------------------------------
# Async user input
# ----------------------------------------------------------------------
async def async_user_input(prompt: str) -> str:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, input, prompt)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return await loop.run_in_executor(None, input, prompt)
        finally:
            loop.close()

# ----------------------------------------------------------------------
# Run Fetcher
# ----------------------------------------------------------------------
async def run_fetcher(symbols=None, days=None, auto_start=False, output_dir=None, test_mode=False):
    effective_symbols = symbols if symbols is not None else DEFAULT_TRADING_SYMBOLS
    effective_days = days if days is not None else DEFAULT_DAYS_TO_FETCH
    effective_data_dir = Path(output_dir) if output_dir is not None else DEFAULT_DATA_DIR
    effective_data_dir.mkdir(exist_ok=True)

    metrics = Metrics()
    metrics.start_time = datetime.now()

    try:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_SYMBOLS + 2)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            try:
                working, api_status = await verify_api_sources(session, metrics)
                log.info(f"Available APIs: {working}")
                
                # Display data limitations upfront
                print("\n📊 DATA AVAILABILITY NOTICE:")
                print("-" * 40)
                print("✅ Candles: Full 1-year history")
                print("✅ Funding Rates: Full available history") 
                print("⚠️  L/S Ratios: Last 30 days only (Binance limitation)")
                print("❌ Liquidations: Not available (endpoint retired)")
                print("-" * 40)
                
                if test_mode:
                    log.info("Test mode completed.")
                    return {
                        "test_mode": True,
                        "api_status": api_status,
                        "working_apis": working,
                        "metrics": {
                            "total_requests": metrics.total_requests,
                            "success_rate": metrics.get_success_rate(),
                            "avg_latency": round(metrics.get_avg_latency(), 4)
                        }
                    }
            except APINotAvailableError:
                log.critical("No APIs available")
                return {}

            if not auto_start:
                choice = await async_user_input("\nStart fetching? (yes/no): ")
                if choice.strip().lower() not in ["y", "yes"]:
                    log.info("Cancelled by user")
                    return {}

            log.info("Starting comprehensive data fetch...")
            candles_df, funding_df, ls_df = await fetch_all_data(session, effective_symbols, effective_data_dir, effective_days, metrics)
            
            # Save all data
            candle_success = await clean_and_save_candles_async(candles_df, effective_data_dir)
            sentiment_success = await save_sentiment_data_async(funding_df, ls_df, effective_data_dir)
            
            total_time = (datetime.now() - metrics.start_time).total_seconds()

            summary = {
                "candles": {
                    "rows": len(candles_df),
                    "symbols": effective_symbols,
                    "success": candle_success
                },
                "sentiment": {
                    "funding_rates_rows": len(funding_df),
                    "long_short_ratio_rows": len(ls_df),
                    "ls_ratio_date_range": {
                        "min": ls_df["timestamp"].min().isoformat() if not ls_df.empty else None,
                        "max": ls_df["timestamp"].max().isoformat() if not ls_df.empty else None
                    } if not ls_df.empty else "No data (30-day limit)",
                    "success": sentiment_success
                },
                "limitations": {
                    "ls_ratio_max_days": 30,
                    "liquidations_available": False
                },
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "success_rate": metrics.get_success_rate(),
                    "avg_latency": metrics.get_avg_latency(),
                    "duration_minutes": total_time / 60
                },
                "config": {
                    "symbols": effective_symbols,
                    "days": effective_days,
                    "output_dir": str(effective_data_dir)
                },
                "timestamp": datetime.now().isoformat()
            }

            summary_file = LOG_DIR / "run_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            log.info(f"Run summary saved → {summary_file}")

            # Final summary
            print("\n🎯 FETCHING COMPLETE")
            print("-" * 40)
            print(f"✅ Candles: {len(candles_df)} rows")
            print(f"✅ Funding Rates: {len(funding_df)} rows")
            print(f"⚠️  L/S Ratios: {len(ls_df)} rows (30-day limit)")
            print(f"⏱️  Duration: {total_time/60:.1f} minutes")
            print("-" * 40)

            return summary

    except KeyboardInterrupt:
        log.info("Cancelled by user")
    except Exception as e:
        log.error(f"Fatal error: {e}")
    return {}

# ----------------------------------------------------------------------
# CLI Entry
# ----------------------------------------------------------------------
async def main():
    args = parse_arguments()
    
    # Check if user wants only BTC
    if args.symbols and len(args.symbols) == 1 and args.symbols[0].upper() == 'BTC':
        symbols = ['BTCUSD']
    else:
        symbols = args.symbols or DEFAULT_TRADING_SYMBOLS

    days = args.days or DEFAULT_DAYS_TO_FETCH
    output_dir = args.output_dir or DEFAULT_DATA_DIR
    auto_start = args.auto == "yes" or args.no_input
    
    await run_fetcher(
        symbols=symbols,
        days=days,
        auto_start=auto_start,
        output_dir=output_dir,
        test_mode=args.test
    )

if __name__ == "__main__":
    asyncio.run(main())