import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
import orjson
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
import ccxt.async_support as ccxt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
log = logging.getLogger("multimodal_ingestion")

class AsyncMacroFetcher:
    """Fetches Macroeconomic data (DXY, SPX, Yields) using yfinance."""
    
    def __init__(self):
        self.tickers = {
            "dxy": "DX-Y.NYB",
            "spx": "^GSPC",
            "us10y": "^TNX",
            "vix": "^VIX"
        }
        self.cache = {}
        self.last_fetch = 0
        self.cache_duration = 300 # 5 minutes

    async def fetch(self) -> Dict[str, float]:
        now = datetime.now().timestamp()
        if now - self.last_fetch < self.cache_duration and self.cache:
            return self.cache

        try:
            # Run blocking yfinance call in executor
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, self._fetch_sync)
            self.cache = data
            self.last_fetch = now
            return data
        except Exception as e:
            log.error(f"Macro fetch failed: {e}")
            return self.cache if self.cache else {k: 0.0 for k in self.tickers}

    def _fetch_sync(self) -> Dict[str, float]:
        data = {}
        try:
            # Fetch all at once
            tickers_str = " ".join(self.tickers.values())
            df = yf.download(tickers_str, period="1d", interval="1m", progress=False)
            
            # Extract latest close
            for key, symbol in self.tickers.items():
                if symbol in df["Close"]:
                    # Get last valid value
                    val = df["Close"][symbol].dropna().iloc[-1]
                    data[key] = float(val)
                else:
                    data[key] = 0.0
        except Exception as e:
            log.error(f"YFinance sync error: {e}")
        
        return data

class AsyncSentimentFetcher:
    """Fetches Sentiment data: Fear & Greed and VADER analysis."""
    
    def __init__(self):
        self.fng_url = "https://api.alternative.me/fng/"
        self.vader = SentimentIntensityAnalyzer()
        self.last_fng = 50
        self.last_fng_time = 0
        
    async def fetch_fear_and_greed(self) -> int:
        now = datetime.now().timestamp()
        if now - self.last_fng_time < 3600: # Cache for 1 hour
            return self.last_fng
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fng_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        val = int(data['data'][0]['value'])
                        self.last_fng = val
                        self.last_fng_time = now
                        return val
        except Exception as e:
            log.error(f"F&G fetch failed: {e}")
            
        return self.last_fng

    def analyze_text(self, text: str) -> float:
        """Returns compound sentiment score (-1 to 1)"""
        if not text: return 0.0
        return self.vader.polarity_scores(text)['compound']

class AsyncCrossMarketFetcher:
    """Fetches prices from other major exchanges for arbitrage signals."""
    
    def __init__(self):
        self.exchange = ccxt.binanceus({'enableRateLimit': True})  # Use Binance US for public data or generic Binance
        
    async def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        results = {}
        try:
            # We assume symbols are in CCXT format e.g. BTC/USDT
            # For efficiency we might fetchTicker for specific symbols
            for sym in symbols:
                ticker = await self.exchange.fetch_ticker(sym)
                results[f"binance_{sym.replace('/','').lower()}_price"] = float(ticker['last'])
        except Exception as e:
            log.error(f"CrossMarket fetch failed: {e}")
        
        return results
        
    async def close(self):
        await self.exchange.close()

class MultimodalIngestor:
    """
    Main Orchestrator for Real-Time Multimodal Data Ingestion.
    Fusion of:
    1. Macro (DXY, VIX)
    2. Sentiment (F&G, News)
    3. Cross-Market (Binance Prices)
    """
    def __init__(self):
        self.macro = AsyncMacroFetcher()
        self.sentiment = AsyncSentimentFetcher()
        self.cross = AsyncCrossMarketFetcher()
        self.fuser = DataFuser()
        
    async def fetch_snapshot(self) -> Dict[str, Any]:
        """
        Returns a fused dictionary of current market state from all sources.
        """
        # Run fetches in parallel
        # Note: We want cross market for specific coins, let's assume BTC/USDT for now
        
        tasks = [
            self.macro.fetch(),
            self.sentiment.fetch_fear_and_greed(),
            self.cross.fetch_prices(["BTC/USDT", "ETH/USDT"])
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        macro_data = results[0] if isinstance(results[0], dict) else {}
        fng = results[1] if isinstance(results[1], int) else 50
        cross_data = results[2] if isinstance(results[2], dict) else {}
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "meta_fng": fng,
            **macro_data,
            **cross_data
        }
        
        # Add basic normalizations on the fly
        snapshot["fng_norm"] = fng / 100.0
        
        return snapshot

    async def close(self):
        await self.cross.close()

class DataFuser:
    """
    The Fusion Layer: Aligns disparate timeframes and datasets.
    Kept for historical backtesting and alignment logic.
    """
    
    def __init__(self, base_timeframe: str = "5min"):
        self.base_timeframe = base_timeframe
        
    def fuse(self, 
             price_df: pd.DataFrame, 
             sentiment_df: pd.DataFrame, 
             macro_df: pd.DataFrame, 
             onchain_df: pd.DataFrame) -> pd.DataFrame:
        
        log.info("Fusion Started: Aligning all datasets to Price Timeframe...")
        
        # 1. Base Alignment (Price Data is the anchor)
        # Ensure timestamps are datetime
        for df in [price_df, sentiment_df, macro_df, onchain_df]:
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.sort_values("timestamp", inplace=True)
            
        merged = price_df.set_index("timestamp").sort_index()
        
        # 2. Resample & Forward Fill Lower Frequency Data
        if not sentiment_df.empty:
            sent_resampled = sentiment_df.set_index("timestamp").resample(self.base_timeframe).ffill()
            merged = merged.join(sent_resampled, rsuffix="_sent")
            
        if not macro_df.empty:
            macro_resampled = macro_df.set_index("timestamp").resample(self.base_timeframe).ffill()
            merged = merged.join(macro_resampled, rsuffix="_macro")
            
        if not onchain_df.empty:
            onchain_resampled = onchain_df.set_index("timestamp").resample(self.base_timeframe).ffill()
            merged = merged.join(onchain_resampled, rsuffix="_onchain")
        
        # 3. Handle Missing Values
        merged.bfill(inplace=True)
        merged.fillna(0, inplace=True) 
        
        log.info(f"Fusion Complete. Shape: {merged.shape}")
        return merged.reset_index()

class LocalFileData:
    """Ingests REAL data from local CSV files (Legacy support for Backtesting)."""
    DATA_DIR = "data"
    
    @staticmethod
    def load_price_data(filename="historical_candles.csv") -> pd.DataFrame:
        # Implementation preserved for backward compatibility
        import os
        path = os.path.join(LocalFileData.DATA_DIR, filename)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "time" in df.columns: df.rename(columns={"time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

if __name__ == "__main__":
    # Test Async Ingestion
    async def test_run():
        ingestor = MultimodalIngestor()
        log.info("Fetching Multimodal Snapshot...")
        data = await ingestor.fetch_snapshot()
        log.info(f"Snapshot: {orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()}")
        await ingestor.close()
        
    asyncio.run(test_run())
