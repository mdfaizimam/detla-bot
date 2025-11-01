# --- feature_engine.py ---
# Complete Updated File

import asyncio
import json
import logging
import numpy as np
import time
import re 
from collections import deque 
import pandas as pd 
import pandas_ta_classic as ta
import aiohttp 
from redis import asyncio as aioredis

from config import (
    REDIS_URL, 
    ENRICHED_CHANNEL, 
    TRADING_SYMBOLS, 
    DELTA_BASE_URL, 
    USER_AGENT,
    VOLUME_TIMEFRAME,
    VOLUME_SMA_PERIOD,
    ATR_TIMEFRAME # ✅ NEW
) 

log = logging.getLogger("feature_engine")

# --- Constants for Feature Calculation ---
TFI_LOOKBACK_SECONDS = 5
TRADE_LOG_TTL_SECONDS = 60 
CANDLE_HISTORY_SIZE = 100 

CANDLE_RESOLUTIONS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "30d"]
RESOLUTION_SECONDS = {
    "1m": 60, 
    "5m": 300, 
    "15m": 900, 
    "1h": 3600, 
    "4h": 14400, 
    "1d": 86400,
    "1w": 604800,
    "30d": 2592000 
}


class FeatureEngine:
    """
    Subscribes to raw WS feed (delta:raw:ws) and emits an enriched stream (delta:enriched).
    - Primes candle history via REST API on startup.
    - Maintains a local orderbook via 'l2_updates'.
    - Maintains a local trade log via 'all_trades'.
    - Maintains a cache of the latest candles, mark price, and funding rate.
    - Calculates OBI, Mid-Price, TFI.
    - Calculates Technical Indicators (EMAs, Pivots, Volume SMA, ATR).
    - Calculates S/R levels (PWH/L, PMH/L).
    - Enforces a "readiness" check before publishing.
    """

    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession, top_n=5):
        self.redis = redis_client 
        self.session = http_session 
        self.top_n = top_n
        
        self.order_books = {} 
        self.trade_logs = {}  
        self.features = {} 
        self.candle_history = {} 
        
        self.symbol_ready_state = {}
        
        self.candle_regex = re.compile(r"candlestick_(\w+)")

    async def _publish(self, payload: dict):
        """Publish enriched data to Redis."""
        try:
            await self.redis.publish(ENRICHED_CHANNEL, json.dumps(payload))
            log.info(f"📡 Published enriched → {payload.get('symbol')}: "
                     f"OBI={payload.get('imbalance'):.4f}, " 
                     f"TFI={payload.get('tfi'):.4f}, "
                     f"Mid={payload.get('mid_price')}, "
                     f"Mark={payload.get('mark_price')}, "
                     f"Funding={payload.get('funding_rate')}")
        except Exception as e:
            log.error(f"❌ Failed to publish enriched event: {e}")

    def _initialize_state(self, symbol):
        """Creates new, empty state structures for a symbol."""
        if symbol not in self.order_books:
            self.order_books[symbol] = {"bids": {}, "asks": {}}
            log.info(f"Initialized new order book for {symbol}")
        
        if symbol not in self.trade_logs:
            self.trade_logs[symbol] = []
            log.info(f"Initialized new trade log for {symbol}")

        if symbol not in self.features:
            self.features[symbol] = {
                "obi": 0.0,
                "mid_price": None,
                "tfi": 0.0,
                "last_trade_price": None,
                "mark_price": None,
                "funding_rate": None,
                "tas": {}, 
                "timestamp": 0,
                "PWH": None, "PWL": None,
                "PMH": None, "PML": None,
            }
            log.info(f"Initialized new feature set for {symbol}")
            
        if symbol not in self.candle_history:
            self.candle_history[symbol] = {} 
            for res in CANDLE_RESOLUTIONS:
                self.candle_history[symbol][res] = deque(maxlen=CANDLE_HISTORY_SIZE)
            log.info(f"Initialized new candle cache for {symbol}")
        
        if symbol not in self.symbol_ready_state:
            self.symbol_ready_state[symbol] = {
                "book": False,
                "mark": False,
                "funding": False
            }
            log.info(f"Initialized readiness state for {symbol}")

    def _is_symbol_ready(self, symbol: str) -> bool:
        """Checks if all required data has been received for a symbol."""
        if symbol not in self.symbol_ready_state:
            return False
            
        state = self.symbol_ready_state[symbol]
        
        if state.get("full", False):
            return True
            
        if state["book"] and state["mark"] and state["funding"]:
            log.info(f"✅ {symbol} is now data-ready. Publishing enriched feed.")
            state["full"] = True 
            return True
            
        return False

    async def _prime_candle_history(self):
        """
        Fetches historical candle data from the REST API to seed the engine.
        """
        log.info("Priming candle history for all symbols...")
        end_time = int(time.time())
        
        for symbol in TRADING_SYMBOLS:
            self._initialize_state(symbol) 
            for res in CANDLE_RESOLUTIONS:
                try:
                    duration = RESOLUTION_SECONDS[res]
                    limit = CANDLE_HISTORY_SIZE + 50 
                    start_time = end_time - (limit * duration) 
                    
                    path = "/v2/history/candles" 
                    params = {
                        "symbol": symbol,
                        "resolution": res,
                        "start": str(start_time), 
                        "end": str(end_time),
                        "limit": str(limit)
                    }
                    url = f"{DELTA_BASE_URL}{path}"
                    
                    log.debug(f"Fetching history: {symbol} {res}...")
                    async with self.session.get(url, params=params, headers={'User-Agent': USER_AGENT}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            candles = data.get("result", [])
                            
                            if not candles:
                                log.warning(f"No historical candles found for {symbol} {res}")
                                continue

                            for candle in candles:
                                candle_data = {
                                    "open": float(candle.get("open", 0)),
                                    "high": float(candle.get("high", 0)),
                                    "low": float(candle.get("low", 0)),
                                    "close": float(candle.get("close", 0)),
                                    "volume": float(candle.get("volume", 0)),
                                    "timestamp": candle.get("time", 0) * 1_000_000 # Convert sec to us
                                }
                                self.candle_history[symbol][res].append(candle_data)
                            
                            log.info(f"✅ Primed {len(candles)} candles for {symbol} {res}")
                        else:
                            log.error(f"Failed to fetch history for {symbol} {res}: HTTP {resp.status} {await resp.text()}")
                    
                    await asyncio.sleep(0.3) 
                except Exception as e:
                    log.error(f"Error priming {symbol} {res}: {e}", exc_info=True)
        
        for symbol in TRADING_SYMBOLS:
            self._calculate_technical_indicators(symbol)
            
        log.info("✅ Candle history priming and initial TA calculation complete.")

    # ----------------------------------------------------------------------
    # WebSocket Message Handlers
    # ----------------------------------------------------------------------

    def _handle_l2_snapshot(self, data: dict):
        symbol = data.get("symbol")
        if not symbol: return
        self._initialize_state(symbol)
        self.order_books[symbol]["bids"].clear()
        self.order_books[symbol]["asks"].clear()
        for price_str, size_str in data.get("bids", []):
            self.order_books[symbol]["bids"][price_str] = float(size_str)
        for price_str, size_str in data.get("asks", []):
            self.order_books[symbol]["asks"][price_str] = float(size_str)
        
        if symbol in self.symbol_ready_state:
            self.symbol_ready_state[symbol]["book"] = True
        log.info(f"✅ Order book snapshot received and built for {symbol}")

    def _handle_l2_update(self, data: dict):
        symbol = data.get("symbol")
        if symbol not in self.order_books: return
        for price_str, size_str in data.get("bids", []):
            size = float(size_str)
            if size == 0: self.order_books[symbol]["bids"].pop(price_str, None)
            else: self.order_books[symbol]["bids"][price_str] = size
        for price_str, size_str in data.get("asks", []):
            size = float(size_str)
            if size == 0: self.order_books[symbol]["asks"].pop(price_str, None)
            else: self.order_books[symbol]["asks"][price_str] = size
        
        if symbol in self.symbol_ready_state and not self.symbol_ready_state[symbol]["book"]:
            self.symbol_ready_state[symbol]["book"] = True

    def _handle_all_trades_snapshot(self, data: dict):
        symbol = data.get("symbol")
        if not symbol: return
        self._initialize_state(symbol)
        log.info(f"Processing trade snapshot for {symbol}...")
        for trade in data.get("trades", []): self._log_trade(symbol, trade)
        log.info(f"✅ Trade snapshot processed for {symbol}")

    def _handle_all_trades(self, trade: dict):
        symbol = trade.get("symbol")
        if not symbol: return
        self._log_trade(symbol, trade)
        self._prune_trade_log(symbol) 

    def _log_trade(self, symbol: str, trade: dict):
        if symbol not in self.trade_logs: self._initialize_state(symbol)
        try:
            side = None
            if trade.get("buyer_role") == "taker": side = "buy"
            elif trade.get("seller_role") == "taker": side = "sell"
            if side:
                ts = trade.get("timestamp", 0) / 1_000_000.0
                size = float(trade.get("size", 0))
                price = float(trade.get("price", 0))
                self.trade_logs[symbol].append((ts, side, size))
                self.features[symbol]["last_trade_price"] = price
        except Exception as e: log.warning(f"Could not parse trade data: {e} | Data: {trade}")

    def _prune_trade_log(self, symbol: str):
        if symbol not in self.trade_logs: return
        cutoff_time = time.time() - TFI_LOOKBACK_SECONDS
        first_valid_index = 0
        for i, (ts, _, _) in enumerate(self.trade_logs[symbol]):
            if ts >= cutoff_time:
                first_valid_index = i
                break
        if first_valid_index > 0:
            self.trade_logs[symbol] = self.trade_logs[symbol][first_valid_index:]

    def _handle_candlestick(self, data: dict):
        symbol = data.get("symbol")
        msg_type = data.get("type")
        if not symbol or not msg_type: return
            
        if symbol not in self.candle_history: self._initialize_state(symbol)
            
        match = self.candle_regex.match(msg_type)
        if not match: return
            
        timeframe = match.group(1) 
        if timeframe not in CANDLE_RESOLUTIONS: return 
        
        candle_data = {
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "close": float(data.get("close", 0)),
            "volume": float(data.get("volume", 0)),
            "timestamp": data.get("candle_start_time", 0) 
        }
        
        history_deque = self.candle_history[symbol][timeframe]
        
        is_new_candle = False
        if not history_deque or (history_deque and history_deque[-1]["timestamp"] < candle_data["timestamp"]):
            history_deque.append(candle_data)
            is_new_candle = True
            log.debug(f"Appended new {symbol} {timeframe} candle. History size: {len(history_deque)}")
        elif history_deque and history_deque[-1]["timestamp"] == candle_data["timestamp"]:
            history_deque[-1] = candle_data 
            log.debug(f"Updated {symbol} {timeframe} candle.")
        elif history_deque and candle_data["timestamp"] < history_deque[-1]["timestamp"]:
            log.warning(f"Ignored stale candle for {symbol} {timeframe}")
        
        if is_new_candle:
             self._calculate_technical_indicators(symbol)


    def _handle_funding_rate(self, data: dict):
        symbol = data.get("symbol")
        if not symbol: return
        if symbol not in self.features: self._initialize_state(symbol)
        self.features[symbol]["funding_rate"] = float(data.get("funding_rate", 0))
        
        if symbol in self.symbol_ready_state:
            self.symbol_ready_state[symbol]["funding"] = True

    def _handle_mark_price(self, data: dict, symbol: str): 
        if symbol not in self.features: self._initialize_state(symbol)
        self.features[symbol]["mark_price"] = float(data.get("price", 0))

        if symbol in self.symbol_ready_state:
            self.symbol_ready_state[symbol]["mark"] = True

    # ----------------------------------------------------------------------
    # FEATURE CALCULATION
    # ----------------------------------------------------------------------

    def _calc_imbalance_and_mid(self, symbol: str):
        book = self.order_books.get(symbol)
        if not book or not book["bids"] or not book["asks"]: return 0.0, None 
        try:
            bid_price_keys = sorted(book["bids"].keys(), key=float, reverse=True)
            ask_price_keys = sorted(book["asks"].keys(), key=float)
            if not bid_price_keys or not ask_price_keys: return 0.0, None
            top_n_bid_keys = bid_price_keys[:self.top_n]
            top_n_ask_keys = ask_price_keys[:self.top_n]
            bid_vol = sum(book["bids"][key] for key in top_n_bid_keys)
            ask_vol = sum(book["asks"][key] for key in top_n_ask_keys)
            denom = bid_vol + ask_vol
            obi = (bid_vol - ask_vol) / denom if denom else 0.0
            top_bid = float(bid_price_keys[0])
            top_ask = float(ask_price_keys[0])
            mid_price = (top_bid + top_ask) / 2.0
            return obi, mid_price
        except Exception as e:
            log.error(f"Error calculating OBI for {symbol}: {e}", exc_info=True)
            return 0.0, None
            
    def _calculate_tfi(self, symbol: str):
        if symbol not in self.trade_logs: return 0.0
        buy_vol = 0.0
        sell_vol = 0.0
        cutoff_time = time.time() - TFI_LOOKBACK_SECONDS
        try:
            for (ts, side, size) in reversed(self.trade_logs[symbol]):
                if ts < cutoff_time: break 
                if side == "buy": buy_vol += size
                elif side == "sell": sell_vol += size
            denom = buy_vol + sell_vol
            tfi = (buy_vol - sell_vol) / denom if denom else 0.0
            return tfi
        except Exception as e:
            log.error(f"Error calculating TFI for {symbol}: {e}", exc_info=True)
            return 0.0

    def _calculate_technical_indicators(self, symbol: str):
        """
        Calculates TAs for all timeframes, including Volume,
        Pivots, and Weekly/Monthly S/R levels.
        """
        if symbol not in self.candle_history:
            return 

        tas = {} 
        
        for timeframe, candle_deque in self.candle_history[symbol].items():
            min_candles = max(50, VOLUME_SMA_PERIOD + 2) 
            if len(candle_deque) < min_candles: 
                log.debug(f"Skipping TA for {symbol} {timeframe}, not enough data ({len(candle_deque)} candles)")
                continue

            try:
                df = pd.DataFrame(list(candle_deque))
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
                
                df = df.drop_duplicates(subset=['timestamp']) 
                df = df.set_index('timestamp')
                df.sort_index(inplace=True) 

                # Standard Trend Indicators
                df.ta.ema(length=20, append=True)
                df.ta.ema(length=50, append=True)
                df.ta.rsi(length=14, append=True)
                df.ta.macd(fast=12, slow=26, signal=9, append=True)

                latest_tas = {
                    "ema_20": df['EMA_20'].iloc[-1],
                    "ema_50": df['EMA_50'].iloc[-1],
                    "rsi_14": df['RSI_14'].iloc[-1],
                    "macd_hist": df['MACDh_12_26_9'].iloc[-1],
                    "close": df['close'].iloc[-1],
                    "open": df['open'].iloc[-1],
                }
                
                # --- Volume Filter Calculation ---
                if timeframe == VOLUME_TIMEFRAME:
                    vol_sma_name = f"SMA_volume_{VOLUME_SMA_PERIOD}"
                    df.ta.sma(close='volume', length=VOLUME_SMA_PERIOD, append=True, col_names=(vol_sma_name,))
                    if vol_sma_name in df.columns:
                        latest_tas["volume"] = df['volume'].iloc[-1]
                        latest_tas[vol_sma_name] = df[vol_sma_name].iloc[-1]
                    else:
                        log.warning(f"Could not calculate {vol_sma_name} for {symbol} {timeframe}")

                # --- ✅ NEW: ATR Calculation ---
                if timeframe == ATR_TIMEFRAME:
                    df.ta.atr(length=14, append=True, col_names=("ATR_14",))
                    if "ATR_14" in df.columns:
                        latest_tas["atr"] = df['ATR_14'].iloc[-1]
                    else:
                        log.warning(f"Could not calculate ATR for {symbol} {timeframe}")

                # --- Manual Pivot Point Calculation ---
                if timeframe == "1d":
                    if len(df) > 1:
                        prev_high = df['high'].iloc[-2]
                        prev_low = df['low'].iloc[-2]
                        prev_close = df['close'].iloc[-2]

                        P = (prev_high + prev_low + prev_close) / 3
                        R1 = (2 * P) - prev_low
                        S1 = (2 * P) - prev_high
                        R2 = P + (prev_high - prev_low)
                        S2 = P - (prev_high - prev_low)
                        R3 = prev_high + 2 * (P - prev_low)
                        S3 = prev_low - 2 * (prev_high - P)

                        latest_tas["pivot"] = P
                        latest_tas["R1"] = R1
                        latest_tas["S1"] = S1
                        latest_tas["R2"] = R2
                        latest_tas["S2"] = S2
                        latest_tas["R3"] = R3
                        latest_tas["S3"] = S3
                    else:
                        log.warning(f"Not enough '1d' data to calculate pivots for {symbol}")

                tas[timeframe] = latest_tas
                log.debug(f"Calculated TA for {symbol} {timeframe}")
                
                # --- Weekly/Monthly S/R Calculation ---
                if len(candle_deque) > 1:
                    prev_candle = candle_deque[-2]
                    if timeframe == "1w":
                        self.features[symbol]["PWH"] = prev_candle["high"]
                        self.features[symbol]["PWL"] = prev_candle["low"]
                    elif timeframe == "30d": 
                        self.features[symbol]["PMH"] = prev_candle["high"]
                        self.features[symbol]["PML"] = prev_candle["low"]

            except Exception as e:
                log.error(f"Error calculating TA for {symbol} {timeframe}: {e}", exc_info=True)
        
        self.features[symbol]["tas"] = tas


    async def _publish_features(self, symbol: str, timestamp_us: int):
        """Calculates all features and publishes the combined payload."""
        if symbol not in self.features:
            self._initialize_state(symbol)

        obi, mid_price = self._calc_imbalance_and_mid(symbol)
        if mid_price is not None:
            self.features[symbol]["obi"] = obi
            self.features[symbol]["mid_price"] = mid_price
        tfi = self._calculate_tfi(symbol)
        self.features[symbol]["tfi"] = tfi
        
        # ✅ FIX: This is no longer called here. It's only called on new candles.
        # self._calculate_technical_indicators(symbol)
        
        self.features[symbol]["timestamp"] = timestamp_us

        if self.features[symbol]["mid_price"] is not None:
            payload = {
                "source": "feature_engine",
                "symbol": symbol,
                "timestamp": self.features[symbol]["timestamp"],
                "mid_price": self.features[symbol]["mid_price"],
                "last_trade_price": self.features[symbol]["last_trade_price"],
                "imbalance": self.features[symbol]["obi"], 
                "tfi": self.features[symbol]["tfi"],
                "mark_price": self.features[symbol]["mark_price"],
                "funding_rate": self.features[symbol]["funding_rate"],
                "tas": self.features[symbol]["tas"],
                
                "PWH": self.features[symbol].get("PWH"),
                "PWL": self.features[symbol].get("PWL"),
                "PMH": self.features[symbol].get("PMH"),
                "PML": self.features[symbol].get("PML"),
            }
            await self._publish(payload)
            
    async def start(self):
        """
        Primes historical data first, then subscribes to delta:raw:ws 
        and publishes enriched data to delta:enriched.
        """
        
        try:
            await self._prime_candle_history()
        except Exception as e:
            log.error(f"💥 Failed to prime candle history: {e}", exc_info=True)
            
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("delta:raw:ws")
        log.info("✅ FeatureEngine subscribed to delta:raw:ws and waiting for messages...")

        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message": continue

                try: raw = json.loads(msg.get("data"))
                except Exception as e: log.warning(f"⚠️ JSON parse failed: {e}"); continue

                msg_type = raw.get("type")
                timestamp = raw.get("timestamp", 0)
                symbol = None 
                
                try:
                    if msg_type == "mark_price":
                        raw_sym = raw.get("symbol")
                        if raw_sym: symbol = raw_sym.split(":", 1)[-1] 
                    else:
                        symbol = raw.get("symbol") 
                    
                    if not symbol or symbol not in TRADING_SYMBOLS:
                        continue 
                        
                    should_publish = False

                    if msg_type == "l2_updates":
                        action = raw.get("action")
                        if action == "snapshot": self._handle_l2_snapshot(raw)
                        elif action == "update": self._handle_l2_update(raw)
                        should_publish = True
                    
                    elif msg_type == "all_trades_snapshot":
                        self._handle_all_trades_snapshot(raw)
                    
                    elif msg_type == "all_trades":
                        self._handle_all_trades(raw)
                        should_publish = True
                    
                    elif msg_type.startswith("candlestick_"):
                        self._handle_candlestick(raw)
                        
                    elif msg_type == "funding_rate":
                        self._handle_funding_rate(raw)
                        should_publish = True
                        
                    elif msg_type == "mark_price":
                        self._handle_mark_price(raw, symbol) 
                        should_publish = True

                    elif msg_type == "v2/ticker":
                        log.debug(f"Ticker for {symbol} received and ignored.")
                    
                    if should_publish and self._is_symbol_ready(symbol):
                        await self._publish_features(symbol, timestamp)
                
                except Exception as e:
                    log.error(f"Error processing message type {msg_type} for {symbol}: {e}", exc_info=True)

        except asyncio.CancelledError: log.info("FeatureEngine cancelled.")
        except Exception as e: log.error(f"💥 FeatureEngine crashed: {e}")
        finally: log.info("🔻 FeatureEngine stopped cleanly.")