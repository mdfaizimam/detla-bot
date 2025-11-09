# --- feature_engine.py ---
# FIX: Replaced the 'is_priming' and 'is_processing_buffer' flags
# with a single asyncio.Lock (_priming_lock) to create an atomic
# "priming + buffer processing" state, fixing the race condition.

import asyncio
import json
import logging
import numpy as np
import time
import re 
import zlib 
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
    ATR_TIMEFRAME,
    SPOT_INDEX_SYMBOLS,
    CONTROL_CHANNEL,
    LATEST_ENRICHED_KEY 
) 

log = logging.getLogger("feature_engine")

# --- Constants for Feature Calculation ---
TFI_LOOKBACK_SECONDS = 5
TRADE_LOG_TTL_SECONDS = 60 
CANDLE_HISTORY_SIZE = 100 

CANDLE_RESOLUTIONS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
RESOLUTION_SECONDS = {
    "1m": 60, 
    "5m": 300, 
    "15m": 900, 
    "1h": 3600, 
    "4h": 14400, 
    "1d": 86400,
    "1w": 604800,
}


class FeatureEngine:
    """
    Subscribes to raw WS feed (delta:raw:ws) and emits an enriched stream (delta:enriched).
    - Includes L2 Checksum and Sequence validation.
    """

    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession, top_n=5):
        self.redis = redis_client 
        self.session = http_session # Used for unauthenticated candle priming
        self.top_n = top_n
        
        self.order_books = {} 
        self.trade_logs = {}  
        self.features = {} 
        self.candle_history = {} 
        self.sequence_numbers = {} # Track sequence numbers for L2 updates
        
        self.symbol_ready_state = {}
        
        self.candle_regex = re.compile(r"candlestick_(\w+)")
        
        # --- FIX: State flags for handling priming race condition ---
        self._stop_flag = False
        self._message_buffer = deque()
        self._priming_lock = asyncio.Lock() # <-- FIX: New lock
        # --- End Fix ---


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
            self.order_books[symbol] = {
                "bids": {}, 
                "asks": {},
                "is_awaiting_snapshot": True  # State to control logging
            }
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
                "spot_price": None, 
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
            
        if symbol not in self.sequence_numbers:
            self.sequence_numbers[symbol] = -1 # Initialize sequence number

    def _is_symbol_ready(self, symbol: str) -> bool:
        """Checks if all required data has been received for a symbol."""
        # <-- FIX: Check lock
        if self._priming_lock.locked(): 
            return False
            
        if symbol not in self.symbol_ready_state:
            return False
            
        state = self.symbol_ready_state[symbol]
        
        if state.get("full", False):
            return True
            
        # We need the core market data to be ready
        if state["book"] and state["mark"] and state["funding"]:
            log.info(f"✅ {symbol} is now data-ready. Publishing enriched feed.")
            state["full"] = True 
            return True
            
        return False
        
    async def _publish_resubscribe_request(self, symbol: str):
        """Publishes a message to the control channel to ask WSManager to resubscribe."""
        payload = {
            "command": "RESUBSCRIBE_L2",
            "symbol": symbol
        }
        await self.redis.publish(CONTROL_CHANNEL, json.dumps(payload))
        
        # Reset book state to wait for the new snapshot
        self.symbol_ready_state[symbol]["book"] = False
        self.sequence_numbers[symbol] = -1
        self.order_books[symbol]["is_awaiting_snapshot"] = True
        log.warning(f"⚠️ Published RESUBSCRIBE_L2 request for {symbol}. Invalidating book state.")


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
                    
                    if res in ["1d", "1w"]:
                        two_years_in_seconds = 3600 * 24 * 365 * 2
                        start_time = end_time - two_years_in_seconds
                        limit = 2000 
                    else:
                        start_time = end_time - (limit * duration * 1.5) 
                        limit = min(2000, limit)
                    
                    path = "/v2/history/candles" 
                    params = {
                        "symbol": symbol,
                        "resolution": res,
                        "start": str(start_time), 
                        "end": str(end_time),
                        "limit": str(limit) 
                    }
                    url = f"{DELTA_BASE_URL}{path}"
                    
                    log.debug(f"Fetching history: {symbol} {res} (Limit: {limit})...")
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
        # self._is_priming = False <-- FIX: Removed, lock handles state

    # ----------------------------------------------------------------------
    # WebSocket Message Handlers with Checksum/Sequence Validation
    # ----------------------------------------------------------------------
    
    def _validate_checksum(self, symbol: str, received_cs: int) -> bool:
        """
        Validates the order book checksum based on top 10 price levels (per Delta API docs).
        """
        book = self.order_books.get(symbol)
        if not book: return False

        def build_string(side_data, key_source, ascending):
            prices = sorted(key_source, key=float, reverse=not ascending)
            parts = []
            for price in prices[:10]:
                size = book[side_data][price]
                size_str = str(int(size)) if float(size).is_integer() else str(size)
                parts.append(f"{price}:{size_str}")
            return ",".join(parts)
        
        asks_keys = book["asks"].keys()
        bids_keys = book["bids"].keys()

        asks_str = build_string("asks", asks_keys, ascending=True)
        bids_str = build_string("bids", bids_keys, ascending=False)
        checksum_string = f"{asks_str}|{bids_str}"

        try:
            calculated_cs = zlib.crc32(checksum_string.encode('utf-8')) & 0xFFFFFFFF
            received_cs_unsigned = received_cs & 0xFFFFFFFF
        except AttributeError:
             log.warning("zlib is required for CRC32 checksum, skipping validation.")
             return True 

        if calculated_cs != received_cs_unsigned:
            log.error(f"❌ CHECKSUM MISMATCH for {symbol}! Recv: {received_cs}, Calc: {calculated_cs}. Resubscribe needed.")
            return False
        
        return True


    def _handle_l2_snapshot(self, data: dict):
        symbol = data.get("symbol")
        if not symbol: return
        self._initialize_state(symbol)
        
        # 1. Store initial sequence number
        self.sequence_numbers[symbol] = data.get("sequence_no", -1)
        
        # 2. Rebuild the book
        self.order_books[symbol]["bids"].clear()
        self.order_books[symbol]["asks"].clear()
        for price_str, size_str in data.get("bids", []):
            self.order_books[symbol]["bids"][price_str] = float(size_str)
        for price_str, size_str in data.get("asks", []):
            self.order_books[symbol]["asks"][price_str] = float(size_str)

        self.order_books[symbol]["is_awaiting_snapshot"] = False
        if symbol in self.symbol_ready_state:
            self.symbol_ready_state[symbol]["book"] = True
        log.info(f"✅ L2 Snapshot processed for {symbol} (Seq: {self.sequence_numbers[symbol]}). Resuming normal updates.")
    
    
    def _handle_l2_update(self, data: dict):
        symbol = data.get("symbol")
        if symbol not in self.order_books or symbol not in self.sequence_numbers: return
        
        new_seq_no = data.get("sequence_no", -1)
        received_cs = data.get("cs", -1)
        
        if self.sequence_numbers[symbol] == -1:
            if self.order_books[symbol]["is_awaiting_snapshot"]:
                log.warning(f"⚠️ Skipping L2 update (Seq {new_seq_no}) for {symbol}. Waiting for snapshot to establish sequence.")
                self.order_books[symbol]["is_awaiting_snapshot"] = False
            return 

        # 1. Check sequence number continuity 
        expected_seq_no = self.sequence_numbers[symbol] + 1
        if new_seq_no != expected_seq_no:
            log.error(f"❌ SEQUENCE MISMATCH for {symbol}! Expected {expected_seq_no}, Got {new_seq_no}. Requires resubscribe.")
            asyncio.create_task(self._publish_resubscribe_request(symbol)) 
            return 
            
        self.sequence_numbers[symbol] = new_seq_no

        # 2. Apply updates
        for price_str, size_str in data.get("bids", []):
            size = float(size_str)
            if size == 0: self.order_books[symbol]["bids"].pop(price_str, None)
            else: self.order_books[symbol]["bids"][price_str] = size
        for price_str, size_str in data.get("asks", []):
            size = float(size_str)
            if size == 0: self.order_books[symbol]["asks"].pop(price_str, None)
            else: self.order_books[symbol]["asks"][price_str] = size

        # 3. Validate Checksum
        if received_cs != -1:
            if not self._validate_checksum(symbol, received_cs):
                 asyncio.create_task(self._publish_resubscribe_request(symbol))
                 return

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

    
    def _handle_spot_price(self, data: dict):
        """Handle v2/spot_price message and store the price."""
        full_symbol = data.get("s")
        price = data.get("p")
        if not full_symbol or price is None: return

        # Map index symbol back to trading symbol (e.g., .DEXBTUSD -> BTCUSD)
        trading_symbol = next((k for k, v in SPOT_INDEX_SYMBOLS.items() if v == full_symbol), None)
        
        if not trading_symbol: return

        if trading_symbol not in self.features: self._initialize_state(trading_symbol)
        
        self.features[trading_symbol]["spot_price"] = float(price)


    # ----------------------------------------------------------------------
    # FEATURE CALCULATION
    # ----------------------------------------------------------------------

    def _calc_imbalance_and_mid(self, symbol: str):
        """Calculates Order Book Imbalance (OBI) and Mid Price."""
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
        """Calculates Trade Flow Index (TFI) over a lookback window."""
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

                # --- ATR Calculation ---
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
                
                # --- Weekly S/R Calculation ---
                if len(candle_deque) > 1:
                    prev_candle = candle_deque[-2]
                    if timeframe == "1w":
                        self.features[symbol]["PWH"] = prev_candle["high"]
                        self.features[symbol]["PWL"] = prev_candle["low"]

            except Exception as e:
                log.error(f"Error calculating TA for {symbol} {timeframe}: {e}", exc_info=True)
        
        self.features[symbol]["tas"] = tas


    async def _publish_features(self, symbol: str, timestamp_us: int):
        """Calculates all features, caches the payload, and publishes to the channel."""
        if symbol not in self.features:
            self._initialize_state(symbol)

        obi, mid_price = self._calc_imbalance_and_mid(symbol)
        if mid_price is not None:
            self.features[symbol]["obi"] = obi
            self.features[symbol]["mid_price"] = mid_price
        tfi = self._calculate_tfi(symbol)
        self.features[symbol]["tfi"] = tfi
        
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
                "spot_price": self.features[symbol].get("spot_price"), 
                "tas": self.features[symbol]["tas"],
                
                "PWH": self.features[symbol].get("PWH"),
                "PWL": self.features[symbol].get("PWL"),
                "PMH": self.features[symbol].get("PMH"),
                "PML": self.features[symbol].get("PML"),
            }
            
            # ✅ NEW STEP: Cache the complete enriched payload for other services (like TSL)
            try:
                # Cache using the key prefix and symbol, set expiry
                await self.redis.set(f"{LATEST_ENRICHED_KEY}{symbol}", json.dumps(payload), ex=300) 
            except Exception as e:
                log.error(f"❌ Failed to cache enriched event to Redis: {e}")
            
            await self._publish(payload)
            
    # --- FIX: New method to process buffered and live messages ---
    async def _process_message(self, raw: dict):
        """Wrapper for the main message processing logic."""
        msg_type = raw.get("type")
        timestamp = raw.get("timestamp", 0)
        symbol = None 
        
        try:
            # Logic to extract symbol for non-spot messages
            if msg_type == "mark_price":
                raw_sym = raw.get("symbol")
                if raw_sym: symbol = raw_sym.split(":", 1)[-1] 
            elif msg_type == "v2/spot_price":
                self._handle_spot_price(raw)
                return # This message type doesn't need publishing
            else:
                symbol = raw.get("symbol") 
            
            if not symbol or symbol not in TRADING_SYMBOLS:
                return 
                
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
                # FIX: Do not process candle messages if priming
                if not self._priming_lock.locked():
                    self._handle_candlestick(raw)
                # If priming, candle messages are just ignored, which is fine
                # as the historical pull will be more accurate.
                
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

    # --- FIX: New method to listen to Redis immediately ---
    async def _message_listener(self):
        """
        Subscribes to Redis and processes messages.
        If priming, it buffers messages to be processed later.
        """
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("delta:raw:ws")
        log.info("✅ FeatureEngine subscribed to delta:raw:ws and waiting for messages...")

        try:
            async for msg in pubsub.listen():
                if self._stop_flag:
                    break
                if msg.get("type") != "message": 
                    continue

                try: 
                    raw = json.loads(msg.get("data"))
                except Exception as e: 
                    log.warning(f"⚠️ JSON parse failed: {e}"); continue
                
                # --- FIX: Check if the priming lock is held ---
                if self._priming_lock.locked():
                    # Buffer messages if priming or buffer processing is in progress
                    self._message_buffer.append(raw)
                else:
                    # Process live messages directly
                    await self._process_message(raw)
                # --- END FIX ---

        except asyncio.CancelledError:
            log.info("FeatureEngine listener task cancelled.")
        except Exception as e:
            log.error(f"💥 FeatureEngine listener crashed: {e}", exc_info=True)
        finally:
            await pubsub.unsubscribe("delta:raw:ws")

    # --- FIX: Updated 'start' method to run tasks concurrently ---
    async def start(self):
        """
        Primes historical data while concurrently listening for live messages.
        """
        self._stop_flag = False
        self._message_buffer.clear()
        
        listener_task = asyncio.create_task(self._message_listener())
        
        try:
            # Run priming
            # --- FIX: Use the lock to manage the critical section ---
            async with self._priming_lock:
                log.info("🔒 Acquired priming lock. Starting history fetch...")
                await self._prime_candle_history()
                log.info(f"✅ Priming complete. Processing {len(self._message_buffer)} buffered messages...")
                
                # Process all buffered messages
                while self._message_buffer:
                    raw_msg = self._message_buffer.popleft()
                    await self._process_message(raw_msg)
            
            log.info("✅ Message buffer processed. 🔓 Releasing lock. Now listening for live data.")
            # --- END FIX ---
            
            # Now, the listener_task will handle messages live.
            # We just await it to keep the service running.
            await listener_task
            
        except asyncio.CancelledError:
            log.info("FeatureEngine main task cancelled.")
        except Exception as e:
            log.error(f"💥 FeatureEngine main task crashed: {e}", exc_info=True)
        finally:
            self._stop_flag = True
            if listener_task and not listener_task.done():
                listener_task.cancel()
            log.info("🔻 FeatureEngine stopped cleanly.")