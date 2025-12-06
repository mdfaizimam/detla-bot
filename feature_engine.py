# --- detla-bot/feature_engine.py ---
# ✅ FIX: Restored missing Checksum Logic
# ✅ UPGRADE: Calculates PMH/PML/Pivots for SNR Strategy

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
import redis.exceptions 

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
    LATEST_ENRICHED_KEY,
    HEALTH_CHECK_KEY_FE 
)
from utils.binance_client import get_latest_ls_ratio

log = logging.getLogger("feature_engine")

# --- Constants for Feature Calculation ---
TFI_LOOKBACK_SECONDS = 5
CANDLE_HISTORY_SIZE = 100 

CANDLE_RESOLUTIONS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
RESOLUTION_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, 
    "4h": 14400, "1d": 86400, "1w": 604800,
}

class FeatureEngine:
    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession, top_n=5):
        self.redis = redis_client 
        self.session = http_session 
        self.top_n = top_n
        
        self.order_books = {} 
        self.trade_logs = {}  
        self.features = {} 
        self.candle_history = {} 
        self.sequence_numbers = {} 
        self.symbol_ready_state = {}
        
        # Binance Data Cache
        self.binance_cache = {}
        self._binance_poll_task = None
        
        self.candle_regex = re.compile(r"candlestick_(\w+)")
        
        self._stop_flag = False
        self._message_buffer = deque()
        self._priming_lock = asyncio.Lock()
        self.last_processed_timestamp = 0

    async def _poll_external_data(self):
        """Polls Binance for L/S Ratio every 60s."""
        log.info("🌍 Starting Binance Data Poller...")
        while not self._stop_flag:
            try:
                for symbol in TRADING_SYMBOLS:
                    lsr = await get_latest_ls_ratio(self.session, symbol)
                    if lsr: 
                        self.binance_cache[f"{symbol}_lsr"] = lsr
                await asyncio.sleep(60)
            except Exception as e:
                log.error(f"External poll error: {e}")
                await asyncio.sleep(10)

    async def _publish(self, payload: dict):
        try:
            await self.redis.publish(ENRICHED_CHANNEL, json.dumps(payload))
        except Exception as e:
            log.error(f"❌ Failed to publish enriched event: {e}")

    def _initialize_state(self, symbol):
        if symbol not in self.order_books:
            self.order_books[symbol] = {
                "bids": {}, "asks": {}, "is_awaiting_snapshot": True 
            }
        
        if symbol not in self.trade_logs:
            self.trade_logs[symbol] = deque()

        if symbol not in self.features:
            self.features[symbol] = {
                "obi": 0.0, "mid_price": None, "tfi": 0.0,
                "tfi_state": {"buy_vol": 0.0, "sell_vol": 0.0},
                "last_trade_price": None, "mark_price": None,
                "funding_rate": None, "spot_price": None, 
                "tas": {}, "timestamp": 0,
                "PWH": None, "PWL": None, "PMH": None, "PML": None, 
                "long_short_ratio": 1.0 
            }
            
        if symbol not in self.candle_history:
            self.candle_history[symbol] = {} 
            for res in CANDLE_RESOLUTIONS:
                self.candle_history[symbol][res] = deque(maxlen=CANDLE_HISTORY_SIZE)
        
        if symbol not in self.symbol_ready_state:
            self.symbol_ready_state[symbol] = {"book": False, "mark": False, "funding": False}
            
        if symbol not in self.sequence_numbers:
            self.sequence_numbers[symbol] = -1 

    def _is_symbol_ready(self, symbol: str) -> bool:
        if self._priming_lock.locked(): return False
        if symbol not in self.symbol_ready_state: return False
        state = self.symbol_ready_state[symbol]
        if state.get("full", False): return True
        if state["book"] and state["mark"] and state["funding"]:
            log.info(f"✅ {symbol} is now data-ready. Publishing enriched feed.")
            state["full"] = True 
            return True
        return False
        
    async def _publish_resubscribe_request(self, symbol: str):
        payload = { "command": "RESUBSCRIBE_L2", "symbol": symbol }
        await self.redis.publish(CONTROL_CHANNEL, json.dumps(payload))
        self.symbol_ready_state[symbol]["book"] = False
        self.sequence_numbers[symbol] = -1
        self.order_books[symbol]["is_awaiting_snapshot"] = True
        log.warning(f"⚠️ Published RESUBSCRIBE_L2 request for {symbol}. Invalidating book state.")

    async def _prime_candle_history(self):
        log.info("Priming candle history for all symbols...")
        end_time = int(time.time())
        
        for symbol in TRADING_SYMBOLS:
            self._initialize_state(symbol) 
            for res in CANDLE_RESOLUTIONS:
                try:
                    duration = RESOLUTION_SECONDS[res]
                    limit = CANDLE_HISTORY_SIZE + 50 
                    if res in ["1d", "1w"]:
                        start_time = end_time - (3600 * 24 * 365 * 2)
                        limit = 2000 
                    else:
                        start_time = end_time - (limit * duration * 1.5) 
                        limit = min(2000, limit)
                    
                    path = "/v2/history/candles" 
                    params = {
                        "symbol": symbol, "resolution": res,
                        "start": str(start_time), "end": str(end_time), "limit": str(limit) 
                    }
                    url = f"{DELTA_BASE_URL}{path}"
                    
                    async with self.session.get(url, params=params, headers={'User-Agent': USER_AGENT}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            candles = data.get("result", [])
                            for candle in candles:
                                candle_data = {
                                    "open": float(candle.get("open", 0)),
                                    "high": float(candle.get("high", 0)),
                                    "low": float(candle.get("low", 0)),
                                    "close": float(candle.get("close", 0)),
                                    "volume": float(candle.get("volume", 0)),
                                    "timestamp": candle.get("time", 0) * 1_000_000 
                                }
                                self.candle_history[symbol][res].append(candle_data)
                            log.info(f"✅ Primed {len(candles)} candles for {symbol} {res}")
                    await asyncio.sleep(0.3) 
                except Exception as e:
                    log.error(f"Error priming {symbol} {res}: {e}")
        
        for symbol in TRADING_SYMBOLS:
            self._calculate_technical_indicators(symbol)

    def _validate_checksum(self, symbol: str, received_cs: int) -> bool:
        """Validates the CRC32 checksum of the order book against Delta's."""
        book = self.order_books.get(symbol)
        if not book: return False
        
        # Helper to build the string according to Delta's format
        def build_string(side_data, key_source, ascending):
            prices = sorted(key_source, key=float, reverse=not ascending)
            parts = []
            # Take top 10 levels
            for price in prices[:10]:
                size = book[side_data][price]
                # Format size: integers without decimal, floats with
                size_str = str(int(size)) if float(size).is_integer() else str(size)
                parts.append(f"{price}:{size_str}")
            return ",".join(parts)

        asks_keys = book["asks"].keys()
        bids_keys = book["bids"].keys()
        
        asks_str = build_string("asks", asks_keys, ascending=True)
        bids_str = build_string("bids", bids_keys, ascending=False)
        
        checksum_string = f"{asks_str}|{bids_str}"
        
        try:
            # Calculate CRC32
            calculated_cs = zlib.crc32(checksum_string.encode('utf-8')) & 0xFFFFFFFF
            received_cs_unsigned = received_cs & 0xFFFFFFFF
        except AttributeError: 
            return True 
            
        if calculated_cs != received_cs_unsigned:
            log.error(f"❌ CHECKSUM MISMATCH for {symbol}! Calc: {calculated_cs}, Recv: {received_cs_unsigned}")
            return False
            
        return True

    def _handle_l2_snapshot(self, data: dict):
        symbol = data.get("symbol")
        if not symbol: return
        self._initialize_state(symbol)
        self.sequence_numbers[symbol] = data.get("sequence_no", -1)
        self.order_books[symbol]["bids"].clear()
        self.order_books[symbol]["asks"].clear()
        for price_str, size_str in data.get("bids", []):
            if size_str is None: continue
            self.order_books[symbol]["bids"][price_str] = float(size_str)
        for price_str, size_str in data.get("asks", []):
            if size_str is None: continue
            self.order_books[symbol]["asks"][price_str] = float(size_str)
        self.order_books[symbol]["is_awaiting_snapshot"] = False
        if symbol in self.symbol_ready_state:
            self.symbol_ready_state[symbol]["book"] = True
        log.info(f"✅ L2 Snapshot processed for {symbol} (Seq: {self.sequence_numbers[symbol]}).")
    
    def _handle_l2_update(self, data: dict):
        symbol = data.get("symbol")
        if symbol not in self.order_books or symbol not in self.sequence_numbers: return
        new_seq_no = data.get("sequence_no", -1)
        received_cs = data.get("cs", -1)
        
        if self.sequence_numbers[symbol] == -1: return 
        expected_seq_no = self.sequence_numbers[symbol] + 1
        
        if new_seq_no != expected_seq_no:
            log.error(f"❌ SEQUENCE MISMATCH for {symbol}! Expected {expected_seq_no}, Got {new_seq_no}.")
            asyncio.create_task(self._publish_resubscribe_request(symbol)) 
            return 
        self.sequence_numbers[symbol] = new_seq_no

        for price_str, size_str in data.get("bids", []):
            if size_str is None: 
                self.order_books[symbol]["bids"].pop(price_str, None)
                continue
            size = float(size_str)
            if size == 0: self.order_books[symbol]["bids"].pop(price_str, None)
            else: self.order_books[symbol]["bids"][price_str] = size
            
        for price_str, size_str in data.get("asks", []):
            if size_str is None: 
                self.order_books[symbol]["asks"].pop(price_str, None)
                continue
            size = float(size_str)
            if size == 0: self.order_books[symbol]["asks"].pop(price_str, None)
            else: self.order_books[symbol]["asks"][price_str] = size

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
        log.info(f"✅ Trade snapshot processed for {symbol}")

    def _handle_all_trades(self, trade: dict):
        symbol = trade.get("symbol")
        if not symbol: return
        try:
            side = None
            if trade.get("buyer_role") == "taker": side = "buy"
            elif trade.get("seller_role") == "taker": side = "sell"
            if side:
                ts = trade.get("timestamp", 0) / 1_000_000.0 
                size = float(trade.get("size", 0))
                price = float(trade.get("price", 0))
                self.trade_logs[symbol].append((ts, side, size))
                tfi_state = self.features[symbol]["tfi_state"]
                if side == "buy": tfi_state["buy_vol"] += size
                elif side == "sell": tfi_state["sell_vol"] += size
                self.features[symbol]["last_trade_price"] = price
                self._prune_trade_log(symbol, time.time())
        except Exception: pass

    def _prune_trade_log(self, symbol: str, current_time_sec: float):
        if symbol not in self.trade_logs: return
        cutoff_time = current_time_sec - TFI_LOOKBACK_SECONDS
        tfi_state = self.features[symbol]["tfi_state"]
        while self.trade_logs[symbol]:
            if self.trade_logs[symbol][0][0] < cutoff_time:
                (ts, side, size) = self.trade_logs[symbol].popleft()
                if side == "buy": tfi_state["buy_vol"] = max(0, tfi_state["buy_vol"] - size)
                elif side == "sell": tfi_state["sell_vol"] = max(0, tfi_state["sell_vol"] - size)
            else:
                break

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
        elif history_deque and history_deque[-1]["timestamp"] == candle_data["timestamp"]:
            history_deque[-1] = candle_data 
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
        full_symbol = data.get("s")
        price = data.get("p")
        if not full_symbol or price is None: return
        trading_symbol = next((k for k, v in SPOT_INDEX_SYMBOLS.items() if v == full_symbol), None)
        if not trading_symbol: return
        if trading_symbol not in self.features: self._initialize_state(trading_symbol)
        self.features[trading_symbol]["spot_price"] = float(price)

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
        except Exception: return 0.0, None
            
    def _calculate_tfi(self, symbol: str):
        if symbol not in self.features: return 0.0
        try:
            state = self.features[symbol]["tfi_state"]
            buy_vol = state["buy_vol"]
            sell_vol = state["sell_vol"]
            denom = buy_vol + sell_vol
            return (buy_vol - sell_vol) / denom if denom else 0.0
        except Exception: return 0.0

    def _calculate_technical_indicators(self, symbol: str):
        if symbol not in self.candle_history: return 
        tas = {} 
        for timeframe, candle_deque in self.candle_history[symbol].items():
            min_candles = max(50, VOLUME_SMA_PERIOD + 2) 
            if len(candle_deque) < min_candles: continue
            try:
                df = pd.DataFrame(list(candle_deque))
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
                df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
                df.sort_index(inplace=True) 
                
                df.ta.ema(length=20, append=True)
                df.ta.ema(length=50, append=True)
                df.ta.rsi(length=14, append=True)
                df.ta.macd(fast=12, slow=26, signal=9, append=True)
                df.ta.obv(append=True)
                df.ta.adx(length=14, append=True)
                
                er_period = 10
                change = df['close'].diff(er_period).abs()
                volatility = df['close'].diff().abs().rolling(er_period).sum()
                df['KER'] = change / (volatility + 1e-9)
                
                df.ta.atr(length=14, append=True)
                df['FRACTAL_DIM'] = df['ATRr_14'] / (df['close'].rolling(20).std() + 1e-9)
                df.ta.bbands(length=20, std=2, append=True)
                
                latest_tas = {
                    "ema_20": df['EMA_20'].iloc[-1],
                    "ema_50": df['EMA_50'].iloc[-1],
                    "rsi_14": df['RSI_14'].iloc[-1],
                    "macd_hist": df['MACDh_12_26_9'].iloc[-1],
                    "close": df['close'].iloc[-1],
                    "open": df['open'].iloc[-1],
                    "obv": df['OBV'].iloc[-1],
                    "adx": df['ADX_14'].iloc[-1],
                    "ker": df['KER'].iloc[-1] if 'KER' in df.columns else 0.5,
                    "fractal_dim": df['FRACTAL_DIM'].iloc[-1] if 'FRACTAL_DIM' in df.columns else 1.0,
                    "bb_lower": df['BBL_20_2.0'].iloc[-1],
                    "bb_upper": df['BBU_20_2.0'].iloc[-1],
                    "bb_mid": df['BBM_20_2.0'].iloc[-1],
                    "bb_width": df['BBB_20_2.0'].iloc[-1] if 'BBB_20_2.0' in df.columns else 0.0,
                    "atr": df['ATRr_14'].iloc[-1] if 'ATRr_14' in df.columns else 0.0
                }
                
                if timeframe == VOLUME_TIMEFRAME:
                    vol_sma_name = f"SMA_volume_{VOLUME_SMA_PERIOD}"
                    df.ta.sma(close='volume', length=VOLUME_SMA_PERIOD, append=True, col_names=(vol_sma_name,))
                    if vol_sma_name in df.columns:
                        latest_tas["volume"] = df['volume'].iloc[-1]
                        latest_tas[vol_sma_name] = df[vol_sma_name].iloc[-1]
                
                # ✅ NEW: Calculate Daily Pivots & Highs/Lows
                if timeframe == "1d":
                    if len(df) > 1:
                        prev = df.iloc[-2]
                        self.features[symbol]["PMH"] = float(prev['high'])
                        self.features[symbol]["PML"] = float(prev['low'])
                        
                        P = (prev['high'] + prev['low'] + prev['close']) / 3
                        latest_tas["pivot"] = P
                        latest_tas["R1"] = (2 * P) - prev['low']
                        latest_tas["S1"] = (2 * P) - prev['high']
                        latest_tas["R2"] = P + (prev['high'] - prev['low'])
                        latest_tas["S2"] = P - (prev['high'] - prev['low'])
                        
                tas[timeframe] = latest_tas
                
            except Exception as e:
                log.error(f"Error calculating TA for {symbol} {timeframe}: {e}")
        self.features[symbol]["tas"] = tas

    async def _publish_features(self, symbol: str, timestamp_us: int):
        if symbol not in self.features: self._initialize_state(symbol)
        obi, mid_price = self._calc_imbalance_and_mid(symbol)
        if mid_price is not None:
            self.features[symbol]["obi"] = obi
            self.features[symbol]["mid_price"] = mid_price
        self.features[symbol]["tfi"] = self._calculate_tfi(symbol)
        self.features[symbol]["timestamp"] = timestamp_us

        lsr_key = f"{symbol}_lsr"
        current_lsr = self.binance_cache.get(lsr_key, 1.0)
        self.features[symbol]["long_short_ratio"] = current_lsr

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
                "long_short_ratio": self.features[symbol]["long_short_ratio"],
                "spot_price": self.features[symbol].get("spot_price"), 
                "tas": self.features[symbol]["tas"],
                # Pass SNR levels
                "PMH": self.features[symbol].get("PMH"),
                "PML": self.features[symbol].get("PML"),
            }
            try:
                await self.redis.set(f"{LATEST_ENRICHED_KEY}{symbol}", json.dumps(payload), ex=300) 
                await self.redis.set(HEALTH_CHECK_KEY_FE, timestamp_us, ex=300)
            except Exception: pass
            await self._publish(payload)
            
    async def _process_message(self, raw: dict):
        if raw.get("type") == "synthetic_heartbeat":
            return

        self.last_processed_timestamp = raw.get("timestamp", 0)
        msg_type = raw.get("type")
        timestamp = raw.get("timestamp", 0)
        symbol = None 
        try:
            if msg_type == "mark_price":
                raw_sym = raw.get("symbol")
                if raw_sym: symbol = raw_sym.split(":", 1)[-1] 
            elif msg_type == "v2/spot_price":
                self._handle_spot_price(raw)
                return
            else:
                symbol = raw.get("symbol") 
            
            if not symbol or symbol not in TRADING_SYMBOLS: return 
            should_publish = False

            if msg_type == "l2_updates":
                action = raw.get("action")
                if action == "snapshot": self._handle_l2_snapshot(raw)
                elif action == "update": self._handle_l2_update(raw)
                should_publish = True
            elif msg_type == "all_trades_snapshot": self._handle_all_trades_snapshot(raw)
            elif msg_type == "all_trades":
                self._handle_all_trades(raw)
                should_publish = True
            elif msg_type.startswith("candlestick_"):
                if not self._priming_lock.locked(): self._handle_candlestick(raw)
            elif msg_type == "funding_rate":
                self._handle_funding_rate(raw)
                should_publish = True
            elif msg_type == "mark_price":
                self._handle_mark_price(raw, symbol) 
                should_publish = True
            
            if should_publish and self._is_symbol_ready(symbol):
                await self._publish_features(symbol, timestamp)
        
        except Exception as e:
            log.error(f"Error processing message type {msg_type} for {symbol}: {e}")

    async def _message_listener(self):
        while not self._stop_flag:
            pubsub = None
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe("delta:raw:ws")
                log.info("✅ FeatureEngine subscribed to delta:raw:ws...")

                async for msg in pubsub.listen():
                    if self._stop_flag: break
                    if msg.get("type") != "message": continue
                    try: 
                        raw = json.loads(msg.get("data"))
                        if self._priming_lock.locked(): self._message_buffer.append(raw)
                        else: await self._process_message(raw)
                    except Exception: continue
                    
            except (redis.exceptions.ConnectionError, ConnectionResetError, OSError) as e:
                log.warning(f"⚠️ Redis PubSub connection lost: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                log.info("FeatureEngine listener task cancelled.")
                break
            except Exception as e:
                log.error(f"FeatureEngine listener unexpected error: {e}")
                await asyncio.sleep(5)
            finally:
                if pubsub: await pubsub.close()

    async def start(self):
        self._stop_flag = False
        self._message_buffer.clear()
        listener_task = asyncio.create_task(self._message_listener())
        self._binance_poll_task = asyncio.create_task(self._poll_external_data()) 
        
        try:
            async with self._priming_lock:
                log.info("🔒 Acquired priming lock. Starting history fetch...")
                await self._prime_candle_history()
                log.info(f"✅ Priming complete. Processing {len(self._message_buffer)} buffered messages...")
                while self._message_buffer:
                    raw_msg = self._message_buffer.popleft()
                    await self._process_message(raw_msg)
            log.info("✅ Message buffer processed. 🔓 Releasing lock. Now listening for live data.")
            await listener_task
        except asyncio.CancelledError: log.info("FeatureEngine main task cancelled.")
        finally:
            self._stop_flag = True
            if listener_task and not listener_task.done(): listener_task.cancel()
            if self._binance_poll_task: self._binance_poll_task.cancel()
            log.info("🔻 FeatureEngine stopped cleanly.")