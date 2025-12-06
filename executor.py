# --- detla-bot/executor.py ---
# ⚡ EXECUTOR FIX: Handles Dict Config & Forces Integers
# ✅ FIX: Reads specific size per symbol from config
# ✅ FIX: Casts size to int() to satisfy API validation

import aiohttp
import asyncio
import json
import logging
import time
import urllib.parse
from typing import Optional, Any, Dict, Tuple, List

from redis import asyncio as aioredis

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    SIGNAL_CHANNEL,
    MONITORING_CHANNEL,
    USER_AGENT,
    TRADING_SYMBOLS,
    DMS_ID,
    TSL_CHANNEL,
    TSL_ENABLED,
    config,
    BRACKET_STOP_TRIGGER,
    BRACKET_ORDER_TYPE,
    REDIS_POSITION_LOCK_PREFIX
)
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager

logger = logging.getLogger("executor")

class OrderExecutionManager:
    REDIS_POSITION_LOCK_PREFIX = REDIS_POSITION_LOCK_PREFIX
    REDIS_POSITION_LOCK_TTL = 60

    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.redis = redis_client
        self.api_client = api_client
        self.session = api_client.session
        self._process_lock = asyncio.Lock()
        self.risk_manager = risk_manager
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.dms_id = DMS_ID
        self.product_info_cache: Dict[str, Dict[str, Any]] = {}
        self._signal_task: Optional[asyncio.Task] = None
        logger.info("✅ OrderExecutionManager initialized.")

    async def start(self):
        logger.info("▶️ OrderExecutionManager starting (listening for signals)...")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(SIGNAL_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    signal = json.loads(msg["data"])
                except Exception:
                    continue
                asyncio.create_task(self._handle_signal(signal))
        except asyncio.CancelledError:
            logger.info("OrderExecutionManager cancelled.")
        finally:
            await pubsub.unsubscribe(SIGNAL_CHANNEL)

    async def _get_product_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol in self.product_info_cache:
            return self.product_info_cache[symbol]
        path = f"/v2/products/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"
        try:
            async with self.session.get(url, headers={'User-Agent': USER_AGENT}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    product = data.get("result", {})
                    if product:
                        product_id = product.get("id")
                        tick_size_str = product.get("tick_size")
                        if product_id and tick_size_str:
                            try:
                                precision = len(str(tick_size_str).split(".")[-1]) if "." in str(tick_size_str) else 0
                                info = {
                                    "id": int(product_id),
                                    "tick_size": float(tick_size_str),
                                    "precision": int(precision),
                                }
                                if product.get("symbol"):
                                    info["symbol"] = product.get("symbol")
                                self.product_info_cache[symbol] = info
                                return info
                            except Exception:
                                return None
                return None
        except Exception as e:
            logger.error("❌ Error fetching product info: %s", e)
            return None

    async def _get_ticker_prices(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        path = f"/v2/tickers/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"
        try:
            async with self.session.get(url, headers={'User-Agent': USER_AGENT}) as resp:
                data = await resp.json()
                if resp.status == 200 and data.get("success"):
                    res = data.get("result", {})
                    mark_price = res.get("mark_price")
                    last_price = res.get("close")
                    return (
                        float(mark_price) if mark_price is not None else None,
                        float(last_price) if last_price is not None else None,
                    )
                return (None, None)
        except Exception:
            return (None, None)

    async def _acquire_position_lock(self, symbol: str, timeout: int = 5) -> bool:
        deadline = time.time() + timeout
        lock_key = f"{self.REDIS_POSITION_LOCK_PREFIX}{symbol}"
        lock_value = json.dumps({"symbol": symbol, "ts": time.time()})
        while time.time() < deadline:
            ok = await self.redis.set(lock_key, lock_value, ex=self.REDIS_POSITION_LOCK_TTL, nx=True)
            if ok: return True
            await asyncio.sleep(0.25)
        logger.warning("⚠️ Could not acquire lock for %s (Busy)", symbol)
        return False

    async def _release_position_lock(self, symbol: str):
        try:
            lock_key = f"{self.REDIS_POSITION_LOCK_PREFIX}{symbol}"
            await self.redis.delete(lock_key)
        except Exception: pass

    async def _handle_signal(self, signal: dict):
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        
        # 🔧 FIX: Handle Dictionary or Float for size
        base_size_config = config["BASE_POSITION_SIZE"]
        if isinstance(base_size_config, dict):
            # Get specific size for symbol, or default to 1
            size_hint = base_size_config.get(symbol, 1)
        else:
            # Fallback for legacy config
            size_hint = float(base_size_config)

        if not symbol or not direction:
            logger.warning("Ignoring invalid signal: %s", signal)
            return

        ok, info = await self.risk_manager.validate_signal(signal)
        if not ok:
            logger.warning("Signal rejected by RiskManager: %s", info)
            return

        side = "buy" if direction == "LONG" else "sell"

        if not await self._acquire_position_lock(symbol):
            logger.warning("Position already active for %s; skipping signal.", symbol)
            return

        try:
            product_info = await self._get_product_info(symbol)
            if not product_info:
                logger.error("Missing product info for %s", symbol)
                await self._release_position_lock(symbol)
                return

            precision = int(product_info["precision"])
            product_id = int(product_info["id"])
            mark_price, last_price = await self._get_ticker_prices(symbol)
            anchor_price = mark_price if BRACKET_STOP_TRIGGER == "mark_price" else last_price

            sl_price = signal.get("sl_price")
            tp_price = signal.get("tp_price")
            
            if sl_price is None or tp_price is None:
                if anchor_price is None:
                    await self._release_position_lock(symbol)
                    return
                entry_reference = float(anchor_price)
                sl_pct = 0.02
                tp_pct = 0.03
                if direction == "LONG":
                    sl_price = entry_reference * (1.0 - sl_pct)
                    tp_price = entry_reference * (1.0 + tp_pct)
                else:
                    sl_price = entry_reference * (1.0 + sl_pct)
                    tp_price = entry_reference * (1.0 - tp_pct)
                sl_price = round(sl_price, precision)
                tp_price = round(tp_price, precision)

            # 4. Execute Trade (Force Integer Size)
            # CRITICAL FIX: int() ensures we send 1, not 1.0
            int_size = int(size_hint)

            res = await self._place_linked_orders(
                symbol, side, int_size, float(tp_price), float(sl_price)
            )
            
            if not res:
                logger.error("Order placement failed for %s. Releasing lock.", symbol)
                await self._release_position_lock(symbol)
                return

            product_id, ret_direction = res
            entry_price = float(anchor_price or 0.0)

            await self._notify_monitor(symbol, int_size, product_id)
            if TSL_ENABLED:
                await self._notify_tsl_manager(symbol, ret_direction, float(int_size), product_id, entry_price)

            logger.info("✅ Signal executed for %s (size=%d). Lock is HELD.", symbol, int_size)

        except Exception as e:
            logger.error("❌ Error handling signal: %s. Releasing lock.", e, exc_info=True)
            await self._release_position_lock(symbol)

    async def _send_order(self, method: str, path: str, payload: dict) -> dict:
        if method.upper() == "POST":
            status, data = await self.api_client.post(path, payload)
        else:
            status, data = await self.api_client.get(path, params=payload)
        return data if status == 200 else {"success": False, "error": data}

    async def _place_linked_orders(self, symbol, side, size, tp_price, sl_price) -> Optional[Tuple[int, str]]:
        product_info = await self._get_product_info(symbol)
        if not product_info: return None
        product_id = int(product_info["id"])
        precision = int(product_info["precision"])
        
        entry_payload = {
            "product_id": product_id,
            "size": size, # int
            "side": side,
            "order_type": "market_order",
        }
        logger.info(f"📦 Placing Market Entry Order: {entry_payload}")
        entry_resp = await self._send_order("POST", "/v2/orders", entry_payload)
        
        if not entry_resp.get("success"):
            logger.error(f"❌ Market Entry Order failed: {entry_resp}")
            return None
        
        # Bracket logic
        final_sl_price = round(float(sl_price), precision)
        final_tp_price = round(float(tp_price), precision)
        sl_str = f"{final_sl_price:.{precision}f}"
        tp_str = f"{final_tp_price:.{precision}f}"

        bracket_payload = {
            "product_id": product_id,
            "product_symbol": symbol,
            "stop_loss_order": {
                "order_type": BRACKET_ORDER_TYPE,
                "stop_price": sl_str,
            },
            "take_profit_order": {
                "order_type": "limit_order",
                "stop_price": tp_str,
                "limit_price": tp_str,
            },
            "bracket_stop_trigger_method": BRACKET_STOP_TRIGGER
        }
        
        if BRACKET_ORDER_TYPE == "limit_order":
            bracket_payload["stop_loss_order"]["limit_price"] = sl_str

        await self.api_client.post("/v2/orders/bracket", bracket_payload)
        
        direction = "LONG" if side == "buy" else "SHORT"
        return product_id, direction

    async def _notify_monitor(self, symbol: str, size: float, product_id: int):
        try:
            message = {
                "type": "start_monitoring",
                "symbol": symbol,
                "size": float(size),
                "product_id": int(product_id),
                "timestamp": time.time(),
            }
            await self.redis.publish(MONITORING_CHANNEL, json.dumps(message))
        except Exception as e:
            logger.error("Failed to notify PositionMonitor: %s", e)

    async def _notify_tsl_manager(self, symbol, direction, size, product_id, entry_price):
        try:
            payload = {
                "command": "START_TSL",
                "symbol": symbol,
                "direction": direction,
                "size": float(size),
                "product_id": int(product_id),
                "entry_price": float(entry_price),
            }
            await self.redis.publish(TSL_CHANNEL, json.dumps(payload))
        except Exception as e:
            logger.error("Failed to notify TSL Manager: %s", e)