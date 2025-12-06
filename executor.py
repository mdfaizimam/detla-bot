# --- detla-bot/executor.py ---
# ⚡ EXECUTOR: MARKET ORDER + BRACKET (Robust & Fast)
# ✅ LOGIC: Immediate Market Entry -> Immediate Bracket Placement
# ✅ FIX: Removed invalid 'trail_amount': None from bracket payload
# ✅ FIX: Increased wait time to capture correct fill price

import asyncio
import json
import logging
import time
from typing import Optional, Any, Dict, Tuple

from redis import asyncio as aioredis

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    SIGNAL_CHANNEL,
    MONITORING_CHANNEL,
    USER_AGENT,
    DMS_ID,
    TSL_ENABLED,
    TSL_CHANNEL,
    config,
    BRACKET_STOP_TRIGGER,
    BRACKET_ORDER_TYPE,
    REDIS_POSITION_LOCK_PREFIX
)
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager

logger = logging.getLogger("executor")

class OrderExecutionManager:
    REDIS_POSITION_LOCK_TTL = 60

    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.redis = redis_client
        self.api_client = api_client
        self.session = api_client.session
        self.risk_manager = risk_manager
        self.product_info_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ OrderExecutionManager initialized (Market Order Mode).")

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
        
        status, response = await self.api_client.get(f"/v2/products/{symbol}")
        if status == 200 and response.get("success"):
            product = response.get("result", {})
            if product:
                try:
                    tick_size = float(product.get("tick_size", "0.5"))
                    precision = 0
                    if "." in str(tick_size):
                        precision = len(str(tick_size).split(".")[-1])
                    
                    info = {
                        "id": int(product.get("id")),
                        "tick_size": tick_size,
                        "precision": precision,
                        "symbol": product.get("symbol")
                    }
                    self.product_info_cache[symbol] = info
                    return info
                except Exception as e:
                    logger.error("Error parsing product info: %s", e)
        return None

    async def _acquire_position_lock(self, symbol: str, timeout: int = 5) -> bool:
        deadline = time.time() + timeout
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        lock_value = json.dumps({"symbol": symbol, "ts": time.time()})
        
        while time.time() < deadline:
            ok = await self.redis.set(lock_key, lock_value, ex=self.REDIS_POSITION_LOCK_TTL, nx=True)
            if ok: return True
            await asyncio.sleep(0.25)
        
        logger.warning("⚠️ Could not acquire lock for %s (Busy)", symbol)
        return False

    async def _release_position_lock(self, symbol: str):
        try:
            lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
            await self.redis.delete(lock_key)
        except Exception: pass

    async def _handle_signal(self, signal: dict):
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        
        # Determine Size
        base_size_config = config["BASE_POSITION_SIZE"]
        if isinstance(base_size_config, dict):
            size_hint = base_size_config.get(symbol, 1)
        else:
            size_hint = float(base_size_config)
        
        int_size = int(size_hint) # Force integer for API

        if not symbol or not direction: return

        # Risk Check
        ok, info = await self.risk_manager.validate_signal(signal)
        if not ok:
            logger.warning("Signal rejected by RiskManager: %s", info)
            return

        # Acquire Lock
        if not await self._acquire_position_lock(symbol):
            return

        try:
            product_info = await self._get_product_info(symbol)
            if not product_info:
                raise Exception("Product Info Unavailable")
            
            # Dynamic TP/SL from Strategy
            tp_price = float(signal.get("tp_price", 0))
            sl_price = float(signal.get("sl_price", 0))
            
            if tp_price == 0 or sl_price == 0:
                logger.error("Invalid TP/SL in signal. Aborting.")
                await self._release_position_lock(symbol)
                return

            # Execute Linked Orders (Entry + Bracket)
            side = "buy" if direction == "LONG" else "sell"
            
            res = await self._place_linked_orders(
                symbol, 
                side, 
                int_size, 
                tp_price, 
                sl_price, 
                product_info
            )
            
            if not res:
                # Entry failed
                await self._release_position_lock(symbol)
                return

            product_id, ret_direction, filled_avg_price = res

            # Notify Systems
            await self._notify_monitor(symbol, int_size, product_id)
            if TSL_ENABLED:
                await self._notify_tsl_manager(symbol, ret_direction, int_size, product_id, filled_avg_price)

            logger.info("✅ Trade Executed for %s @ $%.2f. Lock HELD.", symbol, filled_avg_price)

        except Exception as e:
            logger.error("❌ Error handling signal for %s: %s", symbol, e, exc_info=True)
            await self._release_position_lock(symbol)

    async def _place_linked_orders(self, symbol, side, size, tp_price, sl_price, product_info) -> Optional[Tuple[int, str, float]]:
        product_id = product_info["id"]
        precision = product_info["precision"]
        
        # 1. Place Market Entry
        entry_payload = {
            "product_id": product_id,
            "size": size,
            "side": side,
            "order_type": "market_order",
        }
        logger.info(f"📦 Placing Market Entry: {entry_payload}")
        
        status, entry_resp = await self.api_client.post("/v2/orders", entry_payload)
        
        if status != 200 or not entry_resp.get("success"):
            logger.error(f"❌ Market Entry Failed: {entry_resp}")
            if entry_resp.get('error', {}).get('code') == 'insufficient_margin':
                logger.critical("🛑 Insufficient Margin!")
            return None
        
        # Get Fill Price
        order_id = entry_resp["result"]["id"]
        filled_price = 0.0
        
        # Wait slightly longer for fill reflection to avoid $0.00 logs
        await asyncio.sleep(2.0)
        
        s, d = await self.api_client.get(f"/v2/orders/{order_id}")
        if s == 200:
            filled_price = float(d["result"].get("avg_fill_price", 0))
            # If still 0, try to use close price from signal or just proceed
        
        # 2. Place Bracket (TP/SL)
        sl_str = f"{sl_price:.{precision}f}"
        tp_str = f"{tp_price:.{precision}f}"

        # ✅ FIX: Removed 'trail_amount': None to prevent validation error
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
        
        # If using Limit Stop, set limit price
        if BRACKET_ORDER_TYPE == "limit_order":
            bracket_payload["stop_loss_order"]["limit_price"] = sl_str

        logger.info(f"🛡️ Placing Bracket: SL={sl_str} TP={tp_str}")
        b_status, b_resp = await self.api_client.post("/v2/orders/bracket", bracket_payload)
        
        if b_status != 200:
            logger.error(f"⚠️ Bracket Placement Failed: {b_resp}")
        
        direction = "LONG" if side == "buy" else "SHORT"
        return product_id, direction, filled_price

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