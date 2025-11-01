# --- executor.py ---
# Complete Updated File (Rollback to Static Sizing/Fixed Rounding)

import aiohttp
import asyncio
import json
import logging
import time
import urllib.parse
from redis import asyncio as aioredis

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    SIGNAL_CHANNEL,
    MONITORING_CHANNEL,
    config,
    USER_AGENT,
    TRADING_SYMBOLS 
)
from utils.signing import generate_server_synced_signature
from risk_manager import RiskManager 

logger = logging.getLogger("executor")
logger.setLevel(logging.INFO)


class OrderExecutionManager:
    REDIS_POSITION_LOCK_KEY = "active_position" 
    REDIS_POSITION_LOCK_TTL = 60

    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession):
        self.session = http_session 
        self.redis = redis_client   
        self._process_lock = asyncio.Lock()
        
        self.risk_manager = RiskManager() 
        self.min_confidence = 0.0 
        
        self.api_key = API_KEY
        self.api_secret = API_SECRET

        logger.info("✅ OrderExecutionManager initialized.")

    async def close(self):
        logger.info("🔒 Executor connections closed by main.")
        pass 

    async def _acquire_position_lock(self, symbol: str, timeout: int = 5) -> bool:
        deadline = time.time() + timeout
        lock_value = json.dumps({"symbol": symbol, "ts": time.time()})
        while time.time() < deadline:
            ok = await self.redis.set(
                self.REDIS_POSITION_LOCK_KEY,
                lock_value,
                ex=self.REDIS_POSITION_LOCK_TTL,
                nx=True, 
            )
            if ok:
                logger.info("🔒 Acquired distributed lock for %s", symbol)
                return True
            await asyncio.sleep(0.25)
        logger.warning("⚠️ Could not acquire lock for %s", symbol)
        return False

    async def _release_position_lock(self):
        try:
            await self.redis.delete(self.REDIS_POSITION_LOCK_KEY)
            logger.info("🔓 Released distributed position lock")
        except Exception as e:
            logger.error("❌ Error releasing lock: %s", e)

    # Rollback to old simple ID fetcher
    async def _get_product_id(self, symbol: str):
        path = "/v2/products"
        # Use the general product endpoint and query the symbol
        params = {"symbol": symbol}
        url = f"{DELTA_BASE_URL}{path}" 
        try:
            async with self.session.get(url, params=params, headers={'User-Agent': config.get('USER_AGENT')}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Safely extract ID, assuming the desired product is returned first or alone
                    products = data.get("result", [{}])
                    product = products[0] if isinstance(products, list) else products
                    return product.get("id")
            logger.error(f"❌ Product ID not found for {symbol} (HTTP {resp.status})")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching product ID for {symbol}: {e}", exc_info=True)
            return None


    async def _get_position_by_id(self, product_id: int):
        """Fetches a single position by its product_id."""
        try:
            path = "/v2/positions"
            params = {"product_id": product_id} 
            query_string = urllib.parse.urlencode(params)
            
            signature, timestamp = await generate_server_synced_signature("GET", path, "", query_string)
            headers = {
                "api-key": self.api_key,
                "timestamp": str(timestamp),
                "signature": signature,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT
            }
            url = f"{DELTA_BASE_URL}{path}"
            logger.debug(f"🔍 Fetching position for product_id: {product_id}")

            async with self.session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("result", {})
                else:
                    return None
        except Exception as e:
            logger.error(f"❌ Error fetching position {product_id}: {e}")
            return None 


    async def _reconcile_open_positions(self):
        logger.info("Reconciling state... Checking for existing open positions.")
        
        product_id_map = {}
        for symbol in TRADING_SYMBOLS:
            # Revert to old simple ID fetching
            product_id = await self._get_product_id(symbol) 
            if product_id:
                product_id_map[symbol] = product_id
            await asyncio.sleep(0.2) 

        if not product_id_map:
            logger.error("❌ Could not fetch any product IDs. Reconciliation failed.")
            return

        for symbol, product_id in product_id_map.items():
            position = await self._get_position_by_id(product_id)
            if position and float(position.get("size", 0)) != 0:
                size = float(position.get("size", 0))
                logger.warning(f"⚠️ Found pre-existing open position for {symbol} (Size: {size}, ID: {product_id}).")
                
                lock_acquired = await self._acquire_position_lock(symbol)
                
                if lock_acquired:
                    await self._notify_monitor(symbol, size, product_id)
                    logger.info(f"🔒 Lock acquired. Monitor is now tracking existing {symbol} position.")
                    return
                else:
                    logger.error(f"❌ Found existing position for {symbol}, but FAILED to acquire lock.")
                    return
            
            await asyncio.sleep(0.2) 
        
        logger.info("✅ No pre-existing positions found. Ready for new signals.")


    # Rollback to old _place_bracket_order with static rounding
    async def _place_bracket_order(self, symbol: str, side: str, size: float, tp_price: float, sl_price: float):
        """Places the bracket order. Returns product_id on success, None on failure."""
        
        product_id = await self._get_product_id(symbol) # Static ID fetching
        if not product_id:
            logger.error(f"❌ Product ID missing for {symbol}. Blocking trade.")
            return None
        
        # Arbitrary rounding (rollback from tick_size)
        final_tp_price = round(tp_price, 2)
        final_sl_price = round(sl_price, 2)

        logger.info("📊 [%s] Placing order: Side=%s, Size=%.2f | TP=%.2f | SL=%.2f", 
                    symbol, side, size, final_tp_price, final_sl_price)

        trigger_method = config.get("BRACKET_STOP_TRIGGER", "last_traded_price")

        combined_payload = {
            "product_id": product_id,
            "size": size,
            "side": side,
            "order_type": "market_order",
            "bracket_stop_trigger_method": trigger_method,
            "bracket_take_profit_price": str(final_tp_price),
            "bracket_take_profit_limit_price": str(final_tp_price), 
            "bracket_stop_loss_price": str(final_sl_price)
        }

        logger.info(f"📦 Placing native bracket order: {combined_payload}")
        order_resp = await self._send_order(combined_payload)

        if not order_resp.get("success"):
            logger.error("❌ Native bracket order failed: %s", order_resp)
            return None 

        order_id = order_resp.get("result", {}).get("id")
        logger.info(f"🎯 Native Bracket Order Placed Successfully. Entry Order ID: {order_id}")
        return product_id 

    async def _send_order(self, payload: dict):
        path = "/v2/orders"
        url = f"{DELTA_BASE_URL}{path}"
        body = json.dumps(payload)
        
        signature, timestamp = await generate_server_synced_signature("POST", path, body, "")
        
        headers = {
            "api-key": self.api_key,
            "timestamp": str(timestamp),
            "signature": signature,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT
        }
        
        try:
            async with self.session.post(url, data=body, headers=headers) as resp:
                try:
                    response_json = await resp.json()
                    logger.debug(f"API Response ({resp.status}): {response_json}")
                    return response_json
                except Exception as e:
                    logger.error(f"Failed to decode API response: {e}")
                    return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Error sending order: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _process_signal(self, signal: dict):
        # Rollback to static sizing
        symbol = signal.get("symbol")
        direction = signal.get("direction", "").upper()
        
        is_safe, risk_details = await self.risk_manager.validate_signal(signal)
        if not is_safe:
            logger.warning(f"🚫 Skipping signal for {symbol} — RiskManager check failed: {risk_details.get('reason')}")
            return

        side = "buy" if direction == "LONG" else "sell"
        size_hint = float(config.get("BASE_POSITION_SIZE", 1))
        tp_price = signal.get("tp_price")
        sl_price = signal.get("sl_price")

        if not all([symbol, direction, size_hint, tp_price, sl_price]):
             logger.error(f"Invalid signal received, missing key fields: {signal}")
             return

        lock_acquired = await self._acquire_position_lock(symbol)
        if not lock_acquired:
            logger.warning(f"🚫 Skipping signal for {symbol} — active lock present (position may be open).")
            return

        try:
            product_id = await self._place_bracket_order(symbol, side, size_hint, tp_price, sl_price)
            
            if product_id:
                await self._notify_monitor(symbol, size_hint, product_id)
                logger.info("✅ Trade executed for %s [%s]", symbol, direction)
            else:
                logger.error("❌ Trade failed for %s [%s], releasing lock.", symbol, direction)
                await self._release_position_lock()
                
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}", exc_info=True)
            await self._release_position_lock()

    async def _notify_monitor(self, symbol: str, size: float, product_id: int):
        msg = {
            "type": "start_monitoring",
            "symbol": symbol,
            "size": size,
            "product_id": product_id,
            "timestamp": time.time(),
        }
        await self.redis.publish(MONITORING_CHANNEL, json.dumps(msg))
        logger.info(f"📢 Notified monitor to track {symbol} (ID: {product_id})")

    async def _handle_monitor_update(self, msg: dict):
        """ Handles updates from the PositionMonitor """
        if msg.get("type") == "position_closed":
            symbol = msg.get("symbol")
            logger.info(f"✅ Position closed for {symbol}. Releasing trade lock.")
            await self._release_position_lock()
            
            pnl = msg.get("pnl", 0.0) 
            if pnl != 0.0:
                new_equity = self.risk_manager.current_equity + pnl 
                self.risk_manager.update_equity(new_equity)
                logger.info(f"Equity updated to: {new_equity} (PnL: {pnl})")
                
        elif msg.get("type") == "start_monitoring":
            logger.info(f"Monitor confirmed tracking for {msg.get('symbol')}")

    async def start(self):
        # Revert to old reconciliation logic
        await self._reconcile_open_positions()

        # 2. Now, subscribe to channels
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(SIGNAL_CHANNEL, MONITORING_CHANNEL)
        
        logger.info("🚀 Listening for trade signals and monitor updates...")

        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                
                channel = msg['channel'].decode('utf-8') 
                data_str = msg['data']

                try:
                    data = json.loads(data_str)
                    
                    if channel == SIGNAL_CHANNEL:
                        asyncio.create_task(self._process_signal(data))
                    elif channel == MONITORING_CHANNEL:
                        asyncio.create_task(self._handle_monitor_update(data))
                        
                except Exception as e:
                    logger.error(f"Error processing message from {channel}: {e}")

        except asyncio.CancelledError:
            logger.info("Executor cancelled.")
        except Exception as e:
            logger.error(f"💥 Executor crashed: {e}", exc_info=True)
        finally:
            logger.info("🔻 Executor stopped cleanly.")