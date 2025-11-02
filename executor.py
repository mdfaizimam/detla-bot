# --- executor.py ---
# Complete Updated File (With Precision Fix, Deadman Switch, and Network Retry)

import aiohttp
import asyncio
import json
import logging
import time
import urllib.parse
# FIX: Import client exceptions for error handling
from aiohttp import client_exceptions
from redis import asyncio as aioredis
from typing import Optional, Any, Dict

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    SIGNAL_CHANNEL,
    MONITORING_CHANNEL,
    config,
    USER_AGENT,
    TRADING_SYMBOLS,
    # ASSUMPTION: DMS_ID is now configured
    DMS_ID 
)
from utils.signing import generate_server_synced_signature
from risk_manager import RiskManager 

logger = logging.getLogger("executor")
logger.setLevel(logging.INFO)


class OrderExecutionManager:
    REDIS_POSITION_LOCK_KEY = "active_position" 
    REDIS_POSITION_LOCK_TTL = 60
    
    # FIX: Add retry constants for transient network errors
    MAX_RETRIES = 3 
    RETRY_DELAY = 1.0

    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession):
        self.session = http_session 
        self.redis = redis_client   
        self._process_lock = asyncio.Lock()
        
        self.risk_manager = RiskManager() 
        self.min_confidence = 0.0 
        
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.dms_id = DMS_ID # Use the DMS ID from config
        self.product_info_cache = {} # NEW: Cache for product info/tick size
        
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

    # NEW: Fetches and caches product details including ID and tick_size/precision.
    async def _get_product_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches product details including tick_size/precision and caches them."""
        if symbol in self.product_info_cache:
            return self.product_info_cache[symbol]

        # Use the dedicated endpoint GET /v2/products/{symbol}
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
                                # Calculate the number of decimal places for rounding
                                if '.' in tick_size_str:
                                    # Get number of digits after the decimal point
                                    precision = len(tick_size_str.split('.')[-1])
                                else:
                                    precision = 0

                                info = {
                                    "id": product_id,
                                    "tick_size": float(tick_size_str),
                                    "precision": precision
                                }
                                self.product_info_cache[symbol] = info
                                return info
                            except ValueError as ve:
                                logger.error(f"Invalid tick_size format received for {symbol}: {tick_size_str} -> {ve}")
                                return None

                logger.error(f"❌ Product info not found for {symbol} (HTTP {resp.status})")
                return None
        except Exception as e:
            logger.error(f"❌ Error fetching product info for {symbol}: {e}", exc_info=True)
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
                    logger.error(f"❌ Failed to fetch position {product_id}. HTTP Status: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error fetching position {product_id}: {e}")
            return None 


    async def _reconcile_open_positions(self):
        logger.info("Reconciling state... Checking for existing open positions.")
        
        product_id_map = {}
        for symbol in TRADING_SYMBOLS:
            # Use the new product info fetcher
            product_info = await self._get_product_info(symbol) 
            if product_info:
                product_id_map[symbol] = product_info['id']
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


    # UPDATED: Implements tick_size precision logic - CRITICAL FIX
    async def _place_bracket_order(self, symbol: str, side: str, size: float, tp_price: float, sl_price: float):
        """Places the bracket order with correct tick size precision. Returns product_id on success, None on failure."""
        
        product_info = await self._get_product_info(symbol) 
        if not product_info:
            logger.error(f"❌ Product Info missing for {symbol}. Blocking trade.")
            return None
        
        product_id = product_info["id"]
        precision = product_info["precision"]
        
        # CRITICAL FIX: Round price to the required tick size precision
        final_tp_price = round(tp_price, precision)
        final_sl_price = round(sl_price, precision)
        
        # Format numbers as strings for full precision to the API
        final_tp_price_str = f"{final_tp_price:.{precision}f}"
        final_sl_price_str = f"{final_sl_price:.{precision}f}"

        logger.info("📊 [%s] Placing order: Side=%s, Size=%.2f | TP=%s | SL=%s (Precision: %d)", 
                    symbol, side, size, final_tp_price_str, final_sl_price_str, precision)

        trigger_method = config.get("BRACKET_STOP_TRIGGER", "last_traded_price")

        combined_payload = {
            "product_id": product_id,
            "size": size,
            "side": side,
            "order_type": "market_order",
            "bracket_stop_trigger_method": trigger_method,
            "bracket_take_profit_price": final_tp_price_str,
            "bracket_take_profit_limit_price": final_tp_price_str, 
            "bracket_stop_loss_price": final_sl_price_str
        }

        logger.info(f"📦 Placing native bracket order: {combined_payload}")
        # Use generic sender for POST /v2/orders
        order_resp = await self._send_order("POST", "/v2/orders", combined_payload) 

        if not order_resp.get("success"):
            logger.error("❌ Native bracket order failed: %s", order_resp)
            return None 

        order_id = order_resp.get("result", {}).get("id")
        logger.info(f"🎯 Native Bracket Order Placed Successfully. Entry Order ID: {order_id}")
        return product_id 

    # UPDATED: Generic signed request sender for DMS and Orders with retry logic
    async def _dms_send_request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        """Sends a signed authenticated request with retry logic for network errors."""
        body = json.dumps(payload) if payload else ""
        
        # Note: Query parameters are generally omitted for POST/PUT/DELETE
        signature, timestamp = await generate_server_synced_signature(method, path, body, "") 
        
        headers = {
            "api-key": self.api_key,
            "timestamp": str(timestamp),
            "signature": signature,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT
        }
        
        url = f"{DELTA_BASE_URL}{path}"
        
        # FIX: Implement retry loop for transient network errors
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with self.session.request(method, url, data=body, headers=headers) as resp:
                    try:
                        response_json = await resp.json()
                        return response_json
                    except Exception as e:
                        # Log and treat as a failure if decoding fails but HTTP succeeded (rare)
                        logger.error(f"Failed to decode API response for {path}: {e}")
                        return {"success": False, "error": f"HTTP {resp.status} - Decode Failed"}
            
            # Catch transient network errors for retries
            except (client_exceptions.ServerDisconnectedError, asyncio.TimeoutError) as e:
                if attempt < self.MAX_RETRIES:
                    logger.warning(f"Server connection error on {path} (Attempt {attempt}/{self.MAX_RETRIES}). Retrying in {self.RETRY_DELAY:.1f}s... Error: {type(e).__name__}")
                    await asyncio.sleep(self.RETRY_DELAY)
                else:
                    logger.error(f"❌ API request to {path} failed after {self.MAX_RETRIES} attempts. Error: {type(e).__name__}", exc_info=True)
                    return {"success": False, "error": str(e)}
            
            # Catch all other exceptions (e.g., DNS error, other non-retriable exceptions)
            except Exception as e:
                logger.error(f"❌ Unhandled error sending request to {path} on attempt {attempt}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}

    # Updated _send_order to use _dms_send_request internally
    async def _send_order(self, method: str, path: str, payload: dict) -> dict:
        return await self._dms_send_request(method, path, payload)
        
    # NEW: Deadman Switch Logic (Robustness Feature)
    async def _dms_create_heartbeat(self):
        """Register the Deadman Switch with the exchange to cancel orders on failure."""
        path = "/v2/heartbeat/create"
        payload = {
            "heartbeat_id": self.dms_id,
            "impact": "contracts",
            "contract_types": ["perpetual_futures"],
            # Setting unhealthy_count to 1 means the first missed ACK cancels all open orders
            "config": [{"action": "cancel_orders", "unhealthy_count": 1}] 
        }
        resp = await self._dms_send_request("POST", path, payload)
        if resp.get("success"):
            logger.info("✅ Deadman Switch Heartbeat created successfully.")
        else:
            logger.error(f"❌ Failed to create Deadman Switch: {resp}")
        
    async def _dms_send_acknowledgment(self, ttl_ms: int = 30000):
        """Send periodic acknowledgment to keep the switch alive."""
        path = "/v2/heartbeat"
        payload = {"heartbeat_id": self.dms_id, "ttl": ttl_ms}
        resp = await self._dms_send_request("POST", path, payload)
        if resp.get("success"):
            logger.debug(f"❤️ DMS Acknowledgment sent: {resp.get('result')}")
        else:
            # FIX: Only log as warning here, the retry loop will have handled the transient error
            logger.warning(f"⚠️ DMS Acknowledgment failed: {resp}")

    async def _dms_loop(self):
        """Starts the continuous DMS acknowledgment loop."""
        if not self.dms_id:
            logger.warning("🚫 DMS_ID not set in config. Deadman Switch disabled.")
            return

        await self._dms_create_heartbeat()
        
        while True:
            try:
                # Send acknowledgment just before the TTL expires (TTL: 30s)
                await self._dms_send_acknowledgment() 
                await asyncio.sleep(25) 
            except asyncio.CancelledError:
                logger.info("DMS loop cancelled.")
                break
            except Exception as e:
                logger.error(f"💥 DMS loop crashed: {e}", exc_info=True)
                await asyncio.sleep(5) # Wait and retry


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
        # Start the Deadman Switch before reconciling or trading
        asyncio.create_task(self._dms_loop())
        
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
                
                channel = msg['channel']
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