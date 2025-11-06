# --- executor.py ---
# UPDATED: To use centralized DeltaAPIClient
# FIX: Passes correct trigger_price (not sl_price) to TSL Manager

import aiohttp
import asyncio
import json
import logging
import time
import urllib.parse
from redis import asyncio as aioredis
from typing import Optional, Any, Dict, Tuple

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    SIGNAL_CHANNEL,
    MONITORING_CHANNEL,
    config,
    USER_AGENT,
    TRADING_SYMBOLS,
    DMS_ID,
    TSL_CHANNEL, 
    TSL_ENABLED 
)
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager 

logger = logging.getLogger("executor")


class OrderExecutionManager:
    REDIS_POSITION_LOCK_KEY = "active_position" 
    REDIS_POSITION_LOCK_TTL = 60
    
    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.session = api_client.session # For unauthenticated calls
        self.api_client = api_client       # For authenticated calls
        self.redis = redis_client   
        self._process_lock = asyncio.Lock()
        
        self.risk_manager = risk_manager 
        self.min_confidence = 0.0 
        
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.dms_id = DMS_ID
        self.product_info_cache = {}
        
        logger.info("✅ OrderExecutionManager initialized (using DeltaAPIClient).")

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

    async def _get_product_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches product details including tick_size/precision and caches them."""
        if symbol in self.product_info_cache:
            return self.product_info_cache[symbol]

        path = f"/v2/products/{symbol}" 
        url = f"{DELTA_BASE_URL}{path}"
        
        try:
            # Use the shared unauthenticated session
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
            logger.debug(f"🔍 Fetching position for product_id: {product_id}")

            # UPDATED: Use the centralized API client
            status, data = await self.api_client.get(path, params=params)

            if status == 200:
                return data.get("result", {})
            else:
                logger.error(f"❌ Failed to fetch position {product_id}. HTTP Status: {status}")
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

        is_any_position_found = False

        for symbol, product_id in product_id_map.items():
            position = await self._get_position_by_id(product_id)
            if position and float(position.get("size", 0)) != 0:
                is_any_position_found = True
                size = float(position.get("size", 0))
                direction = "LONG" if size > 0 else "SHORT"
                entry_price = float(position.get("entry_price", 0.0))
                logger.warning(f"⚠️ Found pre-existing open position for {symbol} (Size: {size}, ID: {product_id}).")
                
                lock_acquired = await self._acquire_position_lock(symbol)
                
                if lock_acquired:
                    await self._notify_monitor(symbol, size, product_id)
                    # ✅ TSL INTEGRATION: Start TSL for reconciled position
                    if TSL_ENABLED:
                        # ⭐️ FIX: Pass entry_price, not sl_price (which we don't have here)
                        await self._notify_tsl_manager(symbol, direction, size, product_id, entry_price) 
                        logger.info(f"🔒 Lock acquired. TSL and Monitor are now tracking existing {symbol} position.")
                    else:
                        logger.info(f"🔒 Lock acquired. Monitor is now tracking existing {symbol} position.")
                    return
                else:
                    logger.error(f"❌ Found existing position for {symbol}, but FAILED to acquire lock.")
                    return
            
            await asyncio.sleep(0.2) 
        
        # ✅ FIX: If no position was found on the exchange, check and clear a stale Redis lock.
        if not is_any_position_found:
             lock_status = await self.redis.get(self.REDIS_POSITION_LOCK_KEY)
             if lock_status:
                 logger.warning("🧹 Found stale Redis position lock, but no active position on exchange. Clearing lock.")
                 await self._release_position_lock()
        
        logger.info("✅ No pre-existing positions found. Ready for new signals.")


    async def _place_linked_orders(self, symbol: str, side: str, size: float, tp_price: float, sl_price: float) -> Optional[Tuple[int, str]]:
        """
        [CRITICAL FIX] Places separate Market (Entry) and Stop Market (SL) orders.
        Returns product_id, direction (e.g., "SHORT") on success.
        """
        
        product_info = await self._get_product_info(symbol) 
        if not product_info:
            logger.error(f"❌ Product Info missing for {symbol}. Blocking trade.")
            return None
        
        product_id = product_info["id"]
        precision = product_info["precision"]
        direction = "LONG" if side == "buy" else "SHORT"
        
        # --- 1. Place Market Entry Order ---
        entry_payload = {
            "product_id": product_id,
            "size": abs(size),
            "side": side,
            "order_type": "market_order",
        }
        
        logger.info(f"📦 Placing Market Entry Order: {entry_payload}")
        # UPDATED: Use api_client.post
        entry_resp = await self._send_order("POST", "/v2/orders", entry_payload) 

        if not entry_resp.get("success"):
            logger.error(f"❌ Market Entry Order failed: {entry_resp}")
            return None 

        logger.info(f"🎯 Entry Order Placed Successfully. ID: {entry_resp.get('result', {}).get('id')}")

        
        # --- 2. Place Stop Loss (SL) Order (Standalone Stop Market) ---
        
        # SL order must be the opposite side of the entry order
        sl_side = "sell" if side == "buy" else "buy"
        
        # Round price to the required tick size precision
        final_sl_price = round(sl_price, precision)
        final_sl_price_str = f"{final_sl_price:.{precision}f}"
        
        # Use market_order type along with stop_price to place a Stop Market order
        sl_payload = {
            "product_id": product_id,
            "size": abs(size), 
            "side": sl_side,
            "order_type": "market_order", 
            "stop_price": final_sl_price_str,
            "reduce_only": True,
            "stop_order_type": "stop_loss_order" 
        }
        
        logger.info(f"📦 Placing Standalone Stop Market Order (SL): {sl_payload}")
        # UPDATED: Use api_client.post
        sl_resp = await self._send_order("POST", "/v2/orders", sl_payload)
        
        if not sl_resp.get("success"):
            logger.error(f"❌ Standalone Stop Market Order failed: {sl_resp}. WARNING: Position may be unprotected!")
            # TSL Manager will handle the failed tracking and log a warning.
        else:
            logger.info(f"🎯 Stop Loss Order Placed Successfully. ID: {sl_resp.get('result', {}).get('id')}")
        
        # --- 3. Take Profit (TP) Order OMITTED as requested. ---
        
        # The position is now successfully established, and the SL (Stop Market) is placed
        return product_id, direction 
    
    # Alias the new function as the placement function for simplicity
    _place_bracket_order = _place_linked_orders 

    # Updated _send_order to use api_client internally
    async def _send_order(self, method: str, path: str, payload: dict) -> dict:
        """Helper to map order sending to the new api_client."""
        if method.upper() == "POST":
            status, response = await self.api_client.post(path, payload=payload)
            return response
        elif method.upper() == "PUT":
            status, response = await self.api_client.put(path, payload=payload)
            return response
        # Add GET/DELETE if needed, though not used for orders
        return {"success": False, "error": f"Unsupported method {method}"}
        
        
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
        # UPDATED: Use api_client.post
        status, resp = await self.api_client.post(path, payload=payload)
        if resp.get("success"):
            logger.info("✅ Deadman Switch Heartbeat created successfully.")
        else:
            logger.error(f"❌ Failed to create Deadman Switch: {resp}")
        
    async def _dms_send_acknowledgment(self, ttl_ms: int = 30000):
        """Send periodic acknowledgment to keep the switch alive."""
        path = "/v2/heartbeat"
        payload = {"heartbeat_id": self.dms_id, "ttl": ttl_ms}
        
        # UPDATED: Use api_client.post. The retry logic is now *inside* the client.
        status, resp = await self.api_client.post(path, payload=payload)
        
        if resp.get("success"):
            logger.debug(f"❤️ DMS Acknowledgment sent: {resp.get('result')}")
            return True
        else:
            # Only log as warning for transient errors, but treat as failure if it's an API error
            logger.error(f"❌ DMS Acknowledgment failed persistently: {resp}")
            return False


    async def _dms_loop(self):
        """Starts the continuous DMS acknowledgment loop."""
        if not self.dms_id:
            logger.warning("🚫 DMS_ID not set in config. Deadman Switch disabled.")
            return

        await self._dms_create_heartbeat()
        
        while True:
            try:
                # Send acknowledgment and handle result
                success = await self._dms_send_acknowledgment()
                
                # If acknowledgment failed, pause execution longer before next attempt
                if not success:
                    await asyncio.sleep(config["API_RETRY_DELAY"] * 4) # Pause longer if connection is bad
                else:
                    # Send acknowledgment just before the TTL expires (TTL: 30s)
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
        # Use size from signal, which in this case should be static BASE_POSITION_SIZE
        size_hint = float(signal.get("size_hint", config.get("BASE_POSITION_SIZE", 1)))
        tp_price = signal.get("tp_price")
        sl_price = signal.get("sl_price") # Initial SL for the bracket order
        
        # ⭐️ FIX: Get the trigger_price from the signal to pass to the TSL manager
        trigger_price = signal.get("trigger_price") # This is the mid_price at time of signal

        if not all([symbol, direction, size_hint, tp_price, sl_price, trigger_price]):
             logger.error(f"Invalid signal received, missing key fields (incl. trigger_price): {signal}")
             return

        lock_acquired = await self._acquire_position_lock(symbol)
        if not lock_acquired:
            logger.warning(f"🚫 Skipping signal for {symbol} — active lock present (position may be open).")
            return

        try:
            # Using the manually linked order placement function
            result = await self._place_linked_orders(symbol, side, size_hint, tp_price, sl_price)
            
            if result:
                product_id, trade_direction = result
                
                # Trade successfully placed. Notify monitor and TSL manager.
                await self._notify_monitor(symbol, size_hint, product_id)
                
                if TSL_ENABLED:
                    # ⭐️ FIX: Pass the trigger_price as the starting price, not the sl_price
                    await self._notify_tsl_manager(symbol, trade_direction, size_hint, product_id, trigger_price)
                    
                logger.info("✅ Trade executed for %s [%s]. TSL Manager notified.", symbol, direction)
            else:
                logger.error("❌ Trade failed for %s [%s], releasing lock.", symbol, direction)
                await self._release_position_lock()
                
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}", exc_info=True)
            await self._release_position_lock()
            
    # NEW: Notify TSL Manager function
    async def _notify_tsl_manager(self, symbol: str, direction: str, size: float, product_id: int, entry_price: float):
        """Notify the TSL Manager to start trailing the stop loss for a new position."""
        msg = {
            "type": "start_tsl",
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "product_id": product_id,
            "entry_price": entry_price, # ⭐️ This now correctly holds the trade's entry price
            "timestamp": time.time(),
        }
        await self.redis.publish(TSL_CHANNEL, json.dumps(msg))
        logger.info(f"📢 Notified TSL Manager to start tracking {symbol} (ID: {product_id}) at entry: {entry_price}")

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
                # ✅ UPDATED: Call async update_equity on risk_manager
                await self.risk_manager.update_equity(new_equity)
                logger.info(f"Equity updated to: {self.risk_manager.current_equity} (PnL: {pnl})")
                
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
                        # Ensure handle monitor update is awaited as it now has async calls
                        asyncio.create_task(self._handle_monitor_update(data)) 
                        
                except Exception as e:
                    logger.error(f"Error processing message from {channel}: {e}")

        except asyncio.CancelledError:
            logger.info("Executor cancelled.")
        except Exception as e:
            logger.error(f"💥 Executor crashed: {e}", exc_info=True)
        finally:
            logger.info("🔻 Executor stopped cleanly.")