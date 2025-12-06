# --- detla-bot/reconciler.py ---
# FIXED: Changed 'product_ids' -> 'product_id' (singular) for position check

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, List
from redis import asyncio as aioredis
from utils.api_client import DeltaAPIClient
from config import REDIS_POSITION_LOCK_PREFIX, TRADING_SYMBOLS, MONITORING_CHANNEL

log = logging.getLogger("reconciler")

class StateReconciler:
    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient):
        self.redis = redis_client
        self.api_client = api_client
        self.check_interval = 90  # Run every 90 seconds (even less frequent)
        self._stop_event = asyncio.Event()
        
        # Cache for product IDs
        self.product_id_cache: Dict[str, int] = {}
        
        # Track reconciliation state per symbol
        self.reconciliation_state: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        log.info("🛡️ Reconciliation Service Started (Anti-Ghost Position Guard)")
        while not self._stop_event.is_set():
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"❌ Reconciliation Error: {e}")
            
            # Wait for interval or stop event
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.check_interval)
            except asyncio.TimeoutError:
                continue

    async def stop(self):
        self._stop_event.set()
        log.info("🛡️ Reconciliation Service Stopped")

    async def get_product_id_for_symbol(self, symbol: str) -> Optional[int]:
        """Get product_id for a symbol."""
        if symbol in self.product_id_cache:
            return self.product_id_cache[symbol]
        
        cache_key = f"product_id:{symbol}"
        try:
            # Try cache first
            cached = await self.redis.get(cache_key)
            if cached:
                product_id = int(cached)
                self.product_id_cache[symbol] = product_id
                return product_id
            
            # Fetch from API
            status, products_response = await self.api_client.get("/v2/products")
            if status == 200 and products_response.get("success"):
                products = products_response.get("result", [])
                for product in products:
                    if product.get("symbol") == symbol:
                        product_id = product.get("id")
                        if product_id:
                            product_id = int(product_id)
                            # Cache for future use
                            await self.redis.setex(cache_key, 300, str(product_id))
                            self.product_id_cache[symbol] = product_id
                            return product_id
        except Exception as e:
            log.error(f"❌ Error getting product_id for {symbol}: {e}")
        
        return None

    async def has_active_position_for_symbol(self, symbol: str) -> Optional[bool]:
        """Check if there's an active position for a specific symbol. Returns None if API error."""
        product_id = await self.get_product_id_for_symbol(symbol)
        if not product_id:
            log.warning(f"⚠️ Could not get product_id for {symbol}")
            return None
        
        try:
            # ✅ FIX: Use 'product_id' (singular) instead of 'product_ids'
            status, response = await self.api_client.get(
                "/v2/positions", 
                params={"product_id": str(product_id)}
            )
            
            if status == 200 and response and response.get("success"):
                positions = response.get("result", [])
                for position in positions:
                    size = float(position.get("size", 0))
                    if size != 0:
                        log.debug(f"✅ Found active position for {symbol}: size={size}")
                        return True
                # No position found
                return False
            
            # API error
            log.debug(f"API error checking position for {symbol}: HTTP {status}")
            return None
                
        except Exception as e:
            log.error(f"❌ Exception checking position for {symbol}: {e}")
            return None

    async def has_recent_trade_activity(self, symbol: str) -> Optional[bool]:
        """Check if there's recent trade activity for a symbol. Returns None if API error."""
        product_id = await self.get_product_id_for_symbol(symbol)
        if not product_id:
            return None
        
        try:
            # Check for recent filled orders (last 15 minutes) - WITHOUT states parameter to avoid API error
            status, orders_response = await self.api_client.get(
                "/v2/orders",
                params={
                    "product_ids": str(product_id), # Orders endpoint uses PLURAL
                    "limit": 20,
                    "order_by": "created_at",
                    "order_direction": "desc"
                }
            )
            
            if status == 200 and orders_response.get("success"):
                orders = orders_response.get("result", [])
                if orders:
                    current_time = time.time()
                    for order in orders:
                        # Check if order was filled recently
                        if order.get("state") == "filled":
                            created_at = order.get("created_at")
                            if created_at:
                                try:
                                    order_time = float(created_at)
                                    if current_time - order_time < 900:  # 15 minutes
                                        log.debug(f"✅ Recent filled order found for {symbol} within 15 minutes")
                                        return True
                                except (ValueError, TypeError):
                                    continue
            
            # Check for any stop-loss orders
            status2, stop_orders_response = await self.api_client.get(
                "/v2/orders",
                params={
                    "product_ids": str(product_id),
                    "stop_order_type": "stop_loss_order",
                    "limit": 10
                }
            )
            
            if status2 == 200 and stop_orders_response.get("success"):
                stop_orders = stop_orders_response.get("result", [])
                # Filter for open/pending orders manually
                active_stop_orders = [o for o in stop_orders if o.get("state") in ["open", "pending"]]
                if active_stop_orders:
                    log.debug(f"✅ Active stop-loss orders found for {symbol}: {len(active_stop_orders)} orders")
                    return True
            
            # Check for any open orders (without states parameter, filter manually)
            status3, all_orders_response = await self.api_client.get(
                "/v2/orders",
                params={
                    "product_ids": str(product_id),
                    "limit": 20
                }
            )
            
            if status3 == 200 and all_orders_response.get("success"):
                all_orders = all_orders_response.get("result", [])
                open_orders = [o for o in all_orders if o.get("state") == "open"]
                if open_orders:
                    log.debug(f"✅ Open orders found for {symbol}: {len(open_orders)} orders")
                    return True
                    
            return False
                
        except Exception as e:
            log.debug(f"Could not check recent trade activity for {symbol}: {e}")
            return None

    async def reconcile(self):
        """Reconcile all trading symbols."""
        for symbol in TRADING_SYMBOLS:
            try:
                await self.reconcile_symbol(symbol)
            except Exception as e:
                log.error(f"❌ Error reconciling {symbol}: {e}")

    async def reconcile_symbol(self, symbol: str):
        """Reconcile a single symbol."""
        # Initialize state for this symbol if not exists
        if symbol not in self.reconciliation_state:
            self.reconciliation_state[symbol] = {
                "last_check": 0,
                "consecutive_no_position": 0,
                "consecutive_api_errors": 0,
                "last_action": None
            }
        
        # 1. Check Lock State
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        lock_active = await self.redis.exists(lock_key)
        
        if not lock_active:
            # No lock exists, reset state
            self.reconciliation_state[symbol] = {
                "last_check": time.time(),
                "consecutive_no_position": 0,
                "consecutive_api_errors": 0,
                "last_action": "no_lock"
            }
            return
        
        # 2. Check Exchange State
        has_position = await self.has_active_position_for_symbol(symbol)
        
        # Handle API errors
        if has_position is None:
            # API error occurred
            state = self.reconciliation_state[symbol]
            state["consecutive_api_errors"] += 1
            state["last_check"] = time.time()
            
            if state["consecutive_api_errors"] >= 3:
                log.warning(f"⚠️ {symbol}: Multiple API errors ({state['consecutive_api_errors']}), checking recent activity")
                # Check for recent activity as fallback
                has_recent_activity = await self.has_recent_trade_activity(symbol)
                if has_recent_activity:
                    log.info(f"✅ {symbol}: Recent activity found despite API errors, keeping lock")
                    state["last_action"] = "activity_found_despite_api_errors"
                    state["consecutive_api_errors"] = 0
                else:
                    log.warning(f"⚠️ {symbol}: No recent activity and API errors - keeping lock (conservative)")
                    state["last_action"] = "api_errors_no_activity"
            else:
                log.debug(f"ℹ️ {symbol}: API error checking position ({state['consecutive_api_errors']}/3)")
                state["last_action"] = "api_error"
            
            return
        
        # Reset API error counter on successful API call
        self.reconciliation_state[symbol]["consecutive_api_errors"] = 0
        
        # 3. Compare States
        if has_position:
            # Position exists and lock exists - everything is good
            self.reconciliation_state[symbol] = {
                "last_check": time.time(),
                "consecutive_no_position": 0,
                "consecutive_api_errors": 0,
                "last_action": "position_found"
            }
            log.debug(f"✅ {symbol}: Lock active, position exists - OK")
            return
        
        # 4. No position found but lock exists
        current_time = time.time()
        state = self.reconciliation_state[symbol]
        
        # Update consecutive count
        state["consecutive_no_position"] += 1
        state["last_check"] = current_time
        
        log.warning(f"⚠️ {symbol}: Lock active but no position found (consecutive: {state['consecutive_no_position']})")
        
        # Check for recent trade activity before releasing lock
        has_recent_activity = await self.has_recent_trade_activity(symbol)
        
        if has_recent_activity is None:
            # API error checking activity - be conservative
            log.warning(f"⚠️ {symbol}: API error checking recent activity, keeping lock (conservative)")
            state["last_action"] = "api_error_checking_activity"
            return
        
        if has_recent_activity:
            log.info(f"✅ {symbol}: Recent trade activity found, keeping lock")
            state["last_action"] = "activity_found"
            return
        
        # 5. No position and no recent activity - check if we should release
        # Only release after 3 consecutive checks with no position and no activity
        if state["consecutive_no_position"] >= 3:
            log.warning(f"⚠️ {symbol}: No position and no recent activity for 3 checks - releasing lock")
            await self.redis.delete(lock_key)
            log.info(f"🔓 Lock auto-released for {symbol} by Reconciler.")
            
            # Reset state
            self.reconciliation_state[symbol] = {
                "last_check": current_time,
                "consecutive_no_position": 0,
                "consecutive_api_errors": 0,
                "last_action": "lock_released"
            }
            
            # Notify other components
            try:
                await self.redis.publish(MONITORING_CHANNEL, json.dumps({
                    "type": "reconciler_lock_released",
                    "symbol": symbol,
                    "reason": "no_position_no_activity",
                    "message": f"Lock released for {symbol}: no position found and no recent trade activity for 3 consecutive checks"
                }))
            except Exception as e:
                log.error(f"Failed to publish lock release message: {e}")
        else:
            state["last_action"] = "waiting_for_more_checks"
            log.debug(f"ℹ️ {symbol}: Waiting for more checks before releasing lock ({state['consecutive_no_position']}/3)")

    async def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get current reconciliation status for all symbols."""
        status = {}
        for symbol in TRADING_SYMBOLS:
            try:
                lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
                lock_active = await self.redis.exists(lock_key)
                product_id = await self.get_product_id_for_symbol(symbol)
                
                status[symbol] = {
                    "lock_active": bool(lock_active),
                    "product_id": product_id,
                    "state": self.reconciliation_state.get(symbol, {})
                }
            except Exception as e:
                status[symbol] = {
                    "error": str(e),
                    "lock_active": False,
                    "product_id": None,
                    "state": {}
                }
        
        return status

    async def force_reconcile_symbol(self, symbol: str) -> Dict[str, Any]:
        """Force reconciliation of a single symbol."""
        try:
            lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
            lock_active = await self.redis.exists(lock_key)
            
            result = {
                "symbol": symbol,
                "lock_active": bool(lock_active),
                "product_id": await self.get_product_id_for_symbol(symbol),
                "action_taken": None,
                "reason": None
            }
            
            if lock_active:
                # Check position and activity
                has_position = await self.has_active_position_for_symbol(symbol)
                has_recent_activity = await self.has_recent_trade_activity(symbol)
                
                result["has_position"] = has_position
                result["has_recent_activity"] = has_recent_activity
                
                if has_position is False and has_recent_activity is False:
                    await self.redis.delete(lock_key)
                    result["action_taken"] = "lock_released"
                    result["reason"] = "no_position_no_activity"
                    
                    # Reset state
                    if symbol in self.reconciliation_state:
                        self.reconciliation_state[symbol] = {
                            "last_check": time.time(),
                            "consecutive_no_position": 0,
                            "consecutive_api_errors": 0,
                            "last_action": "manual_lock_release"
                        }
                    
                    # Notify
                    try:
                        await self.redis.publish(MONITORING_CHANNEL, json.dumps({
                            "type": "reconciler_lock_released",
                            "symbol": symbol,
                            "reason": "manual_reconciliation",
                            "message": f"Lock released during manual reconciliation"
                        }))
                    except Exception as e:
                        log.error(f"Failed to publish lock release message: {e}")
                else:
                    result["action_taken"] = "lock_kept"
                    if has_position:
                        result["reason"] = "position_exists"
                    elif has_recent_activity:
                        result["reason"] = "recent_activity_found"
                    else:
                        result["reason"] = "api_error_or_unknown"
            
            return result
                    
        except Exception as e:
            log.error(f"❌ Error in manual reconciliation for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e), "action_taken": None}