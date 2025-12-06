# --- detla-bot/reconciler.py ---
# 🛡️ RECONCILER: Detects & Adopts Orphan Positions on Startup
# ✅ FIX: Scans for positions even if Redis Lock is missing
# ✅ FIX: Corrected attribute name typo (product_id_cache)
# ✅ FIX: Handles API returning Dict instead of List for single positions

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from redis import asyncio as aioredis
from utils.api_client import DeltaAPIClient
from config import (
    REDIS_POSITION_LOCK_PREFIX, 
    TRADING_SYMBOLS, 
    MONITORING_CHANNEL,
    TSL_CHANNEL,
    REDIS_DATA_TTL
)

log = logging.getLogger("reconciler")

class StateReconciler:
    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient):
        self.redis = redis_client
        self.api_client = api_client
        self.check_interval = 60  # Check every 60 seconds
        self._stop_event = asyncio.Event()
        
        # Cache for product IDs
        self.product_id_cache: Dict[str, int] = {}
        
        # Track reconciliation state per symbol
        self.reconciliation_state: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        log.info("🛡️ Reconciliation Service Started (Orphan Detection Active)")
        # Initial delay to allow other services to connect
        await asyncio.sleep(5)
        
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
                        product_id = int(product.get("id"))
                        await self.redis.setex(cache_key, 3600, str(product_id))
                        # ✅ FIX: Correct variable name used here
                        self.product_id_cache[symbol] = product_id
                        return product_id
        except Exception as e:
            log.error(f"❌ Error getting product_id for {symbol}: {e}")
        
        return None

    async def get_active_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Checks if there is an active position for the symbol.
        Returns the position dictionary if found, else None.
        """
        product_id = await self.get_product_id_for_symbol(symbol)
        if not product_id:
            return None
        
        try:
            # Fetch position for specific product
            status, response = await self.api_client.get(
                "/v2/positions", 
                params={"product_id": str(product_id)}
            )
            
            if status == 200 and response.get("success"):
                result = response.get("result")
                
                # ✅ FIX: Handle Dict response (Single Position)
                if isinstance(result, dict):
                    # Check if it's a valid position object (has 'size')
                    if "size" in result:
                        size = float(result.get("size", 0))
                        if size != 0:
                            result["symbol"] = symbol
                            result["product_id"] = product_id # Ensure ID is present
                            return result
                            
                # ✅ FIX: Handle List response (Just in case API behavior changes)
                elif isinstance(result, list):
                    for position in result:
                        size = float(position.get("size", 0))
                        if size != 0:
                            position["symbol"] = symbol 
                            return position
                
            return None
                
        except Exception as e:
            log.error(f"❌ Exception checking position for {symbol}: {e}")
            return None

    async def has_recent_trade_activity(self, symbol: str, product_id: int) -> bool:
        """Check for recent orders to avoid race conditions."""
        try:
            status, orders_response = await self.api_client.get(
                "/v2/orders",
                params={
                    "product_ids": str(product_id),
                    "limit": 5
                }
            )
            if status == 200 and orders_response.get("success"):
                orders = orders_response.get("result", [])
                for order in orders:
                    # If there's an open order or a very recent fill, assume activity
                    if order.get("state") in ["open", "pending"]:
                        return True
                    created_at = float(order.get("created_at", 0)) / 1_000_000
                    if (time.time() - created_at) < 60: # Activity in last 60s
                        return True
            return False
        except Exception:
            return False

    async def reconcile_symbol(self, symbol: str):
        """Reconcile a single symbol."""
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        lock_active = await self.redis.exists(lock_key)
        
        # 1. Fetch Actual Position from Exchange
        position_data = await self.get_active_position_details(symbol)
        has_position = position_data is not None
        
        # --- SCENARIO 1: ORPHAN POSITION (No Lock, But Position Exists) ---
        if has_position and not lock_active:
            log.warning(f"⚠️ Orphan Position detected for {symbol}. Adopting...")
            
            # 1. Re-create Lock
            lock_value = json.dumps({"symbol": symbol, "ts": time.time(), "status": "adopted"})
            await self.redis.set(lock_key, lock_value, ex=60)
            
            product_id = int(position_data.get("product_id"))
            size = float(position_data.get("size"))
            # Entry price might be missing in some views, use 0 or fetch if needed
            entry_price = float(position_data.get("entry_price", 0))
            direction = "LONG" if size > 0 else "SHORT"
            
            # 2. Notify Monitor
            monitor_msg = {
                "type": "start_monitoring",
                "symbol": symbol,
                "size": size,
                "product_id": product_id,
                "timestamp": time.time(),
                "source": "reconciler_adoption"
            }
            await self.redis.publish(MONITORING_CHANNEL, json.dumps(monitor_msg))
            
            # 3. Notify TSL Manager
            tsl_msg = {
                "command": "START_TSL",
                "symbol": symbol,
                "direction": direction,
                "size": size,
                "product_id": product_id,
                "entry_price": entry_price,
                "source": "reconciler_adoption"
            }
            await self.redis.publish(TSL_CHANNEL, json.dumps(tsl_msg))
            
            log.info(f"✅ Adopted {symbol} position. Size: {size}, Entry: {entry_price}")
            return

        # --- SCENARIO 2: GHOST LOCK (Lock Exists, But No Position) ---
        if lock_active and not has_position:
            product_id = await self.get_product_id_for_symbol(symbol)
            if product_id and await self.has_recent_trade_activity(symbol, product_id):
                log.debug(f"⏳ Lock active for {symbol} with recent activity. Waiting...")
                return

            state = self.reconciliation_state.setdefault(symbol, {"consecutive_failures": 0})
            state["consecutive_failures"] += 1
            
            if state["consecutive_failures"] >= 3:
                log.warning(f"👻 Ghost Lock detected for {symbol}. Releasing.")
                await self.redis.delete(lock_key)
                await self.redis.publish(MONITORING_CHANNEL, json.dumps({"type": "position_closed", "symbol": symbol}))
                state["consecutive_failures"] = 0
            return

        # --- SCENARIO 3: HEALTHY (Lock & Position Match) ---
        if lock_active and has_position:
             self.reconciliation_state.setdefault(symbol, {})["consecutive_failures"] = 0
             await self.redis.expire(lock_key, 60)

    async def reconcile(self):
        for symbol in TRADING_SYMBOLS:
            await self.reconcile_symbol(symbol)