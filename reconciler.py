# --- detla-bot/reconciler.py ---
# 🛡️ RECONCILER: Detects & Adopts Orphan Positions on Startup
# ✅ FIX: Scans for positions immediately on startup
# ✅ FIX: Efficiently resolves Product IDs
# ✅ FIX: Robust API Handling
# ✅ FIX: Increased Lock TTL to 300s
# ✅ FIX: BROADCASTS STATE ON "HEALTHY" MATCH (Fixes "No TSL on Restart" bug)

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
        
        # ✅ FIX: Run reconciliation immediately on startup to catch existing trades
        log.info("🔎 Performing initial position scan...")
        try:
            await self.reconcile()
        except Exception as e:
            log.error(f"❌ Initial Reconciliation Failed: {e}")
        
        while not self._stop_event.is_set():
            try:
                # Wait for interval or stop event
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.check_interval)
                except asyncio.TimeoutError:
                    # Timeout reached, proceed to reconcile
                    pass
                
                if self._stop_event.is_set():
                    break

                await self.reconcile()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"❌ Reconciliation Error: {e}")
                await asyncio.sleep(5) # Brief pause on error

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
            
            # ✅ FIX: Fetch specific product only (Efficient)
            status, response = await self.api_client.get(f"/v2/products/{symbol}")
            if status == 200 and response.get("success"):
                product_data = response.get("result", {})
                product_id = int(product_data.get("id"))
                
                await self.redis.setex(cache_key, 3600, str(product_id))
                self.product_id_cache[symbol] = product_id
                return product_id
                
        except Exception as e:
            log.error(f"❌ Error getting product_id for {symbol}: {e}")
        
        return None

    async def get_active_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Checks if there is an active position for the symbol via API.
        Returns the position dictionary if found and open, else None.
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
                    if "size" in result:
                        size = float(result.get("size", 0))
                        if size != 0:
                            result["symbol"] = symbol
                            result["product_id"] = product_id
                            # Ensure entry price is float
                            result["entry_price"] = float(result.get("entry_price", 0))
                            return result
                            
                # ✅ FIX: Handle List response (Defensive coding)
                elif isinstance(result, list):
                    for position in result:
                        size = float(position.get("size", 0))
                        if size != 0:
                            position["symbol"] = symbol 
                            position["product_id"] = product_id
                            position["entry_price"] = float(position.get("entry_price", 0))
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
            await self._adopt_position(symbol, position_data, lock_key)
            return

        # --- SCENARIO 2: GHOST LOCK (Lock Exists, But No Position) ---
        if lock_active and not has_position:
            product_id = await self.get_product_id_for_symbol(symbol)
            
            # Safety check: Don't kill lock if there's very recent activity (race condition)
            if product_id and await self.has_recent_trade_activity(symbol, product_id):
                log.debug(f"⏳ Lock active for {symbol} with recent activity. Waiting...")
                return

            state = self.reconciliation_state.setdefault(symbol, {"consecutive_failures": 0})
            state["consecutive_failures"] += 1
            
            if state["consecutive_failures"] >= 3:
                log.warning(f"👻 Ghost Lock detected for {symbol}. Releasing.")
                await self.redis.delete(lock_key)
                
                # Notify TSL to stop via monitor channel
                await self.redis.publish(MONITORING_CHANNEL, json.dumps({
                    "type": "reconciler_lock_released", 
                    "symbol": symbol
                }))
                
                state["consecutive_failures"] = 0
            return

        # --- SCENARIO 3: HEALTHY (Lock & Position Match) ---
        if lock_active and has_position:
            self.reconciliation_state.setdefault(symbol, {})["consecutive_failures"] = 0
            
            # Refresh lock TTL
            await self.redis.expire(lock_key, 300)
            
            # ✅ CRITICAL FIX: Always broadcast adoption even if healthy.
            # This ensures TSLManager picks up the position after a bot restart,
            # even if Redis still had the lock. TSLManager is idempotent (ignores dupes).
            # log.debug(f"🔄 Refreshing state for {symbol} (Healthy)")
            await self._adopt_position(symbol, position_data, lock_key, update_lock=False)

    async def _adopt_position(self, symbol: str, position_data: dict, lock_key: str, update_lock: bool = True):
        """Helper to broadcast adoption events."""
        product_id = int(position_data.get("product_id"))
        size = float(position_data.get("size"))
        entry_price = float(position_data.get("entry_price", 0))
        direction = "LONG" if size > 0 else "SHORT"
        
        if update_lock:
            # Re-create Lock
            lock_value = json.dumps({"symbol": symbol, "ts": time.time(), "status": "adopted"})
            await self.redis.set(lock_key, lock_value, ex=300)
        
        # Notify Monitor
        monitor_msg = {
            "type": "start_monitoring",
            "symbol": symbol,
            "size": size,
            "product_id": product_id,
            "timestamp": time.time(),
            "source": "reconciler_adoption"
        }
        await self.redis.publish(MONITORING_CHANNEL, json.dumps(monitor_msg))
        
        # Notify TSL Manager
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
        
        if update_lock:
             log.info(f"✅ Adopted {symbol} position. Size: {size}, Entry: {entry_price}")

    async def reconcile(self):
        for symbol in TRADING_SYMBOLS:
            await self.reconcile_symbol(symbol)