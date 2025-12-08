# --- detla-bot/monitor.py ---
# ✅ FIXED: Explicitly kills TSL when position closes
# ✅ ROBUST: Cleans up Redis locks and state
# ✅ FIXED: "PnL Erasure" Bug - Calls risk_manager.sync_equity() on position close

import asyncio
import json
import logging
from redis import asyncio as aioredis
from utils.api_client import DeltaAPIClient
from config import (
    MONITORING_CHANNEL, 
    PRIVATE_CHANNEL, 
    REDIS_POSITION_LOCK_PREFIX,
    TSL_CHANNEL
)
from risk_manager import RiskManager

logger = logging.getLogger("monitor")

class PositionMonitor:
    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.redis = redis_client
        self.api_client = api_client
        self.risk_manager = risk_manager
        
        self.active_symbols = set()
        self._stop_event = asyncio.Event()

    async def start(self):
        logger.info("🚀 Position Monitor started - waiting for events")
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(MONITORING_CHANNEL, PRIVATE_CHANNEL)
        
        try:
            async for msg in pubsub.listen():
                if self._stop_event.is_set(): break
                if msg.get("type") != "message": continue
                
                channel = msg.get("channel")
                try:
                    data = json.loads(msg.get("data"))
                except:
                    continue
                
                if channel == MONITORING_CHANNEL:
                    await self._handle_new_position(data)
                elif channel == PRIVATE_CHANNEL:
                    await self._handle_private_update(data)
                    
        except asyncio.CancelledError:
            logger.info("Monitor cancelled.")
        finally:
            await pubsub.unsubscribe(MONITORING_CHANNEL, PRIVATE_CHANNEL)
            logger.info("🔻 Monitor stopped cleanly.")

    async def stop(self):
        self._stop_event.set()

    async def _handle_new_position(self, data: dict):
        msg_type = data.get("type")
        if msg_type == "start_monitoring":
            symbol = data.get("symbol")
            if symbol:
                self.active_symbols.add(symbol)
                logger.info(f"👀 Started monitoring {symbol}. Active: {self.active_symbols}")

    async def _handle_private_update(self, data: dict):
        # We look for user_trades (fills) or position updates to trigger a check
        msg_type = data.get("type")
        
        if msg_type == "v2/user_trades":
            symbol = data.get("sy") # v2 channel uses short keys
            if not symbol: symbol = data.get("symbol")
            
            # If we see a trade for an active symbol, check if it closed the position
            if symbol in self.active_symbols:
                # Use create_task to avoid blocking the listener
                asyncio.create_task(self._verify_flat_status(symbol))

    async def _verify_flat_status(self, symbol: str):
        """
        Queries the API to check if the position is closed.
        If closed, syncs equity, releases lock, and KILLS TSL.
        """
        await asyncio.sleep(1.5) # Wait for backend settlement
        
        try:
            # GET /positions returns open positions
            # Note: Delta API often filters out closed positions entirely from this list
            status, response = await self.api_client.get("/v2/positions", params={"underlying_asset_symbol": symbol})
            
            if status == 200:
                positions = response.get("result", [])
                
                is_open = False
                current_size = 0.0
                
                for p in positions:
                    p_sym = p.get("product_symbol") or p.get("symbol")
                    if p_sym == symbol:
                        current_size = float(p.get("size", 0))
                        if current_size != 0:
                            is_open = True
                            break # Found open position
                
                if not is_open or current_size == 0:
                    logger.info(f"✅ Position closed for {symbol}. Syncing Equity & Releasing lock.")
                    
                    # 1. Update Risk Manager (Capture PnL)
                    await self.risk_manager.sync_equity()
                    
                    # 2. Release Lock
                    await self._release_lock(symbol)

                    # 3. CRITICAL: Kill TSL Manager for this symbol
                    stop_payload = {"command": "STOP_TSL", "symbol": symbol}
                    await self.redis.publish(TSL_CHANNEL, json.dumps(stop_payload))
                    logger.info(f"☠️ Sent STOP_TSL command for {symbol}")

                else:
                    logger.debug(f"Position still open for {symbol}: Size {current_size}")
            else:
                 logger.warning(f"Failed to verify position status for {symbol}: HTTP {status}")
                 
        except Exception as e:
            logger.error(f"Error checking position status for {symbol}: {e}")

    async def _release_lock(self, symbol: str):
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        await self.redis.delete(lock_key)
        
        logger.info(f"🔓 Lock released for {symbol}.")