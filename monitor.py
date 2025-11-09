# --- monitor.py ---
# UPDATED: Refactored to use centralized DeltaAPIClient
# UPDATED: Accepts RiskManager instance to report PnL on position close.
# UPDATED: Imports REDIS_POSITION_LOCK_KEY and releases the
#          lock on position close, enabling the next trade.
# FIX: Replaced all REST API polling with a fully event-driven
#      listener on the PRIVATE_CHANNEL for 'positions' messages.

import asyncio
import json
import logging
import aiohttp
import urllib.parse
from redis import asyncio as aioredis
from config import (
    DELTA_BASE_URL, API_KEY, API_SECRET, REDIS_URL, 
    MONITORING_CHANNEL, 
    PRIVATE_CHANNEL, # ✅ NEW: Import private channel
    USER_AGENT,
    REDIS_POSITION_LOCK_KEY # ✅ --- FIX: Import global lock key ---
)
# NEW: Import the centralized client
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager


logger = logging.getLogger("monitor")

class PositionMonitor:
    """
    Accepts shared clients and monitors open positions via WebSocket events.
    """

    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.redis = redis_client   
        self.api_client = api_client # Keep for future use (e.g., fallback checks)
        self.risk_manager = risk_manager
        
        self.is_monitoring = False
        self.current_position = None # Will store {'symbol', 'size', 'product_id', 'appearance_confirmed'}
        
        self._created_redis = False

    # ... (connect and close methods are minimal) ...
    async def connect(self):
        """Initialize connections"""
        pass

    async def close(self):
        """Cleanup resources"""
        await self.stop_monitoring() 
        pass

    # ✅ --- FIX: Removed polling methods ---
    # Removed _get_position_by_id
    # Removed _monitoring_loop
    # --- END FIX ---

    async def _notify_position_closed(self, symbol: str, pnl: float):
        """Notify other components that position has closed"""
        try:
            message = {
                "type": "position_closed",
                "symbol": symbol,
                "pnl": pnl, 
                "timestamp": asyncio.get_event_loop().time()
            }
            await self.redis.publish(MONITORING_CHANNEL, json.dumps(message))
            logger.info(f"📢 Notified position closure: {symbol} (PnL: {pnl})")

            # Report PnL to RiskManager
            if self.risk_manager:
                logger.info(f"Reporting PnL of {pnl} to RiskManager.")
                asyncio.create_task(self.risk_manager.update_equity_with_pnl(pnl))
            else:
                logger.warning("No RiskManager instance found. Cannot report PnL.")

            # ✅ --- FIX: Release the global position lock ---
            # This is the action that allows the bot to search for a new trade.
            try:
                await self.redis.delete(REDIS_POSITION_LOCK_KEY)
                logger.info("🔓 Released global position lock. Bot is free to trade.")
            except Exception as e:
                logger.error("❌ FAILED to release global position lock: %s", e)
            # --- END FIX ---

        except Exception as e:
            logger.error(f"❌ Failed to notify position closure: {e}")

    async def start_monitoring(self, symbol: str, size: int, product_id: int):
        """Start monitoring a new position"""
        if self.is_monitoring:
            logger.warning(f"⚠️ Already monitoring {self.current_position['symbol']}, cannot monitor {symbol}")
            return False

        # ✅ --- FIX: Set state for event listener ---
        self.current_position = {
            'symbol': symbol, 
            'size': size, 
            'product_id': product_id,
            'appearance_confirmed': False # Wait for WS event
        }
        self.is_monitoring = True
        
        logger.info(f"🎯 Started monitoring position: {symbol} (size: {size}, ID: {product_id}). Waiting for WS 'positions' event...")
        # --- END FIX ---
        return True

    async def stop_monitoring(self):
        """Stop monitoring current position"""
        self.is_monitoring = False
        self.current_position = None

        # ✅ --- FIX: No monitoring_task to cancel ---
        # self.monitoring_task = None 
        logger.info("🛑 Position monitoring stopped")
        
        # ✅ --- FIX: Safety net to release lock if monitor is stopped externally ---
        try:
            # This is a safety check. The lock *should* be released by
            # _notify_position_closed, but if the monitor is stopped
            # for any other reason (e.g., manual shutdown), we must
            # release the lock to prevent the bot from being permantently stuck.
            if await self.redis.get(REDIS_POSITION_LOCK_KEY):
                 await self.redis.delete(REDIS_POSITION_LOCK_KEY)
                 logger.warning("🔓 Released global position lock during monitor shutdown (safety net).")
        except Exception as e:
            logger.error("❌ Error releasing lock in stop_monitoring (safety net): %s", e)
        # --- END FIX ---


    def is_active(self):
        """Check if currently monitoring a position"""
        return self.is_monitoring

    # ✅ --- FIX: Modified 'start' to be the main listener loop ---
    async def start(self):
        """Start the monitor service (listens for monitoring requests AND private events)"""
        pubsub = self.redis.pubsub()
        # Subscribe to both the control channel and the private data channel
        await pubsub.subscribe(MONITORING_CHANNEL, PRIVATE_CHANNEL)
        logger.info(f"🚀 Position Monitor started - waiting for events on {MONITORING_CHANNEL} and {PRIVATE_CHANNEL}")

        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                
                channel = msg['channel'] 
                
                try:
                    data = json.loads(msg.get("data"))

                    # --- 1. Handle "start monitoring" commands ---
                    if channel == MONITORING_CHANNEL:
                        if data.get("type") == "start_monitoring":
                            symbol = data.get("symbol")
                            size = data.get("size")
                            product_id = data.get("product_id") 
                            
                            if product_id:
                                await self.start_monitoring(symbol, size, product_id)
                            else:
                                logger.error(f"Monitor received 'start_monitoring' for {symbol} but was missing 'product_id'!")

                    # --- 2. Handle live "positions" data from WebSocket ---
                    elif channel == PRIVATE_CHANNEL:
                        # Only process if we are actively monitoring a trade
                        if not self.is_monitoring:
                            continue

                        # Check if it's a position update
                        if data.get("type") == "positions":
                            pos_product_id = data.get("product_id")
                            
                            # Check if this update is for the position we care about
                            if pos_product_id == self.current_position['product_id']:
                                current_size = float(data.get("size", 0))
                                
                                # --- A) Check for position closure ---
                                if current_size == 0:
                                    logger.info(f"✅ Detected position closed for {self.current_position['symbol']} via WebSocket.")
                                    # Use realized_pnl from the closure message
                                    final_pnl = float(data.get("realized_pnl", 0.0))
                                    
                                    await self._notify_position_closed(self.current_position['symbol'], final_pnl) 
                                    await self.stop_monitoring()
                                
                                # --- B) Check for position appearance ---
                                elif not self.current_position.get('appearance_confirmed'):
                                    self.current_position['appearance_confirmed'] = True
                                    logger.info(f"✅ Position appearance confirmed for {self.current_position['symbol']} (Size: {current_size}) via WebSocket.")

                except Exception as e:
                    logger.error(f"❌ Error processing monitoring request: {e}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Monitor cancelled.")
        except Exception as e:
            logger.error(f"💥 Monitor service crashed: {e}")
        finally:
            await pubsub.unsubscribe(MONITORING_CHANNEL, PRIVATE_CHANNEL)
            logger.info("🔻 Monitor stopped cleanly.")