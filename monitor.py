# --- monitor.py ---
# UPDATED: Refactored to use centralized DeltaAPIClient
# UPDATED: Accepts RiskManager instance to report PnL on position close.
# UPDATED: Imports REDIS_POSITION_LOCK_KEY and releases the
#          lock on position close, enabling the next trade.

import asyncio
import json
import logging
import aiohttp
import urllib.parse
from redis import asyncio as aioredis
from config import (
    DELTA_BASE_URL, API_KEY, API_SECRET, REDIS_URL, 
    MONITORING_CHANNEL, USER_AGENT,
    REDIS_POSITION_LOCK_KEY # ✅ --- FIX: Import global lock key ---
)
# NEW: Import the centralized client
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager


logger = logging.getLogger("monitor")

class PositionMonitor:
    """
    Accepts shared clients and monitors open positions.
    """

    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.redis = redis_client   
        self.api_client = api_client 
        self.risk_manager = risk_manager
        
        self.is_monitoring = False
        self.current_position = None
        self.monitoring_task = None
        
        self._created_redis = False

    # ... (methods connect, close, _get_position_by_id are unchanged) ...
    async def connect(self):
        """Initialize connections"""
        pass

    async def close(self):
        """Cleanup resources"""
        await self.stop_monitoring() 
        pass

    async def _get_position_by_id(self, product_id: int):
        """Fetches a single position by its product_id."""
        try:
            path = "/v2/positions"
            params = {"product_id": product_id} 
            logger.debug(f"🔍 Fetching position for product_id: {product_id}")

            status, data = await self.api_client.get(path, params=params)

            if status == 200:
                return data.get("result", {})
            else:
                logger.error(f"❌ API error fetching position {product_id}: HTTP {status} | Body: {data.get('error')}")
                return None 

        except Exception as e:
            logger.error(f"❌ Error fetching position {product_id}: {e}")
            return None 

    async def _monitoring_loop(self):
        """Main monitoring loop - runs until position closes"""
        current_symbol = self.current_position['symbol']
        product_id = self.current_position['product_id']
        logger.info(f"🔍 Starting position monitoring for {current_symbol} (ID: {product_id})")

        await asyncio.sleep(2.0) 

        position_appeared = False
        max_appearance_checks = 5 
        check_interval = 2.0      

        logger.info(f"Verifying position appearance for {current_symbol}...")
        for i in range(max_appearance_checks):
            position = await self._get_position_by_id(product_id)
            
            if position and float(position.get("size", 0)) != 0:
                logger.info(f"✅ Position appearance confirmed for {current_symbol} (Size: {position.get('size')})")
                position_appeared = True
                break
            
            logger.info(f"Position not found, check {i+1}/{max_appearance_checks}. Retrying in {check_interval}s...")
            await asyncio.sleep(check_interval)

        if not position_appeared:
            logger.error(f"❌ Position {current_symbol} did not appear after {max_appearance_checks * check_interval}s. Assuming fill failed or 0.")
            await self._notify_position_closed(current_symbol, 0.0) # Send 0.0 PnL
            await self.stop_monitoring()
            return 

        logger.info(f"Tracking closure for {current_symbol}...")
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_known_pnl = 0.0 # Store PnL

        while self.is_monitoring:
            try:
                position = await self._get_position_by_id(product_id)

                if position is None:
                    consecutive_errors += 1
                    logger.warning(f"API error checking position {current_symbol}, error {consecutive_errors}/{max_consecutive_errors}")
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("❌ Too many errors querying position, stopping monitor")
                        await self._notify_position_closed(current_symbol, last_known_pnl) 
                        await self.stop_monitoring() 
                        break
                    await asyncio.sleep(5)
                    continue
                
                consecutive_errors = 0
                last_known_pnl = float(position.get("unrealized_pnl", 0.0)) # Continuously update last known PnL

                if float(position.get("size", 0)) == 0:
                    logger.info(f"✅ Detected position closed for {current_symbol}")
                    final_pnl = float(position.get("realized_pnl", 0.0))
                    await self._notify_position_closed(current_symbol, final_pnl) 
                    await self.stop_monitoring()
                    break

                await asyncio.sleep(5.0) 

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in position monitoring: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"❌ Too many errors, stopping monitoring for {self.current_position['symbol']}")
                    await self._notify_position_closed(current_symbol, last_known_pnl)
                    await self.stop_monitoring() 
                    break
                await asyncio.sleep(20)

        logger.info(f"🔄 Position monitoring stopped for {current_symbol}")

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

        self.current_position = {'symbol': symbol, 'size': size, 'product_id': product_id}
        self.is_monitoring = True

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"🎯 Started monitoring position: {symbol} (size: {size}, ID: {product_id})")
        return True

    async def stop_monitoring(self):
        """Stop monitoring current position"""
        self.is_monitoring = False
        self.current_position = None

        if self.monitoring_task:
            if not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass # Expected
        self.monitoring_task = None
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

    # ... (start method is unchanged) ...
    async def start(self):
        """Start the monitor service (listens for monitoring requests)"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(MONITORING_CHANNEL)
        logger.info("🚀 Position Monitor started - waiting for monitoring requests")

        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                
                channel = msg['channel'] 
                
                if channel != MONITORING_CHANNEL:
                    continue

                try:
                    data = json.loads(msg.get("data"))
                    if data.get("type") == "start_monitoring":
                        symbol = data.get("symbol")
                        size = data.get("size")
                        product_id = data.get("product_id") 
                        
                        if product_id:
                            await self.start_monitoring(symbol, size, product_id)
                        else:
                            logger.error(f"Monitor received 'start_monitoring' for {symbol} but was missing 'product_id'!")

                except Exception as e:
                    logger.error(f"❌ Error processing monitoring request: {e}")

        except asyncio.CancelledError:
            logger.info("Monitor cancelled.")
        except Exception as e:
            logger.error(f"💥 Monitor service crashed: {e}")
        finally:
            logger.info("🔻 Monitor stopped cleanly.")