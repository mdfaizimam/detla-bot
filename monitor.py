# --- detla-bot/monitor.py ---
# Tracks open positions and releases the lock when they close.
# ⚡ UPGRADE: Releases Symbol-Specific locks for Multi-Symbol Concurrency.

import asyncio
import json
import logging
from redis import asyncio as aioredis
from utils.api_client import DeltaAPIClient
from config import (
    MONITORING_CHANNEL, 
    PRIVATE_CHANNEL, 
    REDIS_POSITION_LOCK_PREFIX, # ✅ NEW
    TRADING_SYMBOLS
)
from risk_manager import RiskManager

logger = logging.getLogger("monitor")

class PositionMonitor:
    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.redis = redis_client
        self.api_client = api_client
        self.risk_manager = risk_manager
        
        # Tracks which symbols are currently active
        self.active_symbols = set()
        self._stop_event = asyncio.Event()

    async def start(self):
        logger.info("🚀 Position Monitor started - waiting for events")
        
        # We listen to two channels:
        # 1. MONITORING_CHANNEL (from Executor): Tells us "I just opened a trade on XYZ"
        # 2. PRIVATE_CHANNEL (from WSManager): Gives us live updates on "v2/user_trades" and "positions"
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(MONITORING_CHANNEL, PRIVATE_CHANNEL)
        
        try:
            async for msg in pubsub.listen():
                if self._stop_event.is_set(): break
                if msg.get("type") != "message": continue
                
                channel = msg.get("channel")
                data = json.loads(msg.get("data"))
                
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
        """Executor says: 'I opened a position on BTCUSD'."""
        msg_type = data.get("type")
        if msg_type == "start_monitoring":
            symbol = data.get("symbol")
            if symbol:
                self.active_symbols.add(symbol)
                logger.info(f"👀 Started monitoring {symbol}. Active: {self.active_symbols}")

    async def _handle_private_update(self, data: dict):
        """Process WebSocket updates to detect position closure."""
        msg_type = data.get("type")
        
        # 1. Check 'positions' updates (Balance updates)
        if msg_type == "positions":
            # This stream often sends the full list of positions or updates
            # We check if our active symbols have size=0 here
            pass # Implementation often complex, usually user_trades is faster for close detection
            
        # 2. Check 'v2/user_trades' (Real-time executions)
        elif msg_type == "v2/user_trades":
            symbol = data.get("symbol")
            size = float(data.get("size", 0))
            side = data.get("side")
            
            # If we see a trade, we need to check if it CLOSED our position.
            # The simplest way is: If we are monitoring this symbol, query the API to confirm we are flat.
            if symbol in self.active_symbols:
                await self._verify_flat_status(symbol)

    async def _verify_flat_status(self, symbol: str):
        """
        Queries the API to check if the position size is effectively zero.
        If zero, releases the lock.
        """
        # Add a small delay to ensure the exchange backend has settled
        await asyncio.sleep(1.0) 
        
        status, response = await self.api_client.get("/v2/positions", params={"underlying_asset_symbol": symbol})
        
        if status == 200:
            positions = response.get("result", [])
            # Find position for this symbol
            # Delta often returns a list. Since we queried by symbol, should be short.
            
            size = 0.0
            pnl = 0.0
            
            for p in positions:
                # Double check symbol match (API sometimes returns all if filter fails)
                if p.get("product_symbol") == symbol or p.get("underlying_asset_symbol") == symbol:
                    size = float(p.get("size", 0))
                    pnl = float(p.get("realized_pnl", 0)) # Note: this is cumulative for the session often
                    
            if size == 0:
                logger.info(f"✅ Position closed for {symbol}. Releasing lock.")
                await self._release_lock(symbol, pnl)
            else:
                logger.debug(f"Position still open for {symbol}: Size {size}")

    async def _release_lock(self, symbol: str, pnl: float):
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            
        # ⚡ UPGRADE: Release specific lock key
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        await self.redis.delete(lock_key)
        
        # Update Risk Manager with PnL
        await self.risk_manager.update_pnl(pnl)
        
        logger.info(f"🔓 Lock released for {symbol}. PnL recorded.")