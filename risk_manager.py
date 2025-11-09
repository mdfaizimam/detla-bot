# --- risk_manager.py ---
# Complete Updated File (with persistence)
# UPDATED: Added 'update_equity_with_pnl' to be called by PositionMonitor
# FIX: Added start() method and _daily_reset_loop to reset the
#      circuit breaker and peak_equity for the new day.

import asyncio
import logging
import time
import json
import datetime # ✅ NEW: Import datetime
from typing import Tuple
from redis import asyncio as aioredis # Import Redis Client

# ✅ NEW: Import constants
from config import MAX_DRAWDOWN_PERCENT, MAX_DAILY_LOSS_PERCENT

log = logging.getLogger("risk_manager")

class RiskManager:
    """
    Manages portfolio risk, including drawdown, daily loss limits, and 
    risk-based position sizing calculations. State is persisted to Redis.
    """

    # ✅ NEW: Redis Keys for persistence
    REDIS_PEAK_EQUITY_KEY = "risk:peak_equity"
    REDIS_CURRENT_EQUITY_KEY = "risk:current_equity"
    
    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client 
        
        self.max_drawdown_pct = MAX_DRAWDOWN_PERCENT
        self.daily_loss_limit = MAX_DAILY_LOSS_PERCENT # This is the "daily" trigger
        
        # Initializing defaults, state loaded in async load method
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.circuit_open = False
        self._reset_task = None
        
        log.info(f"✅ RiskManager initialized (In-memory defaults): Max Drawdown={self.max_drawdown_pct*100}%, Daily Loss Limit={self.daily_loss_limit*100}%")

    # ✅ --- NEW: Start method for background task ---
    async def start(self):
        """Starts the daily reset loop task."""
        self._reset_task = asyncio.create_task(self._daily_reset_loop())
        log.info("RiskManager daily reset loop started.")

    async def _load_state_from_redis(self):
        """Loads persistent equity state from Redis."""
        try:
            peak_eq = await self._redis.get(self.REDIS_PEAK_EQUITY_KEY)
            current_eq = await self._redis.get(self.REDIS_CURRENT_EQUITY_KEY)
            
            if peak_eq:
                self.peak_equity = float(peak_eq)
                log.info(f"💾 Loaded Peak Equity: {self.peak_equity:.2f}")
            if current_eq:
                self.current_equity = float(current_eq)
                log.info(f"💾 Loaded Current Equity: {self.current_equity:.2f}")
            else:
                # If no current equity, start with peak
                self.current_equity = self.peak_equity
                log.info(f"💾 No Current Equity found, setting from Peak: {self.current_equity:.2f}")

            # Ensure current is not higher than peak on reload
            self.peak_equity = max(self.peak_equity, self.current_equity)

        except Exception as e:
            log.error(f"❌ Error loading risk state from Redis: {e}", exc_info=True)
            
    async def _save_state_to_redis(self):
        """Saves current equity state to Redis."""
        try:
            # Set both values atomically
            await self._redis.mset({
                self.REDIS_PEAK_EQUITY_KEY: str(self.peak_equity),
                self.REDIS_CURRENT_EQUITY_KEY: str(self.current_equity)
            })
        except Exception as e:
            log.error(f"❌ Error saving risk state to Redis: {e}", exc_info=True)

    async def validate_signal(self, signal: dict) -> Tuple[bool, dict]:
        """
        For every signal perform pre-trade checks:
         - circuit breaker
         - size limits (now implicit in calculated size)
         - available margin (placeholder)
        """
        if self.circuit_open:
            log.warning("Signal blocked: Circuit breaker is open.")
            return False, {"reason": "circuit_breaker_open"}
            
        # size_hint check is now implicit in ml_strategy calculation but safety check remains
        size_hint = signal.get("size_hint", 0.0)
        if size_hint <= 0:
            return False, {"reason": "invalid_or_zero_size", "size_hint": size_hint}
            
        # check drawdown
        current_drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9)
        
        # Check against the overall max drawdown (a hard stop)
        if current_drawdown > self.max_drawdown_pct:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: MAX DRAWDOWN EXCEEDED ({current_drawdown*100:.2f}%) 🚨")
            return False, {"reason": "max_drawdown_breached"}

        # Check against the daily loss limit (a soft, daily stop)
        if current_drawdown > self.daily_loss_limit:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: DAILY LOSS LIMIT EXCEEDED ({current_drawdown*100:.2f}%) 🚨")
            return False, {"reason": "daily_loss_breached"}
            
        # (Placeholder for margin checks)
        
        return True, {"ok": True}

    async def update_equity(self, new_equity: float):
        """
        Updates equity, checks for breaches, and persists the state.
        This is the private method that handles the logic.
        """
        self.current_equity = new_equity
        # Peak equity is the *daily* high-water mark for PnL calculation
        self.peak_equity = max(self.peak_equity, new_equity)
        
        # Persist the state immediately
        await self._save_state_to_redis()
        
        # Check if this update *causes* a breach
        current_drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9)
        if current_drawdown > self.daily_loss_limit:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED (on update): DAILY LOSS LIMIT EXCEEDED ({current_drawdown*100:.2f}%) 🚨")
        elif current_drawdown > self.max_drawdown_pct:
             self.circuit_open = True
             log.critical(f"🚨 CIRCUIT BREAKER TRIPPED (on update): MAX DRAWDOWN EXCEEDED ({current_drawdown*100:.2f}%) 🚨")

    # ✅ --- (Original) NEW FUNCTION ---
    async def update_equity_with_pnl(self, pnl: float):
        """
        Updates equity based on the PnL of a closed trade.
        This is the new link from the PositionMonitor.
        """
        try:
            # Load the most recent state from Redis to prevent race conditions
            await self._load_state_from_redis()
            
            new_equity = self.current_equity + float(pnl)
            
            log.info(f"Updating equity with PnL. Start: {self.current_equity:.4f}, PnL: {pnl:.4f}, End: {new_equity:.4f}")
            
            # Call the existing update_equity method which handles peak equity and saving
            await self.update_equity(new_equity)
            
        except Exception as e:
            log.error(f"❌ Error updating equity with PnL: {e}", exc_info=True)
    # --- END NEW FUNCTION ---

    # ✅ --- NEW: Daily Reset Logic ---
    async def reset_daily_limits(self):
        """Resets the daily loss circuit breaker and peak equity."""
        log.warning("--- RESETTING DAILY RISK LIMITS (00:00 UTC) ---")
        self.circuit_open = False
        
        # Reset the peak_equity to the current equity.
        # This starts a new "high-water mark" for the day.
        self.peak_equity = self.current_equity
        
        # Persist this new state
        await self._save_state_to_redis()
        log.info(f"Circuit breaker reset. New daily peak equity set to: {self.peak_equity:.4f}")

    async def _daily_reset_loop(self):
        """A background task that sleeps until midnight UTC, then resets limits."""
        try:
            while True:
                # Get current time in UTC
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                
                # Calculate the next midnight UTC
                tomorrow_utc = (now_utc + datetime.timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                
                # Calculate seconds to sleep
                seconds_until_midnight = (tomorrow_utc - now_utc).total_seconds()
                
                log.info(f"RiskManager reset: Sleeping for {seconds_until_midnight:.0f} seconds (until 00:00 UTC).")
                
                await asyncio.sleep(seconds_until_midnight)
                
                # --- Time to reset ---
                await self.reset_daily_limits()
                
                # Sleep for a short duration to prevent rapid looping if something is wrong
                await asyncio.sleep(60) 
                
        except asyncio.CancelledError:
            log.info("Daily reset loop cancelled.")
        except Exception as e:
            log.error(f"💥 Daily reset loop crashed: {e}", exc_info=True)