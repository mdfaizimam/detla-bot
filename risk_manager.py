# --- risk_manager.py ---
# Complete Updated File (with persistence)

import asyncio
import logging
import time
import json
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
        self.daily_loss_limit = MAX_DAILY_LOSS_PERCENT
        
        # Initializing defaults, state loaded in async load method
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.circuit_open = False
        
        log.info(f"✅ RiskManager initialized (In-memory defaults): Max Drawdown={self.max_drawdown_pct*100}%, Daily Loss Limit={self.daily_loss_limit*100}%")

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
            return False, {"reason": "circuit_breaker_open"}
            
        # size_hint check is now implicit in ml_strategy calculation but safety check remains
        size_hint = signal.get("size_hint", 0.0)
        if size_hint <= 0:
            return False, {"reason": "invalid_or_zero_size", "size_hint": size_hint}
            
        # check drawdown
        if (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9) > self.max_drawdown_pct:
            self.circuit_open = True
            log.critical("🚨 CIRCUIT BREAKER TRIPPED: MAX DRAWDOWN EXCEEDED 🚨")
            return False, {"reason": "drawdown_breached"}
            
        # (Placeholder for margin checks)
        
        return True, {"ok": True}

    async def update_equity(self, new_equity: float):
        """
        Updates equity, checks for breaches, and persists the state.
        """
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        
        # Persist the state immediately
        await self._save_state_to_redis()
        
        # auto open circuit if daily loss > threshold (placeholder)
        if (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9) > self.daily_loss_limit:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: DAILY LOSS LIMIT EXCEEDED 🚨")


    async def reset_circuit(self):
        self.circuit_open = False
        log.info("circuit breaker reset")