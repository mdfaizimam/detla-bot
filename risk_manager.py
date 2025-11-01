# --- risk_manager.py ---
# Complete Updated File

import asyncio
import logging
import time
from typing import Tuple

# ✅ NEW: Import settings from config
from config import MAX_DRAWDOWN_PERCENT, MAX_DAILY_LOSS_PERCENT

log = logging.getLogger("risk_manager")

class RiskManager:
    """
    Basic risk manager:
     - position sizing (simple ATR-like placeholder)
     - drawdown breaker (in-memory snapshot; must be connected to persistent equity snapshots in prod)
     - validate_signal returns (bool, details)
    This file provides the required hooks — extend with real portfolio queries and Influx/Postgres snapshots.
    """

    def __init__(self):
        # ✅ NEW: Read from config
        self.max_drawdown_pct = MAX_DRAWDOWN_PERCENT
        self.daily_loss_limit = MAX_DAILY_LOSS_PERCENT
        
        # In-memory equity tracking (placeholder)
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.circuit_open = False
        
        log.info(f"✅ RiskManager initialized: Max Drawdown={self.max_drawdown_pct*100}%, Daily Loss Limit={self.daily_loss_limit*100}%")


    async def validate_signal(self, signal: dict) -> Tuple[bool, dict]:
        """
        For every signal perform pre-trade checks:
         - circuit breaker
         - size limits
         - available margin (placeholder)
        """
        if self.circuit_open:
            return False, {"reason": "circuit_breaker_open"}
            
        # placeholder size check
        size_hint = signal.get("size_hint", 0.01)
        # ✅ FIX: This check was wrong, size_hint is now in contracts, not pct
        # We just need to ensure it's a positive number.
        if size_hint <= 0:
            return False, {"reason": "invalid_size_hint", "size_hint": size_hint}
            
        # check drawdown
        if (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9) > self.max_drawdown_pct:
            self.circuit_open = True
            log.critical("🚨 CIRCUIT BREAKER TRIPPED: MAX DRAWDOWN EXCEEDED 🚨")
            return False, {"reason": "drawdown_breached"}
            
        # (Placeholder for margin checks)
        
        return True, {"ok": True}

    def update_equity(self, new_equity: float):
        """
        Updates equity and checks for breaches.
        In a real system, this would be called by a PnL service.
        """
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        
        # auto open circuit if daily loss > threshold (placeholder)
        if (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9) > self.daily_loss_limit:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: DAILY LOSS LIMIT EXCEEDED 🚨")


    async def reset_circuit(self):
        self.circuit_open = False
        log.info("circuit breaker reset")