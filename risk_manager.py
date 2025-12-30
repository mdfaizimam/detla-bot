# --- risk_manager.py ---
# ✅ FIXED: "Ghost Equity" Bug - Fetches actual wallet balance on startup
# ✅ ADDED: sync_equity() to fetch real-time balance from API
# ✅ UPDATED: Daily reset logic

import asyncio
import logging
import time
import json
import datetime
from typing import Tuple, Optional
from redis import asyncio as aioredis

from config import MAX_DRAWDOWN_PERCENT, MAX_DAILY_LOSS_PERCENT
from utils.api_client import DeltaAPIClient

log = logging.getLogger("risk_manager")

class RiskManager:
    """
    Manages portfolio risk, including drawdown, daily loss limits, and 
    risk-based position sizing calculations. State is persisted to Redis.
    """

    REDIS_PEAK_EQUITY_KEY = "risk:peak_equity"
    REDIS_CURRENT_EQUITY_KEY = "risk:current_equity"
    
    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient):
        self._redis = redis_client 
        self._api_client = api_client # ✅ Added API Client
        
        self.max_drawdown_pct = MAX_DRAWDOWN_PERCENT
        self.daily_loss_limit = MAX_DAILY_LOSS_PERCENT 
        
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.circuit_open = False
        self._reset_task = None
        
        log.info(f"✅ RiskManager initialized: Max Drawdown={self.max_drawdown_pct*100}%, Daily Loss Limit={self.daily_loss_limit*100}%")

    def calculate_dynamic_size(self, symbol: str, confidence: float, regime: str, base_size: float) -> int:
        """
        Calculates position size based on Confidence AND Volatility Regime.
        Returns INTEGER contracts (floor 1).
        Regime Logic:
        - Low Vol (Calm): 1.0x (Standard)
        - Med Vol (Trend): 1.5x (Aggressive)
        - High Vol (Crash): 0.0x (Cash)
        """
        # 1. Regime Scaler
        regime_scaler = 1.0
        if "High Vol" in regime or "Crash" in regime:
            regime_scaler = 0.0 # CASH IS A POSITION
            log.warning(f"🛑 CRASH REGIME DETECTED! Sizing set to 0 for {symbol}")
            return 0
        elif "Med Vol" in regime or "Correction" in regime:
            regime_scaler = 1.5 # Trending / Volatile Upside
        
        # 2. Confidence Scaler (Linear scaling)
        conf_scaler = max(0.5, confidence)
        
        # 3. Raw Size
        raw_size = base_size * conf_scaler * regime_scaler
        
        # 4. Integer Validation
        final_size = int(max(1, raw_size))
        
        log.info(f"📏 Sizing {symbol}: Base={base_size} * Conf({conf_scaler:.2f}) * Reg({regime_scaler} - {regime}) = {raw_size:.2f} -> {final_size}")
        return final_size

    def get_adaptive_sl_multiplier(self, regime: str, base_mult=1.5) -> float:
        """
        Returns ATR multiplier for Stop Loss based on regime.
        - Low Vol: Tight (Base)
        - Med Vol: Loose (Base * 1.5)
        - High Vol: Very Loose (Base * 2.0) - though we likely won't trade
        """
        if "Med Vol" in regime:
            return base_mult * 1.5
        elif "High Vol" in regime:
            return base_mult * 2.0
        return base_mult


    async def start(self):
        """Starts the daily reset loop task and syncs initial equity."""
        # ✅ Sync equity immediately on start to prevent Ghost Equity bug
        await self.sync_equity()
        
        self._reset_task = asyncio.create_task(self._daily_reset_loop())
        log.info("RiskManager daily reset loop started.")

    async def sync_equity(self):
        """
        Fetches the actual wallet balance from Delta API and updates state.
        This is critical to ensure we don't trip circuit breakers on a fresh start.
        """
        try:
            # ✅ Fetch Wallet Balances
            status, response = await self._api_client.get("/v2/wallet/balances")
            if status == 200 and response.get("success"):
                wallets = response.get("result", [])
                total_usd_equity = 0.0
                
                # Sum up equity from USDT/USD wallets (Delta usually uses 'USDT' or 'DETO' or 'USD')
                for wallet in wallets:
                    symbol = wallet.get("asset_symbol")
                    if symbol in ["USDT", "USD"]: 
                        # Use total balance (balance + unrealized pnl is often reflected in margin fields, 
                        # but 'balance' is realized. We start with realized.)
                        total_usd_equity += float(wallet.get("balance", 0))
                
                if total_usd_equity > 0:
                    await self._load_state_from_redis()
                    
                    # If this is a fresh run (redis was empty/low), snap to actual balance
                    if self.current_equity <= 1.0:
                        self.current_equity = total_usd_equity
                        self.peak_equity = total_usd_equity
                        log.info(f"💰 Initial Equity Synced from API: ${self.current_equity:.2f}")
                    else:
                        # If we have running state, just update current, keep peak if higher
                        self.current_equity = total_usd_equity
                        self.peak_equity = max(self.peak_equity, self.current_equity)
                        log.info(f"💰 Equity Re-Synced from API: ${self.current_equity:.2f}")
                    
                    await self._save_state_to_redis()
                else:
                    log.warning("⚠️ Wallet balance is 0 or could not find USD/USDT wallet.")
            else:
                log.error(f"❌ Failed to fetch wallet balance: {response}")
        except Exception as e:
            log.error(f"❌ Error syncing equity: {e}")

    async def _load_state_from_redis(self):
        try:
            peak_eq = await self._redis.get(self.REDIS_PEAK_EQUITY_KEY)
            current_eq = await self._redis.get(self.REDIS_CURRENT_EQUITY_KEY)
            
            if peak_eq: self.peak_equity = float(peak_eq)
            if current_eq: self.current_equity = float(current_eq)
            
            # Sanity check
            self.peak_equity = max(self.peak_equity, self.current_equity)

        except Exception as e:
            log.error(f"❌ Error loading risk state from Redis: {e}")
            
    async def _save_state_to_redis(self):
        try:
            await self._redis.mset({
                self.REDIS_PEAK_EQUITY_KEY: str(self.peak_equity),
                self.REDIS_CURRENT_EQUITY_KEY: str(self.current_equity)
            })
        except Exception as e:
            log.error(f"❌ Error saving risk state to Redis: {e}")

    async def validate_signal(self, signal: dict) -> Tuple[bool, dict]:
        if self.circuit_open:
            return False, {"reason": "circuit_breaker_open"}
            
        size_hint = signal.get("size_hint", 0.0)
        if size_hint <= 0:
            return False, {"reason": "invalid_or_zero_size", "size_hint": size_hint}
            
        # check drawdown
        current_drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9)
        
        if current_drawdown > self.max_drawdown_pct:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: MAX DRAWDOWN ({current_drawdown*100:.2f}%)")
            return False, {"reason": "max_drawdown_breached"}

        if current_drawdown > self.daily_loss_limit:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: DAILY LOSS ({current_drawdown*100:.2f}%)")
            return False, {"reason": "daily_loss_breached"}
            
        return True, {"ok": True}

    async def update_equity(self, new_equity: float):
        """Internal update from PnL calculation."""
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        await self._save_state_to_redis()
        
        current_drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9)
        if current_drawdown > self.daily_loss_limit:
            self.circuit_open = True
            log.critical(f"🚨 CIRCUIT BREAKER TRIPPED (on update): {current_drawdown*100:.2f}%")

    async def update_equity_with_pnl(self, pnl: float):
        """Called by monitor.py to update equity with realized PnL."""
        # For robustness, we prefer re-syncing from API, but this is a fallback or quick update
        # Since we are moving to sync_equity(), this might be deprecated or used for logging
        pass 

    async def reset_daily_limits(self):
        """Resets the daily loss circuit breaker and peak equity."""
        log.warning("--- RESETTING DAILY RISK LIMITS (00:00 UTC) ---")
        self.circuit_open = False
        # Reset peak to current to start a new PnL day
        self.peak_equity = self.current_equity 
        await self._save_state_to_redis()
        log.info(f"Circuit breaker reset. New daily peak equity set to: {self.peak_equity:.4f}")

    async def _daily_reset_loop(self):
        try:
            while True:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                tomorrow_utc = (now_utc + datetime.timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                seconds_until_midnight = (tomorrow_utc - now_utc).total_seconds()
                log.info(f"RiskManager: Sleeping {seconds_until_midnight:.0f}s until reset.")
                await asyncio.sleep(seconds_until_midnight)
                await self.reset_daily_limits()
                await asyncio.sleep(60) 
        except asyncio.CancelledError:
            log.info("Daily reset loop cancelled.")