# --- tests/test_full_system.py ---
# 🚀 END-TO-END SYSTEM TEST
# Simulates the entire bot lifecycle:
# Market Data -> ML Strategy -> Signal -> Executor -> Order Placement -> Monitor -> Trade Closure -> Risk Update

import sys
import os

# ✅ FIX: Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import json
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from collections import defaultdict

# --- Import Bot Components ---
from risk_manager import RiskManager
from monitor import PositionMonitor
from executor import OrderExecutionManager
from ml_strategy import MLForecastingStrategy
from utils.api_client import DeltaAPIClient
from config import (
    ENRICHED_CHANNEL, SIGNAL_CHANNEL, MONITORING_CHANNEL, 
    PRIVATE_CHANNEL, REDIS_POSITION_LOCK_PREFIX
)

# --- 1. Infrastructure Mocks (Redis & API) ---

class InMemoryPubSub:
    """Simulates Redis PubSub for inter-component communication."""
    def __init__(self, broker):
        self.broker = broker
        self.queue = asyncio.Queue()
        self.active = True

    async def subscribe(self, *channels):
        for chan in channels:
            self.broker.subscribe(chan, self.queue)

    async def unsubscribe(self, *channels):
        self.active = False

    async def listen(self):
        while self.active:
            msg = await self.queue.get()
            yield msg

class MockRedis:
    """In-Memory Redis to allow components to talk without a real server."""
    def __init__(self):
        self.store = {}
        self.subscribers = defaultdict(list)

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store: return False
        self.store[key] = value
        return True

    async def delete(self, key):
        if key in self.store: del self.store[key]
        return 1

    async def mset(self, mapping):
        self.store.update(mapping)
        return True

    def pubsub(self):
        return InMemoryPubSub(self)

    async def publish(self, channel, message):
        # Broadcast to all listeners
        if channel in self.subscribers:
            payload = {"type": "message", "channel": channel, "data": message}
            for q in self.subscribers[channel]:
                await q.put(payload)
        return 1

    def subscribe(self, channel, queue):
        self.subscribers[channel].append(queue)
        
    async def close(self): pass
    async def aclose(self): pass

# --- 2. The End-to-End Test ---

@pytest.mark.asyncio
async def test_full_trade_lifecycle():
    """
    Scenario:
    1. System Start: Components initialize and sync equity.
    2. Market Data: 'Enriched' candle data arrives on Redis.
    3. Strategy: ML Model predicts 'LONG', publishes Signal.
    4. Executor: Receives Signal, places Market Order & Bracket on API.
    5. Monitor: Detects new trade, locks position.
    6. Market Event: 'User Trade' (Fill) arrives indicating trade closed.
    7. Monitor: Verifies position is flat, syncs Equity, releases lock.
    """
    
    # --- A. Setup Mocks ---
    redis = MockRedis()
    
    # API Client Mock - Critical for simulating Exchange responses
    api = AsyncMock(spec=DeltaAPIClient)
    
    # 1. Mock Wallet Balance (For RiskManager Startup)
    api.get.side_effect = lambda url, params=None: _handle_api_get(url, params)
    api.post.side_effect = lambda url, payload=None: _handle_api_post(url, payload)
    
    def _handle_api_get(url, params):
        # Risk Manager Balance Check
        if "wallet/balances" in url:
            return (200, {"success": True, "result": [{"asset_symbol": "USDT", "balance": "1000.0"}]})
        
        # Executor Product Info Check
        if "products/BTCUSD" in url:
            return (200, {"success": True, "result": {"id": 100, "tick_size": 0.5, "symbol": "BTCUSD"}})
        
        # Executor Fill Price Check
        # This simulates the API returning a filled price
        if "orders/555" in url:
            return (200, {"result": {"avg_fill_price": "50000.0"}})
            
        # Monitor Position Check (Returns EMPTY list = Position Closed)
        if "positions" in url:
            return (200, {"success": True, "result": []})
            
        return (404, {})

    def _handle_api_post(url, payload):
        # Order Placement
        if "orders" in url:
            return (200, {"success": True, "result": {"id": 555}})
        return (400, {})

    # --- B. Initialize Components ---
    
    # 1. Risk Manager
    risk = RiskManager(redis, api)
    await risk.start() # Should sync equity to 1000.0
    assert risk.current_equity == 1000.0, "Risk Manager failed to sync initial equity"

    # 2. Strategy (Mocking the heavy ML model loading)
    with patch("joblib.load") as mock_load:
        mock_model = MagicMock()
        # Predict Proba: [Short, Neutral, Long] -> High Long probability
        mock_model.predict_proba.return_value = [[0.05, 0.05, 0.9]]
        mock_model.feature_names_in_ = ["EMA_8"] # Dummy feature
        mock_load.return_value = mock_model
        
        strategy = MLForecastingStrategy(redis)
        # Override config for test deterministic behavior
        strategy.config["BASE_CONFIDENCE"] = 0.8
        strategy.config["GATEKEEPER_ENABLED"] = False
    
    # 3. Executor
    executor = OrderExecutionManager(redis, api, risk)
    executor.product_info_cache["BTCUSD"] = {"id": 100, "tick_size": 0.5, "precision": 1}

    # 4. Monitor
    monitor = PositionMonitor(redis, api, risk)

    # --- C. Start Background Loops ---
    # We use asyncio.create_task to run them as they would in main.py
    tasks = [
        asyncio.create_task(strategy.start(risk)),
        asyncio.create_task(executor.start()),
        asyncio.create_task(monitor.start())
    ]
    
    # Give them a moment to subscribe to Redis channels
    await asyncio.sleep(0.1)

    print("\n🚀 --- SYSTEM STARTED ---")

    # --- D. Step 1: Simulate Incoming Market Data ---
    print("📈 Sending Enriched Market Data...")
    market_data = {
        "symbol": "BTCUSD",
        "timestamp": 1234567890000000,
        "mid_price": 50000.0,
        "imbalance": 0.5,
        "funding_rate": 0.0001,
        "tas": {
            "5m": {
                "ema_20": 50000, "ema_50": 49000, "atr": 100, 
                "rsi_14": 50, "ker": 0.8, "adx": 25, 
                "bb_width": 0.1, "macd_hist": 10, "obv": 1000
            },
            "1m": {}
        }
    }
    await redis.publish(ENRICHED_CHANNEL, json.dumps(market_data))
    
    # ✅ FIX: Increased wait time to 2.0s to allow Executor to sleep(1.0) and place bracket
    await asyncio.sleep(2.0)

    # --- E. Verification 1: Order Placement ---
    # 1. Strategy should have fired a signal
    # 2. Executor should have called API to place order
    # 3. Executor should have published to Monitor
    
    # Check API Calls
    # We expect 2 POST calls: 1 for Market Entry, 1 for Bracket
    assert api.post.call_count >= 2, f"Executor called API {api.post.call_count} times, expected 2 (Entry + Bracket)"
    print("✅ Executor placed orders on API")

    # Check Monitor State
    # Monitor should have received "start_monitoring" and added symbol to active list
    assert "BTCUSD" in monitor.active_symbols, "Monitor failed to track new position"
    
    # Check Lock
    is_locked = await redis.get(f"{REDIS_POSITION_LOCK_PREFIX}BTCUSD")
    assert is_locked is not None, "Position Lock was not acquired in Redis"
    print("✅ Position Locked & Monitored")

    # --- F. Step 2: Simulate Trade Closure (Take Profit) ---
    print("💰 Simulating Take Profit Fill...")
    
    # Simulate a "user_trade" event from WebSocket saying we sold
    fill_msg = {
        "type": "v2/user_trades",
        "sy": "BTCUSD",
        "S": "sell", # Sold to close
        "s": 100,
        "p": "51000.0" # Profit!
    }
    await redis.publish(PRIVATE_CHANNEL, json.dumps(fill_msg))

    # Wait for Monitor -> API -> Risk flow
    # Monitor sleeps 1.5s inside verify_flat_status, so we wait 2.5s to be safe
    await asyncio.sleep(2.5) 

    # --- G. Verification 2: Trade Closure & Risk Sync ---
    
    # 1. Monitor should have checked API positions
    # 2. Monitor should have released the lock
    # 3. Risk Manager should have re-synced equity
    
    # Check Lock Release
    is_locked_now = await redis.get(f"{REDIS_POSITION_LOCK_PREFIX}BTCUSD")
    assert is_locked_now is None, "Monitor failed to release lock after trade close"
    print("✅ Lock Released")
    
    # Check Monitor Internal State
    assert "BTCUSD" not in monitor.active_symbols, "Monitor is still tracking closed position"
    
    # Verify Risk Manager synced equity (It calls API get wallet balances)
    # We count calls to wallet/balances. Initial start = 1. Sync on close = 1. Total = 2.
    balance_calls = [call for call in api.get.mock_calls if "wallet/balances" in str(call)]
    assert len(balance_calls) >= 2, "Risk Manager did not sync equity after trade close"
    print("✅ Equity Synced from Exchange")

    print("🏆 END-TO-END TEST PASSED SUCCESSFULLY")

    # Cleanup
    for t in tasks: t.cancel()