# --- test_signal.py ---
import asyncio
import json
import redis.asyncio as aioredis
import time
from config import REDIS_URL, SIGNAL_CHANNEL, BASE_POSITION_SIZE # Import necessary values

# --- Test Signals for End-to-End Execution ---

# Mock data for a LONG trade on ETHUSD
MOCK_SIGNAL_LONG = {
    "symbol": "ETHUSD",
    "direction": "LONG",
    "confidence": 0.95,
    "size_hint": BASE_POSITION_SIZE, # ✅ ADDED: Uses static size (e.g., 1.0)
    "trigger_price": 3850.00, # ⭐️ ADDED: The entry price
    "tp_price": 3950.00,  
    "sl_price": 3800.00
}

# Mock data for a SHORT trade on SOLUSD (using a priority symbol)
MOCK_SIGNAL_SHORT = {
    "symbol": "SOLUSD",
    "direction": "SHORT",
    "confidence": 0.95,
    "size_hint": BASE_POSITION_SIZE, # ✅ ADDED: Uses static size (e.g., 1.0)
    "trigger_price": 180.00, # ⭐️ ADDED: The entry price
    # Using more realistic prices for SOL (e.g., if price is ~$180)
    "tp_price": 170.00, 
    "sl_price": 190.00
}


async def publish_mock_signal(signal: dict):
    # Quick check for the mandatory size field before sending
    if 'size_hint' not in signal or 'trigger_price' not in signal:
        print("❌ Error: Signal must contain 'size_hint' and 'trigger_price' for the Executor.")
        return
    
    # Initialize redis client
    redis = aioredis.from_url(REDIS_URL)
    
    # Add a unique timestamp (in microseconds)
    signal["timestamp"] = int(time.time() * 1_000_000)
    
    await redis.publish(SIGNAL_CHANNEL, json.dumps(signal))
    
    print(f"✅ Published {signal['direction']} signal for {signal['symbol']} (Size: {signal['size_hint']}) to {SIGNAL_CHANNEL}")
    
    await redis.aclose()

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 🚀 IMPORTANT: Ensure your main bot (main.py) is running first!
    # -----------------------------------------------------------------
    
    # Uncomment the signal you wish to test:

    # Example 1: Test LONG trade on ETHUSD
    # asyncio.run(publish_mock_signal(MOCK_SIGNAL_LONG))

    # Example 2: Test SHORT trade on SOLUSD
    asyncio.run(publish_mock_signal(MOCK_SIGNAL_SHORT))