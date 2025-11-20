# --- test_signal.py ---
# WORLD-CLASS TESTER
# ✅ UPDATED: Auto-fetches LIVE prices to ensure valid SL/TP orders
# ✅ UPDATED: payload matches ml_strategy.py exactly
# ✅ FIX: Prevents "Order Rejected" errors due to bad price levels

import asyncio
import json
import time
import aiohttp
from redis import asyncio as aioredis
from config import REDIS_URL, SIGNAL_CHANNEL, BASE_POSITION_SIZE, DELTA_BASE_URL

# --- Configuration ---
# Set the symbol you want to test
TEST_SYMBOL = "SOLUSD" 
TEST_DIRECTION = "LONG"  # "LONG" or "SHORT"

async def get_live_price(symbol: str) -> float:
    """Fetches the current Mark Price from Delta Exchange to generate valid signals."""
    url = f"{DELTA_BASE_URL}/v2/tickers/{symbol}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data['result']['mark_price'])
                    print(f"🔍 Fetched Live Price for {symbol}: {price}")
                    return price
                else:
                    print(f"❌ Failed to fetch price. HTTP {resp.status}")
                    return 0.0
        except Exception as e:
            print(f"❌ Error fetching price: {e}")
            return 0.0

async def generate_smart_signal(symbol: str, direction: str):
    """Generates a signal with valid SL/TP based on live market data."""
    
    # 1. Get Live Price
    entry_price = await get_live_price(symbol)
    if entry_price == 0:
        print("❌ Cannot generate signal without live price.")
        return None

    # 2. Calculate realistic ATR (Approx 1% of price)
    mock_atr = entry_price * 0.01
    
    # 3. Calculate valid SL/TP based on direction
    # R/R Ratio 1.5:1 logic similar to strategy
    sl_dist = mock_atr * 2.0
    tp_dist = sl_dist * 1.5

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else: # SHORT
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    # 4. Construct Payload (Matches ml_strategy.py)
    signal = {
        "symbol": symbol,
        "direction": direction,
        "confidence": 0.99, # High confidence for testing
        "size_hint": BASE_POSITION_SIZE,
        "trigger_price": entry_price,
        "tp_price": round(tp_price, 4),
        "sl_price": round(sl_price, 4),
        "atr": round(mock_atr, 4),
        "candles": [], # Empty list is fine for execution test
        "timestamp": int(time.time() * 1_000_000)
    }
    
    return signal

async def publish_test_signal():
    # 1. Connect to Redis
    redis = await aioredis.from_url(REDIS_URL)
    
    print(f"🚀 Generating {TEST_DIRECTION} Test Signal for {TEST_SYMBOL}...")
    
    # 2. Generate Signal
    signal = await generate_smart_signal(TEST_SYMBOL, TEST_DIRECTION)
    
    if signal:
        # 3. Publish
        await redis.publish(SIGNAL_CHANNEL, json.dumps(signal))
        print(f"✅ Published to {SIGNAL_CHANNEL}:")
        print(json.dumps(signal, indent=2))
        print("\n👉 Check your 'main.py' logs to see the Executor react!")
    
    await redis.aclose()

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 🚀 IMPORTANT: Ensure 'main.py' is running in another terminal!
    # -----------------------------------------------------------------
    try:
        asyncio.run(publish_test_signal())
    except KeyboardInterrupt:
        pass