# --- detla-bot/test_signal.py ---
# WORLD-CLASS TESTER
# ✅ UPDATED: Market Order Mode (Uses Mark Price)
# ✅ FIX: Handles Dictionary-based Position Sizing

import asyncio
import json
import time
import aiohttp
from redis import asyncio as aioredis
from config import REDIS_URL, SIGNAL_CHANNEL, BASE_POSITION_SIZE, DELTA_BASE_URL

# --- Configuration ---
TEST_SYMBOL = "BTCUSD"   
TEST_DIRECTION = "LONG"  # "LONG" or "SHORT"

async def get_mark_price(symbol: str) -> float:
    """Fetches the current Mark Price from Delta Exchange."""
    url = f"{DELTA_BASE_URL}/v2/tickers/{symbol}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data['result']['mark_price'])
                    print(f"🔍 Fetched Mark Price for {symbol}: {price}")
                    return price
                else:
                    print(f"❌ Failed to fetch price. HTTP {resp.status}")
                    return 0.0
        except Exception as e:
            print(f"❌ Error fetching price: {e}")
            return 0.0

async def generate_smart_signal(symbol: str, direction: str):
    """Generates a signal with valid SL/TP and Size."""
    
    # 1. Get Live Price
    entry_price = await get_mark_price(symbol)
    if entry_price == 0:
        print("❌ Cannot generate signal without live price.")
        return None

    # 2. Calculate realistic ATR (1%)
    mock_atr = entry_price * 0.01
    
    # 3. Calculate Bracket
    sl_dist = mock_atr * 2.0
    tp_dist = sl_dist * 1.5

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else: 
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    # 4. Extract Size
    if isinstance(BASE_POSITION_SIZE, dict):
        trade_size = BASE_POSITION_SIZE.get(symbol, 0.001)
    else:
        trade_size = float(BASE_POSITION_SIZE)

    print(f"📏 Using Test Size: {trade_size} for {symbol}")

    # 5. Construct Payload
    signal = {
        "symbol": symbol,
        "direction": direction,
        "confidence": 0.99, 
        "size_hint": trade_size,
        "trigger_price": entry_price,
        "tp_price": round(tp_price, 4),
        "sl_price": round(sl_price, 4),
        "atr": round(mock_atr, 4),
        "strategy": "MANUAL_TEST_MARKET",
        "timestamp": int(time.time() * 1_000_000)
    }
    
    return signal

async def publish_test_signal():
    redis = await aioredis.from_url(REDIS_URL)
    
    print(f"🚀 Generating {TEST_DIRECTION} Test Signal for {TEST_SYMBOL} (Market Mode)...")
    
    signal = await generate_smart_signal(TEST_SYMBOL, TEST_DIRECTION)
    
    if signal:
        await redis.publish(SIGNAL_CHANNEL, json.dumps(signal))
        print(f"✅ Published to {SIGNAL_CHANNEL}:")
        print(json.dumps(signal, indent=2))
        print("\n👉 Check your 'main.py' logs to see the Executor react!")
    
    await redis.aclose()

if __name__ == "__main__":
    try:
        asyncio.run(publish_test_signal())
    except KeyboardInterrupt:
        pass