import asyncio
import json
import time
import logging
from redis import asyncio as aioredis

# Import all the config variables we need
from config import (
    REDIS_URL,
    SIGNAL_CHANNEL,
    PRIVATE_CHANNEL,
    REDIS_POSITION_LOCK_KEY
)

# --- FAKE TEST DATA ---

# 1. A fake signal to trigger the Executor
FAKE_SIGNAL = {
    "symbol": "BTCUSD",
    "direction": "LONG",
    "confidence": 0.99,
    "size_hint": 1.0,
    "trigger_price": 65000.0,
    "tp_price": 66000.0,
    "sl_price": 64000.0,
    "atr": 150.0,
    "candles": []
}

# 2. A fake "position closed" message from the exchange
# This triggers the Monitor
FAKE_POSITION_CLOSE_MESSAGE = {
    "type": "positions",
    "action": "update",
    "symbol": "BTCUSD",
    "product_id": 27,  # Using the real BTCUSD product ID
    "size": 0,         # <-- Size 0 means closed
    "realized_pnl": 12.34
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [E2E_TEST]: %(message)s")
log = logging.getLogger("E2E_TEST")


async def run_e2e_test():
    """
    Runs a full end-to-end test on the live bot.
    """
    redis_client = None
    try:
        redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        log.info(f"Connected to Redis at {REDIS_URL}")
        
        # --- PRE-TEST CLEANUP ---
        log.info("Cleaning up any old locks...")
        await redis_client.delete(REDIS_POSITION_LOCK_KEY)
        
        # --- PHASE 1: INJECT SIGNAL & VERIFY LOCK ---
        log.info("--- PHASE 1: EXECUTOR TEST ---")
        log.info(f"Injecting FAKE_SIGNAL to '{SIGNAL_CHANNEL}'...")
        await redis_client.publish(SIGNAL_CHANNEL, json.dumps(FAKE_SIGNAL))
        
        log.info(f"Waiting for lock '{REDIS_POSITION_LOCK_KEY}' to appear...")
        
        lock_value = None
        for _ in range(10): # Wait up to 10 seconds
            lock_value = await redis_client.get(REDIS_POSITION_LOCK_KEY)
            if lock_value:
                break
            await asyncio.sleep(1)
            
        assert lock_value is not None, "TEST FAILED: Executor did not acquire the lock."
        log.info(f"✅ SUCCESS: Lock acquired. Value: {lock_value}")
        
        # --- PHASE 2: INJECT CLOSE & VERIFY UNLOCK ---
        log.info("--- PHASE 2: MONITOR TEST ---")
        log.info(f"Injecting FAKE_POSITION_CLOSE_MESSAGE to '{PRIVATE_CHANNEL}'...")
        await redis_client.publish(PRIVATE_CHANNEL, json.dumps(FAKE_POSITION_CLOSE_MESSAGE))
        
        log.info(f"Waiting for lock '{REDIS_POSITION_LOCK_KEY}' to be released...")
        
        lock_released = False
        for _ in range(10): # Wait up to 10 seconds
            lock_value = await redis_client.get(REDIS_POSITION_LOCK_KEY)
            if not lock_value:
                lock_released = True
                break
            await asyncio.sleep(1)
            
        assert lock_released, "TEST FAILED: Monitor did not release the lock."
        log.info("✅ SUCCESS: Lock was released by the Monitor.")
        
        log.info("🎉 --- E2E TEST PASSED! --- 🎉")

    except Exception as e:
        log.error(f"❌ --- E2E TEST FAILED --- ❌")
        log.error(f"Error: {e}", exc_info=True)
    finally:
        if redis_client:
            await redis_client.aclose()
            log.info("Redis connection closed.")


if __name__ == "__main__":
    log.info("Starting End-to-End test in 3 seconds...")
    log.info("PLEASE MAKE SURE main.py IS RUNNING IN ANOTHER TERMINAL.")
    time.sleep(3)
    asyncio.run(run_e2e_test())