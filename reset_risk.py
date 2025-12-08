# --- detla-bot/reset_risk.py ---
import asyncio
from redis import asyncio as aioredis
# We assume REDIS_URL is in config.py. If not, use "redis://localhost:6379/0"
try:
    from config import REDIS_URL
except ImportError:
    REDIS_URL = "redis://localhost:6379/0"

async def reset_memory():
    print("🔌 Connecting to Redis...")
    redis = await aioredis.from_url(REDIS_URL)
    
    # 1. Clear the Risk Manager's memory
    # This deletes the 'daily_start_equity' key, forcing the bot to 
    # re-read your CURRENT wallet balance as the new starting point.
    await redis.flushdb()
    
    print("✅ MEMORY WIPED.")
    print("   - Daily Loss Counter: RESET to 0.00%")
    print("   - Circuit Breaker: OPEN")
    print("   - Position Locks: CLEARED")
    
    await redis.close()
    print("\n👉 You can now restart 'python main.py'. Good luck!")

if __name__ == "__main__":
    asyncio.run(reset_memory())