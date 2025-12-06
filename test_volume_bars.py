import asyncio
import redis.asyncio as aioredis
import orjson
from config import REDIS_URL, VOLUME_BAR_CHANNEL

async def listen_to_volume_bars():
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(VOLUME_BAR_CHANNEL)

    print(f"👂 Listening for Volume Bars on {VOLUME_BAR_CHANNEL}...")
    print("   (You should see data appear when market activity is high)")

    async for msg in pubsub.listen():
        if msg["type"] == "message":
            data = orjson.loads(msg["data"])
            print(f"📊 [VOLUME BAR] {data['symbol']} | Vol: {data['volume']} | VWAP: {data['vwap']:.2f}")

if __name__ == "__main__":
    try:
        asyncio.run(listen_to_volume_bars())
    except KeyboardInterrupt:
        print("🛑 Stopped.")