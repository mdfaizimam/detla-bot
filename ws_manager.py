# --- detla-bot/ws_manager.py ---
# ✅ OPTIMIZATION: Uses orjson for high-performance JSON handling
# ✅ FIX: Artificial Heartbeat Loop (prevents Error 10054)

import asyncio
import json
import logging
import time
import aiohttp
import redis.asyncio as aioredis
import orjson # ✅ FAST JSON
from config import (
    WS_URL, 
    RAW_CHANNEL, 
    PRIVATE_CHANNEL, 
    TRADING_SYMBOLS, 
    USER_AGENT, 
    SPOT_INDEX_SYMBOLS, 
    CONTROL_CHANNEL,
    API_KEY,  
    API_SECRET
)
from utils.signing import generate_ws_keyauth_signature_for_live

logger = logging.getLogger("ws_manager")

class WebSocketManager:
    """
    Handles WebSocket connection and data publishing.
    Accepts shared redis and http clients.
    """
    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession):
        self.redis = redis_client
        self.session = http_session
        self.ws_url = WS_URL
        self.ws = None
        self.is_authenticated = False
        self._stop_flag = False
        
        # Keepalive Task
        self._keepalive_task = None 

        # Reconnect policy
        self.reconnect_delay = 3
        self.reconnect_max = 60
        self.backoff_factor = 2
        
        self.PRIVATE_CHANNELS = {
            "v2/user_trades", 
            "orders", 
            "positions", 
            "margins"
        }

    async def connect(self):
        """Establish WebSocket connection and subscribe to channels."""
        if self._stop_flag:
            return

        try:
            logger.info(f"🔌 Connecting to WS: {self.ws_url}")
            self.ws = await self.session.ws_connect(self.ws_url, heartbeat=30, headers={'User-Agent': USER_AGENT})
            self.reconnect_delay = 3  # reset delay after successful connect

            logger.info("✅ Connected to Delta WebSocket")
            
            await self.ws.send_json({"type": "enable_heartbeat"}) 
            logger.info("❤️ Sent request to enable Delta WebSocket heartbeat.")

            await self.subscribe_public_channels()
            await self.authenticate()
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}", exc_info=True)
            asyncio.create_task(self.schedule_reconnect())

    async def authenticate(self):
        """Authorize with Delta Exchange API key."""
        try:
            if self.is_authenticated:
                return
            
            auth_payload = generate_ws_keyauth_signature_for_live(API_KEY, API_SECRET)
            await self.ws.send_json(auth_payload) 
            logger.info("🔐 Sent authentication payload to WebSocket")
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}", exc_info=True)

    # ✅ NEW: Artificial Heartbeat Generator
    async def _keepalive_loop(self):
        """Sends a dummy message to Redis every 10s to prevent TCP Idle Timeout."""
        logger.info("💓 Artificial Redis Heartbeat generator started.")
        while not self._stop_flag:
            try:
                # Send a small dummy payload
                payload = {
                    "type": "synthetic_heartbeat",
                    "timestamp": time.time()
                }
                # Use orjson here
                await self.redis.publish(RAW_CHANNEL, orjson.dumps(payload))
                await asyncio.sleep(10) # Pulse every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat error (harmless): {e}")
                await asyncio.sleep(10)

    async def _handle_control_messages(self):
        """Listens to the internal control channel."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(CONTROL_CHANNEL)
        logger.info(f"👂 Listening to internal control channel: {CONTROL_CHANNEL}")
        
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    try:
                        # Use orjson load
                        data = orjson.loads(msg['data'])
                        command = data.get("command")
                        symbol = data.get("symbol")
                        
                        if command == "RESUBSCRIBE_L2" and symbol:
                            logger.warning(f"🚨 Received RESUBSCRIBE_L2 command for {symbol}. Restarting L2 stream.")
                            asyncio.create_task(self._resubscribe_l2_updates(symbol))
                            
                    except Exception as e:
                        logger.error(f"Error processing control message: {e}")
        except asyncio.CancelledError:
            logger.info("Control message handler cancelled.")

    async def _resubscribe_l2_updates(self, symbol: str):
        if self.ws is None or self.ws.closed:
            logger.error("Cannot resubscribe: WebSocket is closed.")
            return

        unsubscribe_payload = {
            "type": "unsubscribe",
            "payload": {"channels": [{"name": "l2_updates", "symbols": [symbol]}]},
        }
        await self.ws.send_str(json.dumps(unsubscribe_payload))
        await asyncio.sleep(0.5) 
        
        subscribe_payload = {
            "type": "subscribe",
            "payload": {"channels": [{"name": "l2_updates", "symbols": [symbol]}]},
        }
        await self.ws.send_str(json.dumps(subscribe_payload))
        logger.info(f"⬅️ Sent resubscribe for l2_updates:{symbol}")

    async def subscribe_public_channels(self):
        standard_symbols = TRADING_SYMBOLS
        mark_price_symbols = [f"MARK:{symbol}" for symbol in standard_symbols]
        spot_symbols = [SPOT_INDEX_SYMBOLS[s] for s in standard_symbols if s in SPOT_INDEX_SYMBOLS]
        candle_resolutions = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        channels_payload = [
            {"name": "v2/ticker", "symbols": standard_symbols},
            {"name": "l2_updates", "symbols": standard_symbols},
            {"name": "all_trades", "symbols": standard_symbols},
            {"name": "funding_rate", "symbols": standard_symbols},
            {"name": "mark_price", "symbols": mark_price_symbols},
            {"name": "v2/spot_price", "symbols": spot_symbols}
        ]
        
        for res in candle_resolutions:
            channels_payload.append({
                "name": f"candlestick_{res}",
                "symbols": standard_symbols
            })

        payload = {
            "type": "subscribe",
            "payload": {"channels": channels_payload},
        }
        
        try:
            await self.ws.send_str(json.dumps(payload))
            logger.info(f"📈 Subscribed to channels for {TRADING_SYMBOLS}")
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to public channels: {e}")

    async def subscribe_private_channels(self):
        payload = {
            "type": "subscribe", 
            "payload": {
                "channels": [
                    {"name": "v2/user_trades", "symbols": ["all"]},
                    {"name": "orders", "symbols": ["all"]},
                    {"name": "positions", "symbols": ["all"]},
                    {"name": "margins"}
                ]
            }
        }
        try:
            await self.ws.send_str(json.dumps(payload))
            logger.info("🧠 Subscribed to private channels")
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to private channels: {e}")

    async def start(self):
        await self.connect()
        
        control_task = asyncio.create_task(self._handle_control_messages())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        
        try:
            while not self._stop_flag:
                if self.ws is None or self.ws.closed:
                    logger.warning("WS connection lost, waiting for reconnect...")
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                msg = await self.ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        # Use standard json here as initial parse is safer with it for mixed types
                        # orjson is strict. But let's stick to standard json for the initial wrapper
                        # to be safe, then orjson for redis.
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "unknown")

                    if msg_type in ("subscriptions", "key-auth"):
                        if msg_type == "key-auth" and data.get("success"):
                            self.is_authenticated = True
                            logger.info("✅ Authenticated successfully")
                            await self.subscribe_private_channels()
                        elif msg_type == "key-auth" and not data.get("success"):
                            logger.error(f"❌ Authentication Failed: {data.get('message')}")
                    elif msg_type not in ("ping", "pong", "heartbeat"):
                         logger.debug(f"🛰️ WS Message: {msg_type}")

                    if msg_type == "heartbeat":
                         continue

                    try:
                        if msg_type in self.PRIVATE_CHANNELS:
                            await self.redis.publish(PRIVATE_CHANNEL, orjson.dumps(data))
                        elif msg_type not in ("subscriptions", "key-auth"):
                            # ✅ Use orjson for heavy throughput
                            await self.redis.publish(RAW_CHANNEL, orjson.dumps(data))
                    except Exception as e:
                        logger.error(f"❌ Redis publish failed: {e}")

                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    logger.warning(f"⚠️ WebSocket closed or error: {msg.data}")
                    self.is_authenticated = False
                    asyncio.create_task(self.schedule_reconnect())

        except asyncio.CancelledError:
            logger.info("WebSocketHandler cancelled.")
        except Exception as e:
            logger.error(f"💥 WebSocket loop crashed: {e}", exc_info=True)
            if not self._stop_flag:
                asyncio.create_task(self.schedule_reconnect())
        finally:
            control_task.cancel()
            if self._keepalive_task: self._keepalive_task.cancel()
            await self.close()

    async def schedule_reconnect(self):
        if self._stop_flag: return
        self.is_authenticated = False
        if self.ws and not self.ws.closed: await self.ws.close()
        logger.warning(f"🔁 Attempting reconnect in {self.reconnect_delay}s...")
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * self.backoff_factor, self.reconnect_max)
        if not self._stop_flag: await self.connect()

    async def close(self):
        self._stop_flag = True
        if self._keepalive_task: self._keepalive_task.cancel()
        logger.info("🔻 Closing WebSocketManager...")
        if self.ws and not self.ws.closed: await self.ws.close()
        logger.info("🔻 WebSocketManager closed cleanly.")