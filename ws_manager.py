import asyncio
import json
import logging
import aiohttp
import redis.asyncio as aioredis
from config import WS_URL, RAW_CHANNEL, TRADING_SYMBOLS, USER_AGENT
from utils.signing import generate_ws_keyauth_signature_for_live

logger = logging.getLogger("ws_manager")
logger.setLevel(logging.INFO)


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

        # Reconnect policy
        self.reconnect_delay = 3
        self.reconnect_max = 60
        self.backoff_factor = 2

    async def connect(self):
        """Establish WebSocket connection and subscribe to channels."""
        if self._stop_flag:
            return

        try:
            logger.info(f"🔌 Connecting to WS: {self.ws_url}")
            # Use AIOHTTP's native heartbeat, but rely on Delta's app-level one too
            self.ws = await self.session.ws_connect(self.ws_url, heartbeat=30, headers={'User-Agent': USER_AGENT})
            self.reconnect_delay = 3  # reset delay after successful connect

            logger.info("✅ Connected to Delta WebSocket")
            
            # NEW ROBUSTNESS: Enable application-level heartbeat 
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
            # CRITICAL FIX: Calls the updated function which returns 'key-auth' type and correct signature
            auth_payload = generate_ws_keyauth_signature_for_live()
            await self.ws.send_json(auth_payload)
            logger.info("🔐 Sent authentication payload to WebSocket")
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")

    async def subscribe_public_channels(self):
        """Subscribe to all public market data streams."""
        
        standard_symbols = TRADING_SYMBOLS
        mark_price_symbols = [f"MARK:{symbol}" for symbol in standard_symbols]
        
        candle_resolutions = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        channels_payload = [
            # --- Core Analysis Streams ---
            {"name": "v2/ticker", "symbols": standard_symbols},
            {"name": "l2_updates", "symbols": standard_symbols},
            {"name": "all_trades", "symbols": standard_symbols},
            
            # --- Context & Risk Streams ---
            {"name": "funding_rate", "symbols": standard_symbols},
            {"name": "mark_price", "symbols": mark_price_symbols}
        ]
        
        for res in candle_resolutions:
            channels_payload.append({
                "name": f"candlestick_{res}",
                "symbols": standard_symbols
            })
            
        payload = {
            "type": "subscribe",
            "payload": {
                "channels": channels_payload
            },
        }
        
        try:
            await self.ws.send_str(json.dumps(payload))
            
            subscribed_channels = [ch['name'] for ch in channels_payload]
            logger.info(f"📈 Subscribed to {len(subscribed_channels)} channels for {TRADING_SYMBOLS}")
            logger.debug(f"Subscribed to: {subscribed_channels}")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to public channels: {e}")

    async def subscribe_private_channels(self):
        """Subscribe to private user data channels."""
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
            logger.info("🧠 Subscribed to private channels (orders, positions, v2/user_trades, margins)")
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to private channels: {e}")

    async def start(self):
        """Main message loop — receives and republishes to Redis."""
        await self.connect()
        
        try:
            while not self._stop_flag:
                if self.ws is None or self.ws.closed:
                    logger.warning("WS connection lost, waiting for reconnect...")
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                msg = await self.ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Received malformed WS message: {msg.data[:100]}")
                        continue

                    msg_type = data.get("type", "unknown")

                    if msg_type not in ("subscriptions", "ping", "pong", "heartbeat"):
                        # Log all non-noise messages
                        logger.info(f"🛰️ WS Message: {msg_type}")

                    # CRITICAL FIX: Handle the new 'key-auth' response type 
                    if msg_type == "key-auth":
                        if data.get("success"):
                            self.is_authenticated = True
                            logger.info("✅ Authenticated successfully")
                            await self.subscribe_private_channels()
                        else:
                            logger.error(f"❌ Authentication Failed: {data.get('message', 'Unknown error')}")
                        continue
                    
                    # NEW ROBUSTNESS: Handle Delta's application-level heartbeat message 
                    if msg_type == "heartbeat":
                         continue
                         
                    try:
                        if msg_type not in ("subscriptions", "key-auth"):
                            await self.redis.publish(RAW_CHANNEL, json.dumps(data))
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
            await self.close()

    async def schedule_reconnect(self):
        """Handle reconnection with exponential backoff."""
        if self._stop_flag:
            return
        
        self.is_authenticated = False
        if self.ws and not self.ws.closed:
            await self.ws.close()
            
        logger.warning(f"🔁 Attempting reconnect in {self.reconnect_delay}s...")
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * self.backoff_factor, self.reconnect_max)

        if not self._stop_flag:
            await self.connect()

    async def close(self):
        """Gracefully close all resources."""
        self._stop_flag = True
        logger.info("🔻 Closing WebSocketManager...")
        if self.ws and not self.ws.closed:
            await self.ws.close()
        logger.info("🔻 WebSocketManager closed cleanly.")