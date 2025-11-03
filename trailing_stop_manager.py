# --- trailing_stop_manager.py ---

import asyncio
import aiohttp
import json
import logging
import time
import hashlib
import hmac
import urllib.parse
from typing import Optional, Dict, Any, Tuple
from redis import asyncio as aioredis

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    TSL_CHANNEL,
    MONITORING_CHANNEL,
    USER_AGENT,
    config
)
# NOTE: We have removed the import of generate_server_synced_signature 
# and replaced it with dedicated, Delta-compliant signing logic below.

logger = logging.getLogger("tsl_manager")
logger.setLevel(logging.INFO)

# --- Delta Exchange Compliant Signing Functions (From User's Working Code) ---

def _generate_sha256_signature(secret: str, message: str) -> str:
    """Generate HMAC SHA256 signature."""
    message_bytes = bytes(message, 'utf-8')
    secret_bytes = bytes(secret, 'utf-8')
    hash_obj = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
    return hash_obj.hexdigest()

async def generate_signature(
    method: str, 
    path: str, 
    query_string: str,
    body: str,
    api_secret: str,
) -> Tuple[str, str]:
    """
    Generate signature according to Delta Exchange API documentation.
    Signature = method + timestamp + path + query_string + body
    """
    timestamp = str(int(time.time()))
    
    # CRITICAL: Signature data must include method + timestamp + path + query params + body
    # This construction mirrors the logic in the user's working script.
    signature_data = method + timestamp + path + query_string + body
    logger.debug(f"TSL Sig Data: {signature_data}")
    
    signature = _generate_sha256_signature(api_secret, signature_data)
    return signature, timestamp

async def _make_authenticated_get_request(
    session: aiohttp.ClientSession, path: str, params: dict, api_key: str, api_secret: str
) -> Tuple[int, Optional[Dict]]:
    """Make authenticated GET request using correct signature building."""
    
    # 1. Build query string for signature (Note: this is raw param joining, NOT URL encoding)
    query_string = ""
    if params:
        # Sort parameters alphabetically and join key=value pairs with '&'
        # The URL encoding happens naturally in the HTTP client call, but NOT for the signature.
        sorted_params = sorted(params.items())
        query_string = "?" + "&".join([f"{k}={v}" for k, v in sorted_params])

    # 2. Generate signature
    signature, timestamp = await generate_signature(
        "GET", path, query_string, "", api_secret
    )
    
    # 3. Prepare headers
    headers = {
        "api-key": api_key,
        "timestamp": timestamp,
        "signature": signature,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    }
    
    url = f"{DELTA_BASE_URL}{path}"
    
    # 4. Make the request (params argument here handles the actual URL encoding for the request)
    async with session.get(url, headers=headers, params=params) as resp:
        status = resp.status
        response_text = await resp.text()
        return status, json.loads(response_text) if response_text else None

async def _make_authenticated_put_request(
    session: aiohttp.ClientSession, 
    path: str, 
    data: dict,
    api_key: str,
    api_secret: str
) -> Tuple[int, Optional[Dict]]:
    """Make authenticated PUT request using correct signature building."""
    query_string = ""
    
    # CRITICAL: Convert data to JSON string with consistent formatting (sorted keys, no spaces)
    body = json.dumps(data, separators=(',', ':'), sort_keys=True)
    
    signature, timestamp = await generate_signature(
        "PUT", path, query_string, body, api_secret
    )
    
    headers = {
        "api-key": api_key,
        "timestamp": timestamp,
        "signature": signature,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    url = f"{DELTA_BASE_URL}{path}"
    
    async with session.put(url, headers=headers, data=body) as resp:
        status = resp.status
        response_text = await resp.text()
        return status, json.loads(response_text) if response_text else None

# --- TrailingStopManager Class ---

class TrailingStopManager:
    """
    Monitors an active position and automatically trails the stop-loss order
    using dedicated, Delta-compliant API access methods.
    """
    
    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession):
        self.redis = redis_client
        self.session = http_session
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.tsl_tasks: Dict[int, asyncio.Task] = {} 
        self.tsl_config = {
            "trail_amount": config["TSL_TRAIL_AMOUNT"],
            "check_interval": config["TSL_CHECK_INTERVAL"]
        }
        
        if not config["TSL_ENABLED"]:
            logger.warning("🚫 TSL Manager is initialized but TSL_ENABLED is False. It will not run.")

    # --- API Helper Functions using local authenticated methods ---
    
    async def fetch_ticker_data(self, symbol: str) -> Optional[float]:
        """Fetches the live Mark Price for a specific product symbol (Unauthenticated)."""
        path = f"/v2/tickers/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"
        
        headers = {'Accept': 'application/json', 'User-Agent': USER_AGENT} 
        
        try:
            async with self.session.get(url, headers=headers) as resp:
                response_data = await resp.json()
                if resp.status == 200 and response_data.get('success'):
                    mark_price = response_data.get("result", {}).get("mark_price")
                    if mark_price:
                        return float(mark_price)
                
                logger.error(f"❌ Failed to fetch ticker (HTTP {resp.status}) for {symbol}.")
                return None
        except Exception as e:
            logger.error(f"❌ Error fetching ticker data: {e}", exc_info=True)
            return None


    async def fetch_open_stop_order_id(self, product_id: int) -> Optional[int]:
        """Fetches the ID of the open stop-loss order for a given product."""
        path = "/v2/orders"
        # CRITICAL FIX: Ensure params uses raw strings, not URL-encoded values
        params = {
            "product_ids": str(product_id), 
            "states": "open,pending", 
            "stop_order_type": "stop_loss_order" 
        }
        
        status, data = await _make_authenticated_get_request(
            self.session, path, params, self.api_key, self.api_secret
        )
        
        if status == 200 and data and data.get('success'):
            stop_orders = data.get("result", [])
            
            for order in stop_orders:
                if order.get("stop_order_type") == "stop_loss_order":
                    logger.debug(f"Found existing SL Order ID: {order['id']} @ {order.get('stop_price')}")
                    return order['id']
            
            logger.warning(f"No open Stop-Loss order found for Product ID {product_id}. Waiting for one to appear.")
            return None
        else:
            logger.error(f"❌ Failed to fetch stop orders (HTTP {status}). Response: {data}")
            return None

    async def update_stop_price(
        self, 
        order_id: int, 
        product_id: int, 
        size: int, 
        new_stop_price: float, 
    ) -> bool:
        """Edits an existing stop order with the new trailing stop price."""
        
        path = "/v2/orders"
        
        # Determine side based on position size
        side = "buy" if size < 0 else "sell"
        
        request_data = {
            "id": order_id,
            "product_id": product_id,
            "size": abs(size), 
            "side": side, 
            "stop_price": f"{new_stop_price:.4f}",
            "order_type": "stop_limit", 
            "limit_price": f"{new_stop_price:.4f}", 
            "reduce_only": True
        }
        
        logger.info(f"Attempting PUT update for Stop Order {order_id} to price {new_stop_price:.4f}...")
        
        status, response_data = await _make_authenticated_put_request(
            self.session, path, request_data, self.api_key, self.api_secret
        )
        
        if status == 200 and response_data and response_data.get('success'):
            logger.info(f"✅ Stop order {order_id} updated successfully to {new_stop_price:.4f}!")
            return True
        else:
            logger.error(f"❌ Failed to update stop order (HTTP {status}). Response: {response_data}")
            return False

    # --- Core Trailing Logic (Unchanged from original structure) ---

    async def _trailing_loop(self, product_id: int, symbol: str, direction: str, size: int):
        """Continuous trailing stop logic for a single position."""
        
        trail_amount = self.tsl_config["trail_amount"]
        check_interval = self.tsl_config["check_interval"]
        
        best_price_seen: float = float('-inf') if direction == "LONG" else float('inf')
        stop_order_id: Optional[int] = None
        
        # 1. Wait for the SL order to appear in the exchange system
        wait_attempts = 0
        max_wait_attempts = 10
        while stop_order_id is None and wait_attempts < max_wait_attempts:
            logger.info(f"TSL Manager: Waiting for initial SL order for {symbol} (Attempt {wait_attempts+1}/{max_wait_attempts})...")
            # This call now uses the fixed signing logic
            stop_order_id = await self.fetch_open_stop_order_id(product_id)
            if stop_order_id is None:
                wait_attempts += 1
                await asyncio.sleep(1.0) 
        
        if stop_order_id is None:
            logger.error(f"❌ TSL Manager failed to find initial SL order for {symbol}. TSL disabled for this position.")
            return

        logger.info(f"TSL Manager activated for {symbol} ({direction}) | Order ID: {stop_order_id}")
        
        while True:
            try:
                # 2. Fetch Live Price 
                live_mark_price = await self.fetch_ticker_data(symbol)
                
                if live_mark_price is None:
                    await asyncio.sleep(check_interval)
                    continue

                # 3. Update the best price seen (most favorable price)
                is_new_best = False
                if direction == "LONG" and live_mark_price > best_price_seen:
                    best_price_seen = live_mark_price
                    is_new_best = True
                elif direction == "SHORT" and live_mark_price < best_price_seen:
                    best_price_seen = live_mark_price
                    is_new_best = True
                
                if is_new_best:
                    logger.info(f"New Best Mark Price tracked for {symbol}: {best_price_seen:.4f}")

                # 4. Calculate the required Trailing Stop Price
                if direction == "LONG":
                    required_stop_price = best_price_seen - trail_amount
                else: # SHORT
                    required_stop_price = best_price_seen + trail_amount
                
                # 5. Fetch the current stop price of the active order for comparison
                # Note: This GET call is simplified, as we only need the current stop price.
                # However, for consistency, we'll keep fetching the order list.
                path = "/v2/orders"
                params = {"order_ids": str(stop_order_id)}
                
                status, order_details = await _make_authenticated_get_request(
                    self.session, path, params, self.api_key, self.api_secret
                )
                
                current_stop_price = None
                if status == 200 and order_details and order_details.get('result'):
                    order_result = order_details['result']
                    if order_result:
                        current_stop_price = float(order_result[0].get('stop_price', 0))
                
                if current_stop_price is None or current_stop_price == 0.0:
                    logger.warning(f"Could not retrieve current stop price for order ID {stop_order_id}. Skipping update.")
                    await asyncio.sleep(check_interval)
                    continue
                    
                # 6. Check if update is required
                
                tolerance = 0.0001 
                update_required = False
                
                if direction == "LONG" and required_stop_price > current_stop_price + tolerance:
                    update_required = True
                elif direction == "SHORT" and required_stop_price < current_stop_price - tolerance:
                    update_required = True
                
                if update_required:
                    logger.info(f"Condition met: Moving SL for {symbol} from {current_stop_price:.4f} "
                                f"to {required_stop_price:.4f} (Best Price: {best_price_seen:.4f})")
                                
                    await self.update_stop_price(
                        stop_order_id, 
                        product_id, 
                        size, 
                        required_stop_price
                    )
                else:
                    logger.debug(f"Stop Price for {symbol} does not require an update. (Current SL: {current_stop_price:.4f} / Required SL: {required_stop_price:.4f})")

                # Wait for the next check
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in TSL loop for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(check_interval * 2)

    # --- Manager Core (Unchanged) ---
    
    async def _handle_tsl_control_message(self, data: Dict[str, Any]):
        """Starts or stops TSL tracking based on messages from Executor or Monitor."""
        message_type = data.get("type")
        symbol = data.get("symbol")
        product_id = data.get("product_id")
        
        if not product_id: return
        
        if message_type == "start_tsl" and config["TSL_ENABLED"]:
            direction = data.get("direction")
            size = data.get("size")
            
            if product_id in self.tsl_tasks: return

            logger.info(f"🎯 Starting TSL for {symbol} ({direction}, ID: {product_id})")
            
            task = asyncio.create_task(
                self._trailing_loop(product_id, symbol, direction, size),
                name=f"TSL_Loop_{symbol}_{product_id}"
            )
            self.tsl_tasks[product_id] = task

        elif message_type == "position_closed":
            if product_id in self.tsl_tasks:
                logger.info(f"🛑 Position closed for {symbol}. Stopping TSL task.")
                task = self.tsl_tasks.pop(product_id)
                if not task.done():
                    task.cancel()
            
    
    async def start(self):
        """Main entry point for the TSL Manager service."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(TSL_CHANNEL, MONITORING_CHANNEL)
        logger.info(f"🚀 TrailingStopManager started, listening to {TSL_CHANNEL} and {MONITORING_CHANNEL}")

        try:
            async for raw in pubsub.listen():
                if raw is None or raw.get("type") != "message":
                    continue
                
                channel = raw['channel']
                data_str = raw['data']
                
                try:
                    data = json.loads(data_str)
                    
                    if channel == TSL_CHANNEL:
                        await self._handle_tsl_control_message(data)
                    
                    elif channel == MONITORING_CHANNEL and data.get("type") == "position_closed":
                        closed_symbol = data.get("symbol")
                        inferred_pid = next((pid for pid, task in self.tsl_tasks.items() if task.get_name().split('_')[2] == closed_symbol), None)
                        
                        if inferred_pid:
                            data["product_id"] = inferred_pid
                            await self._handle_tsl_control_message(data)
                        
                except Exception as e:
                    logger.error(f"Error processing message from {channel}: {e}")
                    
        except asyncio.CancelledError:
            logger.info("TrailingStopManager cancelled.")
        except Exception as e:
            logger.error(f"💥 TrailingStopManager crashed: {e}", exc_info=True)
        finally:
            logger.info("🔻 TrailingStopManager stopped cleanly.")
            for task in self.tsl_tasks.values():
                if not task.done():
                    task.cancel()
