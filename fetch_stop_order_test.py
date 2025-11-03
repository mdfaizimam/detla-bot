import asyncio
import aiohttp
import json
import os
import logging
import hashlib
import hmac
from typing import Tuple
from dotenv import load_dotenv

# --- Configuration and Utilities (Keeping previous setup for reference) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("LIVE_PRICE_FETCHER")
load_dotenv()
API_KEY = os.getenv("DELTA_API_KEY", "YOUR_API_KEY_HERE")
API_SECRET = os.getenv("DELTA_API_SECRET", "YOUR_API_SECRET_HERE")
DELTA_BASE_URL = "https://api.india.delta.exchange"
USER_AGENT = "DeltaInstitutionalBot/1.0"
TEST_PRODUCT_ID = 14823 
TEST_SYMBOL = "SOLUSD" # We now use this symbol to fetch the ticker

# --- Placeholder for Signing Functions ---
def _generate_sha256_signature(secret: str, message: str) -> str:
    # ... (Implementation as before) ...
    message_bytes = bytes(message, 'utf-8')
    secret_bytes = bytes(secret, 'utf-8')
    hash_obj = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
    return hash_obj.hexdigest()

async def generate_server_synced_signature(
    method: str, 
    path: str, 
    body: str, 
    query_string: str,
    api_secret: str,
) -> Tuple[str, str]:
    import time
    timestamp = str(int(time.time()))
    # IMPORTANT: The signature string format is strictly defined by the API
    signature_data = method + timestamp + path + '?' + query_string + body
    signature = _generate_sha256_signature(api_secret, signature_data)
    return signature, timestamp

async def _make_authenticated_get_request(
    session: aiohttp.ClientSession, path: str, params: dict, raw_query_keys: list
):
    # ... (Implementation as before for authenticated calls) ...
    param_list = [f"{k}={params[k]}" for k in raw_query_keys]
    raw_query_string = "&".join(param_list) 
    
    signature, timestamp = await generate_server_synced_signature(
        "GET", path, "", raw_query_string, API_SECRET
    )
    
    headers = {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    }
    
    url = f"{DELTA_BASE_URL}{path}"
    
    async with session.get(url, headers=headers, params=params) as resp:
        status = resp.status
        response_text = await resp.text()
        return status, json.loads(response_text) if response_text else None
# --- End of Authentication/Utility Functions ---


## 🚀 Live Price and Position Fetch Functions

async def fetch_ticker_data(session: aiohttp.ClientSession, symbol: str) -> str:
    """
    Fetches the live Mark Price for a specific product symbol (Unauthenticated).
    Endpoint: GET /v2/tickers/{symbol}
    """
    # Use the /v2/tickers/{symbol} endpoint directly
    path = f"/v2/tickers/{symbol}"
    url = f"{DELTA_BASE_URL}{path}"
    
    # This is an unauthenticated, public endpoint
    headers = {'Accept': 'application/json', 'User-Agent': USER_AGENT} 
    
    logger.info(f"Fetching live ticker for {symbol}...")
    async with session.get(url, headers=headers) as resp:
        status = resp.status
        response_data = await resp.json()
        
        if status == 200 and response_data.get('success'):
            # The Mark Price is in the 'mark_price' field of the result object
            mark_price = response_data.get("result", {}).get("mark_price", "N/A")
            logger.info(f"✅ Live Mark Price for {symbol}: {mark_price}")
            return mark_price
        else:
            logger.error(f"❌ Failed to fetch ticker (HTTP {status}) for {symbol}.")
            logger.error(f"Response: {response_data}")
            return "N/A"

async def fetch_open_position(session: aiohttp.ClientSession, product_id: int):
    """
    Fetches real-time position data (size, entry_price, and symbol) for a product.
    Endpoint: GET /v2/positions
    """
    path = "/v2/positions"
    # The API documentation specifies this endpoint returns real-time data for size and entry price.
    params = {"product_id": str(product_id)}
    raw_query_keys = ['product_id'] 
    
    logger.info(f"Fetching open position for Product ID {product_id}...")
    status, data = await _make_authenticated_get_request(session, path, params, raw_query_keys)
    
    if status == 200 and data and data.get('success'):
        position_data = data.get('result', {})
        size = position_data.get('size', 0)
        symbol = position_data.get('product_symbol', TEST_SYMBOL) # Fallback to TEST_SYMBOL
        
        if size != 0:
            logger.info(f"✅ Open Position found: Size {size}, Entry Price {position_data.get('entry_price')}")
            return symbol, size
        else:
            logger.warning(f"No active position (Size=0) found for Product ID {product_id}. Using fallback symbol: {symbol}")
            return symbol, size
    else:
        logger.error(f"❌ Failed to fetch position (HTTP {status}). Response: {data}")
        return TEST_SYMBOL, 0 # Return fallback symbol and size 0 on failure

async def fetch_open_stop_orders(session: aiohttp.ClientSession, product_id: int):
    """Fetches open stop orders for a specific product."""
    path = "/v2/orders"
    params = {
        "product_ids": str(product_id), 
        "states": "open,pending", 
        "order_types": "stop_limit,stop_market", 
    } 
    raw_query_keys = ['product_ids', 'states', 'order_types'] 
    
    logger.info(f"Fetching open stop orders for Product ID {product_id}...")
    status, data = await _make_authenticated_get_request(session, path, params, raw_query_keys)
    
    if status == 200 and data and data.get('success'):
        stop_orders = [
            order for order in data.get("result", []) 
            if order.get("stop_order_type") == "stop_loss_order" or order.get("order_type") in ["stop_limit", "stop_market"]
        ]
        
        if stop_orders:
            logger.info(f"✅ Found {len(stop_orders)} open stop orders.")
            for order in stop_orders:
                logger.info(f"-> ID: {order['id']}, Side: {order['side']}, Stop Price: {order.get('stop_price')}, Trail Amount: {order.get('trail_amount')}")
        else:
            logger.warning("No open Stop orders found.")
        
    else:
        logger.error(f"❌ Failed to fetch stop orders (HTTP {status}). Response: {data}")


async def main():
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        logger.error("❌ API_KEY or API_SECRET not set in environment. Please configure .env file.")
        return

    async with aiohttp.ClientSession() as session:
        # 1. Fetch Open Position to get the symbol
        symbol, size = await fetch_open_position(session, TEST_PRODUCT_ID)
        
        if symbol:
            logger.info("-" * 50)
            # 2. Fetch Live Price (Ticker) using the symbol
            live_price = await fetch_ticker_data(session, symbol)
            logger.info("-" * 50)
            
            # 3. Fetch Open Stop Orders
            await fetch_open_stop_orders(session, TEST_PRODUCT_ID)

if __name__ == "__main__":
    try:
        print(f"Testing fetch for Product ID {TEST_PRODUCT_ID} ({TEST_SYMBOL})...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except RuntimeError as e:
        if "Event loop stopped" not in str(e):
            raise e