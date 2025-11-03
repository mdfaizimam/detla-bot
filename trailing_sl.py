import asyncio
import aiohttp
import json
import os
import logging
import hashlib
import hmac
import time
from typing import Tuple
from dotenv import load_dotenv

# --- Configuration and Utilities ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("LIVE_PRICE_FETCHER")
load_dotenv()
# Ensure these environment variables are correctly set in your .env file!
API_KEY = os.getenv("DELTA_API_KEY", "")
API_SECRET = os.getenv("DELTA_API_SECRET", "")
DELTA_BASE_URL = "https://api.india.delta.exchange"
USER_AGENT = "DeltaInstitutionalBot/1.0"
TEST_PRODUCT_ID = 14823 
TEST_SYMBOL = "SOLUSD" 

# --- Signing Functions (Fixed according to Delta Exchange API docs) ---
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
    
    # According to Delta docs: method + timestamp + requestPath + query params + body
    signature_data = method + timestamp + path + query_string + body
    logger.debug(f"Signature data: {signature_data}")
    
    signature = _generate_sha256_signature(api_secret, signature_data)
    return signature, timestamp

async def _make_authenticated_get_request(
    session: aiohttp.ClientSession, path: str, params: dict
):
    """Make authenticated GET request."""
    # Build query string for signature
    query_string = ""
    if params:
        query_string = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    
    signature, timestamp = await generate_signature(
        "GET", path, query_string, "", API_SECRET
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

async def _make_authenticated_put_request(
    session: aiohttp.ClientSession, 
    path: str, 
    data: dict
):
    """Make authenticated PUT request."""
    # For PUT/POST requests, query_string is empty and body is included in signature
    query_string = ""
    
    # Convert data to JSON string with consistent formatting
    body = json.dumps(data, separators=(',', ':'), sort_keys=True)
    
    signature, timestamp = await generate_signature(
        "PUT", path, query_string, body, API_SECRET
    )
    
    headers = {
        "api-key": API_KEY,
        "timestamp": timestamp,
        "signature": signature,
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    url = f"{DELTA_BASE_URL}{path}"
    
    logger.debug(f"PUT Request URL: {url}")
    logger.debug(f"PUT Request Headers: {headers}")
    logger.debug(f"PUT Request Body: {body}")
    
    async with session.put(url, headers=headers, data=body) as resp:
        status = resp.status
        response_text = await resp.text()
        return status, json.loads(response_text) if response_text else None

# --- Application Functions ---

async def fetch_ticker_data(session: aiohttp.ClientSession, symbol: str) -> str:
    """Fetches the live Mark Price for a specific product symbol (Unauthenticated)."""
    path = f"/v2/tickers/{symbol}"
    url = f"{DELTA_BASE_URL}{path}"
    
    headers = {'Accept': 'application/json', 'User-Agent': USER_AGENT} 
    
    logger.info(f"Fetching live ticker for {symbol}...")
    async with session.get(url, headers=headers) as resp:
        status = resp.status
        response_data = await resp.json()
        
        if status == 200 and response_data.get('success'):
            mark_price = response_data.get("result", {}).get("mark_price", "N/A")
            logger.info(f"✅ Live Mark Price for {symbol}: {mark_price}")
            return mark_price
        else:
            logger.error(f"❌ Failed to fetch ticker (HTTP {status}) for {symbol}.")
            logger.error(f"Response: {response_data}")
            return "N/A"

async def fetch_open_position(session: aiohttp.ClientSession, product_id: int):
    """Fetches real-time position data for a product."""
    path = "/v2/positions"
    params = {"product_id": str(product_id)}
    
    logger.info(f"Fetching open position for Product ID {product_id}...")
    status, data = await _make_authenticated_get_request(session, path, params)
    
    if status == 200 and data and data.get('success'):
        position_data = data.get('result', {})
        size = position_data.get('size', 0)
        symbol = position_data.get('product_symbol', TEST_SYMBOL)
        entry_price = position_data.get('entry_price', '0.0')
        
        if size != 0:
            logger.info(f"✅ Open Position found: Size {size}, Entry Price {entry_price}")
            return symbol, int(size), float(entry_price)
        else:
            logger.warning(f"No active position (Size=0) found for Product ID {product_id}.")
            return symbol, 0, 0.0
    else:
        logger.error(f"❌ Failed to fetch position (HTTP {status}). Response: {data}")
        return TEST_SYMBOL, 0, 0.0

async def fetch_open_stop_orders(session: aiohttp.ClientSession, product_id: int):
    """Fetches open stop orders for a specific product."""
    path = "/v2/orders"
    params = {
        "product_ids": str(product_id), 
        "states": "open,pending", 
        "order_types": "stop_limit,stop_market", 
    }
    
    logger.info(f"Fetching open stop orders for Product ID {product_id}...")
    status, data = await _make_authenticated_get_request(session, path, params)
    
    if status == 200 and data and data.get('success'):
        stop_orders = [
            order for order in data.get("result", []) 
            if (order.get("stop_order_type") == "stop_loss_order" or order.get("order_type") in ["stop_limit", "stop_market"])
            and order.get("state") in ["open", "pending"]
        ]
        
        if stop_orders:
            logger.info(f"✅ Found {len(stop_orders)} open stop orders.")
            for order in stop_orders:
                logger.info(f"-> ID: {order['id']}, Side: {order['side']}, Stop Price: {order.get('stop_price')}, Trail Amount: {order.get('trail_amount')}")
            return stop_orders
        else:
            logger.warning("No open Stop orders found.")
            return []
    else:
        logger.error(f"❌ Failed to fetch stop orders (HTTP {status}). Response: {data}")
        return []

async def update_trailing_stop(
    session: aiohttp.ClientSession, 
    order_id: int, 
    product_id: int, 
    size: int, 
    new_stop_price: float, 
    trail_amount: float
):
    """Edits an existing stop order to set a new trailing stop price."""
    
    path = "/v2/orders"
    
    # Prepare request data according to Delta Exchange API format
    request_data = {
        "id": order_id,
        "product_id": product_id,
        "size": abs(size),  # Size should be positive
        "stop_price": f"{new_stop_price:.4f}",
        "trail_amount": f"{trail_amount:.4f}",
        "reduce_only": True
    }
    
    logger.info(f"Attempting PUT update for Stop Order {order_id} to price {new_stop_price:.4f}...")
    
    status, response_data = await _make_authenticated_put_request(session, path, request_data)
    
    if status == 200 and response_data.get('success'):
        logger.info(f"✅ Stop order {order_id} updated successfully!")
        return True
    else:
        logger.error(f"❌ Failed to update stop order (HTTP {status}).")
        if response_data and 'error' in response_data:
            error_info = response_data['error']
            logger.error(f"Error code: {error_info.get('code')}")
            if 'context' in error_info and 'signature_data' in error_info['context']:
                logger.error(f"Server expected signature data: {error_info['context']['signature_data']}")
        logger.error(f"Full response: {response_data}")
        return False

# --- Main Trailing Logic ---

async def main_loop():
    """Main function simulating the continuous trailing stop logic."""
    if not API_KEY or not API_SECRET or API_KEY == "YOUR_API_KEY_HERE":
        logger.error("❌ API_KEY or API_SECRET not set in environment.")
        return
        
    # --- Trailing Stop Configuration ---
    TRAIL_AMOUNT = 2.00  # The fixed distance in USD
    # Tracks the most profitable price seen so far for a short position (the lowest price)
    lowest_mark_price_seen = float('inf') 
    # --- --------------------------- ---

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # 1. Fetch live position data
                symbol, size, entry_price = await fetch_open_position(session, TEST_PRODUCT_ID)
                
                if size == 0:
                    logger.info("No active position found. Exiting continuous loop.")
                    break

                # 2. Fetch Live Price 
                logger.info("-" * 50)
                live_mark_price_str = await fetch_ticker_data(session, symbol)
                live_mark_price = float(live_mark_price_str) if live_mark_price_str != 'N/A' else None
                logger.info("-" * 50)
                
                if live_mark_price is None:
                    logger.warning("Skipping price check iteration.")
                    await asyncio.sleep(5)
                    continue

                # --- Core Trailing Logic for a SHORT Position ---
                
                # A. Update the lowest_mark_price_seen (the lowest price reached)
                if live_mark_price < lowest_mark_price_seen:
                    lowest_mark_price_seen = live_mark_price
                    logger.info(f"New Lowest Mark Price tracked: {lowest_mark_price_seen:.4f}")
                elif lowest_mark_price_seen != float('inf'):
                    logger.info(f"Current lowest mark tracked: {lowest_mark_price_seen:.4f}")

                # B. Calculate the required Trailing Stop Price
                # For a SHORT position: Stop Price = Lowest Mark Price + Trail Amount
                required_stop_price = lowest_mark_price_seen + TRAIL_AMOUNT
                
                logger.info(f"Calculated Required Stop Price: {required_stop_price:.4f} (Lowest: {lowest_mark_price_seen:.4f} + Trail: {TRAIL_AMOUNT:.2f})")
                
                # C. Fetch Existing Stop Orders
                stop_orders = await fetch_open_stop_orders(session, TEST_PRODUCT_ID)
                
                if stop_orders:
                    current_stop_order = stop_orders[0]
                    # Convert stop_price from string in API response to float
                    current_stop_price = float(current_stop_order.get('stop_price', 0))
                    order_id = current_stop_order['id']

                    # D. Check for update condition: move the stop price DOWN only if the market moved in our favor
                    # Use a small tolerance to prevent unnecessary updates due to tiny floating point changes
                    if required_stop_price < current_stop_price - 0.0001: 
                        
                        logger.info(f"Condition met: Moving SL from {current_stop_price:.4f} DOWN to {required_stop_price:.4f}")
                        
                        success = await update_trailing_stop(
                            session, 
                            order_id, 
                            TEST_PRODUCT_ID, 
                            size, 
                            required_stop_price, 
                            TRAIL_AMOUNT
                        )
                        
                        if success:
                            # Reset the tracking after successful update to avoid repeated updates
                            lowest_mark_price_seen = live_mark_price
                    else:
                        logger.info(f"Stop Price does not require an update. (Current SL: {current_stop_price:.4f} / Required SL: {required_stop_price:.4f})")

                else:
                    logger.warning("No existing stop order found. Logic to place a NEW trailing stop would go here.")
                
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            
            # Wait 5 seconds before the next loop iteration
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        print(f"Starting Trailing SL Bot for {TEST_SYMBOL}...")
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)