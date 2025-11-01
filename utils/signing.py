# signing.py - UPDATED

import hmac
import hashlib
import time
import logging
import urllib.parse
from config import API_KEY, API_SECRET

logger = logging.getLogger("signing")
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# ✅ Use local system time (Delta India doesn't expose /time endpoint)
# ----------------------------------------------------------------------
async def get_server_time():
    """Return Unix timestamp in seconds (int)."""
    return int(time.time())

# ----------------------------------------------------------------------
# ✅ CORRECTED REST HMAC Signature Generator (India version)
# ----------------------------------------------------------------------
async def generate_server_synced_signature(method: str, path: str, body: str = "", query_params: str = ""):
    """
    Generate HMAC-SHA256 signature for Delta India REST API.

    ✅ FIXED: Now includes query parameters in signature calculation
    """
    timestamp = int(time.time())

    # The signature message includes Method, Timestamp, Path, Query Params, and Body [cite: 89]
    if query_params:
        message = f"{method.upper()}{timestamp}{path}?{query_params}{body}"
    else:
        message = f"{method.upper()}{timestamp}{path}{body}"

    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    logger.info(f"🧮 Signature base: {message}")
    logger.info(f"🧾 Computed signature: {signature}")
    return signature, timestamp

# ----------------------------------------------------------------------
# ✅ WS Authentication payload (for private WS channels) - CRITICAL FIX
# ----------------------------------------------------------------------
def generate_ws_keyauth_signature_for_live():
    """
    WebSocket authentication payload builder for the new 'key-auth' method.
    
    CRITICAL FIX: Uses the correct message format and timestamp in seconds.
    """
    # Timestamp must be in seconds (int) [cite: 540]
    timestamp = int(time.time()) 
    
    # Signature message format: 'GET' + string(TIMESTAMP) + '/live' [cite: 541]
    method = 'GET'
    path = '/live'
    signature_data = f"{method}{timestamp}{path}"
    
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        signature_data.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    return {
        # Payload uses 'key-auth' type and 'api-key' key [cite: 539]
        "type": "key-auth", 
        "payload": {
            "api-key": API_KEY, 
            "signature": signature,
            "timestamp": timestamp
        }
    }