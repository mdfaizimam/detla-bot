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

    # ✅ FIX: Include query parameters in signature message
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
# ✅ WS Authentication payload (for private WS channels)
# ----------------------------------------------------------------------
def generate_ws_keyauth_signature_for_live():
    """WebSocket authentication payload builder."""
    timestamp = int(time.time() * 1000)  # ms
    message = f"{timestamp}{API_KEY}"
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return {
        "type": "auth",
        "payload": {
            "api_key": API_KEY,
            "signature": signature,
            "timestamp": timestamp
        }
    }
