# --- utils/signing.py ---
# FIX: Merged working REST signature with the correct WS signature.
# ✅ FIX: Added server time synchronization to prevent clock drift errors.

import hmac
import hashlib
import time
import logging
import aiohttp
from typing import Tuple
from email.utils import parsedate_to_datetime

from config import DELTA_BASE_URL, USER_AGENT

logger = logging.getLogger("signing")

# ----------------------------------------------------------------------
# ✅ NEW: Server Time Synchronization
# ----------------------------------------------------------------------
_time_offset: float = 0.0
_offset_last_synced: float = 0.0

async def sync_time_offset(http_session: aiohttp.ClientSession):
    """
    Fetches server time from an unauthenticated endpoint to calculate
    the offset between local time and server time.
    """
    global _time_offset, _offset_last_synced
    try:
        # Use a lightweight, unauthenticated endpoint
        url = f"{DELTA_BASE_URL}/v2/assets"
        headers = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
        
        async with http_session.get(url, headers=headers) as resp:
            if resp.status == 200:
                date_header = resp.headers.get("Date")
                if date_header:
                    server_time_dt = parsedate_to_datetime(date_header)
                    server_time_unix = server_time_dt.timestamp()
                    local_time_unix = time.time()
                    
                    _time_offset = server_time_unix - local_time_unix
                    _offset_last_synced = local_time_unix
                    
                    logger.info(f"✅ Server time offset synced: {_time_offset:+.2f} seconds.")
                else:
                    logger.warning("Could not sync time: 'Date' header missing from response.")
            else:
                logger.warning(f"Could not sync time: API request failed with status {resp.status}")
                
    except Exception as e:
        logger.error(f"❌ Failed to sync server time: {e}", exc_info=True)

def get_synced_time() -> int:
    """Returns the current Unix time (in seconds) adjusted by the server offset."""
    if (time.time() - _offset_last_synced) > 6 * 3600:
        logger.warning("Time offset is stale. Re-sync should be triggered.")
    return int(time.time() + _time_offset)
# ----------------------------------------------------------------------

def _generate_sha256_signature(secret: str, message: str) -> str:
    """Generate HMAC SHA256 signature."""
    message_bytes = bytes(message, 'utf-8')
    secret_bytes = bytes(secret, 'utf-8')
    hash_obj = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
    return hash_obj.hexdigest()

# ----------------------------------------------------------------------
# ✅ CORRECTED REST API Signature (for TSL, etc.)
# ----------------------------------------------------------------------
def _build_rest_signature_base(
    method: str, 
    timestamp: str, 
    request_path: str, 
    query_string: str, 
    body: str
) -> str:
    """
    Helper to construct the signature base for REST API.
    Signature = method + timestamp + path + query_string + body
    (e.g., GET<ts>/v2/orders?states=open,pending)
    """
    base = method.upper() + timestamp + request_path
    if query_string:
        # query_string includes the '?' and is NOT URL-encoded
        base += query_string  
    if body:
        base += body
    
    return base

async def generate_server_synced_signature(
    method: str,
    request_path: str,
    body: str,
    query_string: str, # This is the unencoded query string (e.g., "?states=open,pending")
    api_key: str,
    api_secret: str,
) -> Tuple[str, str]:
    """
    Generates a signature for REST API calls using local system time.
    """
    # ✅ FIX: Use synced time
    timestamp = str(get_synced_time())
    
    base_string = _build_rest_signature_base(
        method, timestamp, request_path, query_string, body
    )
    logger.debug(f"🧮 REST Signature base: {base_string}")
    
    signature = _generate_sha256_signature(api_secret, base_string)
    logger.debug(f"🧾 REST Computed signature: {signature}")
    
    return signature, timestamp

# ----------------------------------------------------------------------
# ✅ CORRECTED WebSocket API Signature (for Authentication)
# ----------------------------------------------------------------------
def generate_ws_keyauth_signature_for_live(api_key: str, api_secret: str):
    """
    WebSocket authentication payload builder.
    Signature base format: 'GET' + string(TIMESTAMP) + '/live'
    """
    # ✅ FIX: Use synced time
    timestamp = get_synced_time() 
    
    method = 'GET'
    path = '/live' # <-- This was the critical missing piece
    signature_data = f"{method}{timestamp}{path}"
    
    signature = hmac.new(
        api_secret.encode("utf-8"),
        signature_data.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    logger.debug(f"🧮 WS Auth Signature base: {signature_data}")
    logger.debug(f"🧾 WS Auth Computed signature: {signature}")
    
    return {
        "type": "key-auth", 
        "payload": {
            "api-key": api_key, 
            "signature": signature,
            "timestamp": timestamp
        }
    }