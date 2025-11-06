# --- utils/signing.py ---
# FIX: Merged working REST signature with the correct WS signature.

import hmac
import hashlib
import time
import logging
from typing import Tuple

logger = logging.getLogger("signing")

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
    timestamp = str(int(time.time()))
    
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
    timestamp = int(time.time()) 
    
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