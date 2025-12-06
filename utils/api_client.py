# --- detla-bot/utils/api_client.py ---
# FIX: Kills bot on "IP not whitelisted" error
# FIX: Correctly builds the unencoded query string for signature

import asyncio
import json
import logging
import aiohttp
import urllib.parse
import sys  # ✅ NEW: For hard exit
from aiohttp import client_exceptions
from typing import Optional, Dict, Any, Tuple

from config import (
    DELTA_BASE_URL, 
    USER_AGENT, 
    API_MAX_RETRIES, 
    API_RETRY_DELAY
)
from utils.signing import generate_server_synced_signature

logger = logging.getLogger("api_client")

class DeltaAPIClient:
    """
    A centralized asynchronous client for handling all authenticated
    REST API requests to Delta Exchange.
    """
    
    def __init__(self, session: aiohttp.ClientSession, api_key: str, api_secret: str):
        self.session = session
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = DELTA_BASE_URL
        self.max_retries = API_MAX_RETRIES
        self.retry_delay = API_RETRY_DELAY
        logger.info("✅ DeltaAPIClient initialized.")

    async def _request(
        self, 
        method: str, 
        path: str, 
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Internal method to make a signed, authenticated request with retry logic.
        """
        
        # --- FIX: Build query_string with unencoded commas for signature ---
        query_string_for_sig = ""
        if params:
            # Manually build query string to avoid URL-encoding commas
            # This creates: "?key=val&key2=val,val3"
            query_string_for_sig = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        # --- END FIX ---

        body = json.dumps(payload, separators=(',', ':'), sort_keys=True) if payload else ""
        
        url = f"{self.base_url}{path}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Generate a fresh signature for each attempt
                signature, timestamp = await generate_server_synced_signature(
                    method, 
                    path, 
                    body, 
                    query_string_for_sig, # Pass the unencoded string
                    self.api_key, 
                    self.api_secret
                )
                
                headers = {
                    "api-key": self.api_key,
                    "timestamp": str(timestamp),
                    "signature": signature,
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT
                }

                # aiohttp's 'params' argument WILL URL-encode the request,
                # which is correct for the HTTP request itself.
                async with self.session.request(
                    method, 
                    url, 
                    params=params, # Pass the dict here
                    data=body, 
                    headers=headers
                ) as resp:
                    status = resp.status
                    try:
                        response_json = await resp.json()
                    except aiohttp.ContentTypeError:
                        response_text = await resp.text()
                        logger.error(f"API Request to {path} returned non-JSON response with status {status}. Response: {response_text[:200]}")
                        response_json = {"success": False, "error": f"Non-JSON response: {response_text[:200]}"}

                    if status == 200:
                        return status, response_json
                    
                    if status == 401:
                        # ✅ NEW: Hard Kill Switch for IP Issues
                        error_code = response_json.get("error", {}).get("code")
                        if error_code == "ip_not_whitelisted_for_api_key":
                            logger.critical("🚨 FATAL ERROR: IP Address is NOT whitelisted. Stopping bot immediately to prevent ban/spam.")
                            sys.exit(1)

                        logger.warning(
                            f"Server expected signature data: {response_json.get('error', {}).get('context', {}).get('signature_data')}"
                        )
                    
                    logger.warning(f"API Request to {path} failed with status {status}. Response: {response_json}")
                    
                    if 400 <= status < 500:
                        return status, response_json

            except (client_exceptions.ServerDisconnectedError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    logger.warning(f"Network error on {path} (Attempt {attempt}/{self.max_retries}). Retrying in {self.retry_delay:.1f}s... Error: {type(e).__name__}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ API request to {path} failed after {self.max_retries} attempts. Error: {type(e).__name__}", exc_info=True)
                    return 504, {"success": False, "error": str(e)}
            
            except Exception as e:
                logger.error(f"❌ Unhandled error sending request to {path} on attempt {attempt}: {e}", exc_info=True)
                return 500, {"success": False, "error": str(e)}

        return 503, {"success": False, "error": "Max retries exceeded"}

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """Perform an authenticated GET request."""
        return await self._request("GET", path, params=params, payload=None)

    async def post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """Perform an authenticated POST request."""
        return await self._request("POST", path, params=None, payload=payload)

    async def put(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """Perform an authenticated PUT request."""
        return await self._request("PUT", path, params=None, payload=payload)