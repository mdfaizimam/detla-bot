# --- trailing_stop_manager.py ---
# UPDATED: Refactored to use centralized DeltaAPIClient
# FIX: Correctly initializes 'best_price_seen' from the 'entry_price' message.

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
    config,
    ATR_TIMEFRAME, 
    LATEST_ENRICHED_KEY 
)
from utils.api_client import DeltaAPIClient

logger = logging.getLogger("tsl_manager")

# --- TrailingStopManager Class ---

class TrailingStopManager:
    
    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession, api_client: DeltaAPIClient):
        self.redis = redis_client
        self.session = http_session # For unauthenticated calls
        self.api_client = api_client # For authenticated calls
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.tsl_tasks: Dict[int, asyncio.Task] = {} 
        
        # DYNAMIC TSL CONFIGURATION
        self.tsl_config = {
            "trail_multiplier": config["TSL_ATR_MULTIPLIER"],
            "min_trail_amount": config["TSL_MIN_TRAIL_AMOUNT"],
            "check_interval": config["TSL_CHECK_INTERVAL"],
            "atr_timeframe": ATR_TIMEFRAME
        }
        
        if not config["TSL_ENABLED"]:
            logger.warning("🚫 TSL Manager is initialized but TSL_ENABLED is False. It will not run.")

    # --- Helper to fetch ATR from cached enriched data ---
    async def _get_latest_atr(self, symbol: str) -> Optional[float]:
        """Fetches the latest ATR value from the FeatureEngine's cached data."""
        try:
            enriched_json = await self.redis.get(f"{LATEST_ENRICHED_KEY}{symbol}")
            if enriched_json:
                enriched = json.loads(enriched_json)
                # Access 'tas' dictionary, then ATR_TIMEFRAME, then 'atr' key
                atr_data = enriched.get("tas", {}).get(self.tsl_config["atr_timeframe"], {})
                atr = atr_data.get("atr")
                return float(atr) if atr is not None else None
        except Exception as e:
            logger.warning(f"Error fetching/parsing ATR for {symbol}. Error: {e}")
            return None
        return None
        
    async def fetch_ticker_data(self, symbol: str) -> Optional[float]:
        """Fetches the live Mark Price (current_market_price) for a specific product symbol."""
        path = f"/v2/tickers/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"
        
        headers = {'Accept': 'application/json', 'User-Agent': USER_AGENT} 
        
        try:
            # Use the shared unauthenticated session
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
        params = {
            "product_ids": str(product_id), 
            "states": "open,pending", 
            "stop_order_type": "stop_loss_order" 
        }
        
        # UPDATED: Use the centralized API client
        status, data = await self.api_client.get(path, params=params)
        
        if status == 200 and data and data.get('success'):
            stop_orders = data.get("result", [])
            
            for order in stop_orders:
                if order.get("stop_order_type") == "stop_loss_order" and order.get("state") in ["open", "pending"]:
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
        
        # For SHORT position (size < 0), stop is a 'buy'
        # For LONG position (size > 0), stop is a 'sell'
        sl_side = "buy" if size < 0 else "sell"
        
        request_data = {
            "id": order_id,
            "product_id": product_id,
            "size": abs(size), 
            "side": sl_side,
            # CRITICAL: We update the stop_price and keep the order_type as market_order 
            "order_type": "market_order", 
            "stop_price": f"{new_stop_price:.4f}", # Format price to string
            "reduce_only": True
        }
        
        logger.info(f"Attempting PUT update for Stop Order {order_id} to price {new_stop_price:.4f}...")
        
        # UPDATED: Use the centralized API client
        status, response_data = await self.api_client.put(path, payload=request_data)
        
        if status == 200 and response_data and response_data.get('success'):
            logger.info(f"✅ Stop order {order_id} updated successfully to {new_stop_price:.4f}!")
            return True
        else:
            logger.error(f"❌ Failed to update stop order (HTTP {status}). Response: {response_data}")
            return False

    # --- Core Trailing Logic ---

    async def _trailing_loop(
        self, 
        product_id: int, 
        symbol: str, 
        direction: str, 
        size: int, 
        entry_price: float  # ⭐️ FIX: Accept the initial trade price
    ):
        """Continuous trailing stop logic using dynamic ATR-based trail amount."""
        
        trail_multiplier = self.tsl_config["trail_multiplier"]
        min_trail_amount = self.tsl_config["min_trail_amount"]
        check_interval = self.tsl_config["check_interval"]
        
        # ⭐️ FIX: Initialize best_price_seen to the actual entry price
        # This is the "starting point" you mentioned.
        best_price_seen: float = entry_price
        stop_order_id: Optional[int] = None
        
        logger.info(f"TSL Loop started for {symbol}. Initial best price set to entry price: {best_price_seen:.4f}")
        
        # 1. Wait for the SL order to appear in the exchange system
        wait_attempts = 0
        max_wait_attempts = 10
        while stop_order_id is None and wait_attempts < max_wait_attempts:
            logger.info(f"TSL Manager: Waiting for initial SL order for {symbol} (Attempt {wait_attempts+1}/{max_wait_attempts})...")
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
                # 2. Fetch Live Price (This is the "current_market_price")
                live_mark_price = await self.fetch_ticker_data(symbol)
                
                if live_mark_price is None:
                    logger.warning(f"Could not fetch live_mark_price for {symbol}. Skipping TSL loop.")
                    await asyncio.sleep(check_interval)
                    continue

                # 3. Dynamic Trail Amount Calculation
                latest_atr = await self._get_latest_atr(symbol)
                
                if latest_atr is not None and latest_atr > 0:
                    # Calculate dynamic trail amount, ensuring it meets the minimum floor
                    dynamic_trail_amount = max(latest_atr * trail_multiplier, min_trail_amount)
                    current_trail_amount = dynamic_trail_amount
                    logger.debug(f"Dynamic Trail calculated for {symbol}: ATR={latest_atr:.4f}, Trail={current_trail_amount:.4f}")
                else:
                    # Fallback to the original static amount if ATR is missing
                    current_trail_amount = config["TSL_TRAIL_AMOUNT"]
                    logger.debug(f"ATR missing. Using static fallback trail amount: {current_trail_amount:.4f}")


                # 4. Update the best price seen (most favorable price)
                is_new_best = False
                if direction == "LONG" and live_mark_price > best_price_seen:
                    best_price_seen = live_mark_price
                    is_new_best = True
                elif direction == "SHORT" and live_mark_price < best_price_seen:
                    best_price_seen = live_mark_price
                    is_new_best = True
                
                if is_new_best:
                    logger.info(f"New Best Mark Price tracked for {symbol}: {best_price_seen:.4f}")

                # 5. Calculate the required Trailing Stop Price based on the best price
                if direction == "LONG":
                    # For LONG: Stop Price = Highest Mark Price - Dynamic Trail Amount
                    required_stop_price = best_price_seen - current_trail_amount
                else: # SHORT
                    # For SHORT: Stop Price = Lowest Mark Price + Dynamic Trail Amount
                    required_stop_price = best_price_seen + current_trail_amount
                
                # 6. Fetch current stop price and check for update
                path = "/v2/orders"
                params = {"order_ids": str(stop_order_id)}
                
                status, order_details = await self.api_client.get(path, params=params)
                
                current_stop_price = None
                if status == 200 and order_details and order_details.get('result'):
                    order_result = order_details['result']
                    if order_result:
                        current_stop_price = float(order_result[0].get('stop_price', 0))
                
                if current_stop_price is None or current_stop_price == 0.0:
                    logger.warning(f"Could not find current stop price for order {stop_order_id}. Skipping loop.")
                    await asyncio.sleep(check_interval)
                    continue
                    
                tolerance = 0.0001 # To prevent floating point issues
                update_required = False
                
                # Logic to only move the stop "in profit"
                if direction == "LONG" and required_stop_price > current_stop_price + tolerance:
                    update_required = True
                elif direction == "SHORT" and required_stop_price < current_stop_price - tolerance:
                    update_required = True
                
                if update_required:
                    logger.info(f"Condition met: Moving SL for {symbol} from {current_stop_price:.4f} "
                                f"to {required_stop_price:.4f} (Trail: {current_trail_amount:.4f})")
                                
                    await self.update_stop_price(
                        stop_order_id, 
                        product_id, 
                        size, 
                        required_stop_price
                    )
                else:
                    logger.debug(f"Stop Price for {symbol} does not require an update. (Current SL: {current_stop_price:.4f} / Required SL: {required_stop_price:.4f})")

                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in TSL loop for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(check_interval * 2)
    
    async def _handle_tsl_control_message(self, data: Dict[str, Any]):
        """Starts or stops TSL tracking based on messages from Executor or Monitor."""
        message_type = data.get("type")
        symbol = data.get("symbol")
        product_id = data.get("product_id")
        
        if not product_id: return
        
        if message_type == "start_tsl" and config["TSL_ENABLED"]:
            direction = data.get("direction")
            size = data.get("size")
            # ⭐️ FIX: Read the entry_price from the message
            entry_price = data.get("entry_price")
            
            # ⭐️ FIX: Add validation for new required field
            if not all([direction, size is not None, entry_price is not None]):
                logger.error(f"TSL Manager received 'start_tsl' for {symbol} but was missing direction, size, or entry_price.")
                return
            
            if product_id in self.tsl_tasks: return

            logger.info(f"🎯 Starting TSL for {symbol} ({direction}, ID: {product_id}) at price {entry_price}")
            
            # ⭐️ FIX: Pass entry_price to the trailing loop task
            task = asyncio.create_task(
                self._trailing_loop(product_id, symbol, direction, size, entry_price),
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
                        # If monitor sees position closed, stop TSL task
                        closed_symbol = data.get("symbol")
                        # Infer product_id from running task names
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