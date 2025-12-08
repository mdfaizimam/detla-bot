# --- detla-bot/trailing_stop_manager.py ---
# ✅ FIX: Explicitly handles "Order Already Triggered" error (Stops Zombie Loop)
# ✅ FIX: Implements TSL Activation Buffer (Prevents immediate stop out)
# ✅ FIX: Added "Heartbeat" log to show monitoring status even when not updating
# ✅ NEW: Added DYNAMIC TSL LOGIC (Tightens stop as profit increases)
# ✅ FIX: Ghost Order Crash (Handles 'open_order_not_found' gracefully)

import asyncio
import aiohttp
import json
import logging
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
    LATEST_ENRICHED_KEY,
    BRACKET_STOP_TRIGGER,
)
from utils.api_client import DeltaAPIClient

logger = logging.getLogger("tsl_manager")


class TrailingStopManager:
    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession, api_client: DeltaAPIClient):
        self.redis = redis_client
        self.session = http_session
        self.api_client = api_client

        self.tsl_tasks: Dict[int, asyncio.Task] = {}
        self.sl_search_attempts: Dict[int, int] = {}
        self.active_positions: Dict[int, Dict[str, Any]] = {}
        self.product_to_symbol: Dict[int, str] = {}

        self.tsl_config = {
            "trail_multiplier": config["TSL_ATR_MULTIPLIER"],
            "min_trail_amount": config["TSL_MIN_TRAIL_AMOUNT"],
            "check_interval": config["TSL_CHECK_INTERVAL"],
            "atr_timeframe": ATR_TIMEFRAME,
            "activation_pct": config.get("TSL_ACTIVATION_PCT", 0.005) # ✅ Default 0.5%
        }

        self._runner_task: Optional[asyncio.Task] = None

    async def start(self):
        """Listen for TSL start requests and position close events."""
        logger.info("▶️ TSLManager starting...")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(TSL_CHANNEL, MONITORING_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                channel = msg.get("channel")
                try:
                    data = json.loads(msg["data"])
                except Exception:
                    continue
                
                # ✅ NEW: Handle explicit STOP command from Monitor
                if channel == TSL_CHANNEL:
                    command = data.get("command")
                    
                    if command == "START_TSL":
                        product_id = int(data["product_id"])
                        symbol = data["symbol"]
                        
                        self.product_to_symbol[product_id] = symbol
                        
                        if product_id in self.tsl_tasks and not self.tsl_tasks[product_id].done():
                            logger.info("TSL already active for product_id=%s", product_id)
                            continue
                        
                        task = asyncio.create_task(
                            self._trailing_loop(
                                product_id=product_id,
                                symbol=symbol,
                                direction=data["direction"],
                                size=int(data["size"]) if float(data["size"]) == int(data["size"]) else float(data["size"]),
                                entry_price=float(data["entry_price"]),
                            ),
                            name=f"TSL-{product_id}"
                        )
                        self.tsl_tasks[product_id] = task
                        self.sl_search_attempts[product_id] = 0
                        
                        self.active_positions[product_id] = {
                            "symbol": symbol,
                            "direction": data["direction"],
                            "size": float(data["size"]),
                            "entry_price": float(data["entry_price"]),
                            "active": True,
                            "last_validated": asyncio.get_event_loop().time()
                        }
                    
                    elif command == "STOP_TSL":
                        # Monitor tells us to kill TSL because position is closed
                        symbol = data.get("symbol")
                        product_id = None
                        # Find product_id by symbol
                        for pid, info in self.active_positions.items():
                            if info["symbol"] == symbol:
                                product_id = pid
                                break
                        
                        if product_id:
                            logger.info(f"🛑 Received STOP_TSL command for {symbol}. Killing loop.")
                            if product_id in self.active_positions:
                                self.active_positions[product_id]["active"] = False
                            if product_id in self.tsl_tasks and not self.tsl_tasks[product_id].done():
                                self.tsl_tasks[product_id].cancel()

                elif channel == MONITORING_CHANNEL:
                    event_type = data.get("type")
                    
                    if event_type == "position_closed":
                        await self._handle_position_closed(data)
                    
                    elif event_type == "reconciler_lock_released":
                        await self._handle_lock_released(data)
                    
                    elif event_type == "position_updated":
                        product_id = data.get("product_id")
                        if product_id and product_id in self.active_positions:
                            size = float(data.get("size", 0))
                            if size == 0:
                                await self._handle_position_closed({"product_id": product_id, "symbol": data.get("symbol")})

        except asyncio.CancelledError:
            logger.info("TSLManager cancelled.")
        finally:
            await pubsub.unsubscribe(TSL_CHANNEL, MONITORING_CHANNEL)

    async def _handle_position_closed(self, data: Dict[str, Any]):
        """Handle position closed event."""
        product_id = data.get("product_id")
        symbol = data.get("symbol")
        
        if product_id:
            if product_id in self.tsl_tasks and not self.tsl_tasks[product_id].done():
                self.tsl_tasks[product_id].cancel()
                logger.info("🛑 Stopping TSL (position closed). product_id=%s", product_id)
            
            await self._cleanup_product(product_id)

    async def _handle_lock_released(self, data: Dict[str, Any]):
        """Handle reconciler lock released event."""
        symbol = data.get("symbol")
        product_id = None
        for pid, pos_info in self.active_positions.items():
            if pos_info["symbol"] == symbol:
                product_id = pid
                break
        
        if product_id:
            logger.warning("🛑 Reconciler released lock for %s. Stopping TSL.", symbol)
            if product_id in self.tsl_tasks and not self.tsl_tasks[product_id].done():
                self.tsl_tasks[product_id].cancel()
            
            await self._cleanup_product(product_id)

    async def validate_position_active(self, product_id: int) -> bool:
        """Check if position is still active before updating stop-loss."""
        if product_id not in self.active_positions:
            return False
        
        if not self.active_positions[product_id].get("active", True):
            return False
        
        current_time = asyncio.get_event_loop().time()
        last_validated = self.active_positions[product_id].get("last_validated", 0)
        if current_time - last_validated < 30:
            return True
        
        try:
            path = "/v2/positions"
            params = {"product_id": str(product_id)}
            
            status, data = await self.api_client.get(path, params=params)
            
            if status == 200 and data and isinstance(data, dict) and data.get("success"):
                result = data.get("result")
                
                if isinstance(result, dict):
                     size = float(result.get("size", 0))
                     direction = self.active_positions[product_id]["direction"]
                     if (direction == "LONG" and size > 0) or (direction == "SHORT" and size < 0):
                        self.active_positions[product_id]["last_validated"] = current_time
                        return True
                
                elif isinstance(result, list):
                    for position in result:
                        if not isinstance(position, dict): continue
                        if int(position.get("product_id", 0)) == product_id:
                            size = float(position.get("size", 0))
                            direction = self.active_positions[product_id]["direction"]
                            
                            if (direction == "LONG" and size > 0) or (direction == "SHORT" and size < 0):
                                self.active_positions[product_id]["last_validated"] = current_time
                                return True
                
                self.active_positions[product_id]["active"] = False
                return False
            
            elif status == 400:
                symbol = self.active_positions[product_id].get("symbol")
                if symbol and isinstance(symbol, str):
                    logger.info(f"⚠️ Validation by ID failed. Retrying validation for {symbol} using symbol param.")
                    params_sym = {"underlying_asset_symbol": symbol}
                    status2, data2 = await self.api_client.get("/v2/positions", params=params_sym)
                    
                    if status2 == 200 and isinstance(data2, dict) and data2.get("success"):
                        positions = data2.get("result", [])
                        if isinstance(positions, list):
                            for position in positions:
                                if not isinstance(position, dict): continue
                                if int(position.get("product_id", 0)) == product_id:
                                    size = float(position.get("size", 0))
                                    direction = self.active_positions[product_id]["direction"]
                                    
                                    if (direction == "LONG" and size > 0) or (direction == "SHORT" and size < 0):
                                        self.active_positions[product_id]["last_validated"] = current_time
                                        return True
                        
                        self.active_positions[product_id]["active"] = False
                        return False
            
            logger.warning("⚠️ Failed to validate position for product_id=%s (HTTP %s). Assuming active.", product_id, status)
            return True
                
        except Exception as e:
            logger.warning("Error validating position for product_id=%s: %s", product_id, e)
            return True
        
        return True

    async def _get_latest_atr(self, symbol: str) -> Optional[float]:
        try:
            enriched_json = await self.redis.get(f"{LATEST_ENRICHED_KEY}{symbol}")
            if not enriched_json:
                return None
            enriched = json.loads(enriched_json)
            atr = (
                enriched.get("tas", {})
                .get(self.tsl_config["atr_timeframe"], {})
                .get("atr")
            )
            return float(atr) if atr is not None else None
        except Exception as e:
            logger.warning("Error fetching ATR for %s: %s", symbol, e)
            return None

    async def fetch_ticker_data(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        path = f"/v2/tickers/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        try:
            async with self.session.get(url, headers=headers) as resp:
                data = await resp.json()
                if resp.status == 200 and data.get("success"):
                    r = data.get("result", {})
                    return (
                        float(r.get("mark_price")) if r.get("mark_price") is not None else None,
                        float(r.get("close")) if r.get("close") is not None else None,
                    )
                logger.error("❌ Failed to fetch ticker (HTTP %s) for %s: %s", resp.status, symbol, data)
                return (None, None)
        except Exception as e:
            logger.error("❌ Error fetching ticker data: %s", e, exc_info=True)
            return (None, None)

    async def fetch_open_stop_order_id(self, product_id: int, increment_attempt: bool = True) -> Optional[Tuple[int, str]]:
        if increment_attempt:
            if product_id in self.sl_search_attempts:
                self.sl_search_attempts[product_id] += 1
            else:
                self.sl_search_attempts[product_id] = 1
            
            if self.sl_search_attempts[product_id] > 3:
                logger.warning("❌ Max search attempts (3) reached for stop-loss order on product_id=%s.", product_id)
                return None
        
        path = "/v2/orders"
        params = {
            "product_ids": str(product_id), 
            "stop_order_type": "stop_loss_order",
        }
        status, data = await self.api_client.get(path, params=params)
        if status == 200 and data and data.get("success"):
            for order in data.get("result", []):
                if (order.get("stop_order_type") == "stop_loss_order" and 
                    order.get("state") in ("open", "pending")):
                    oid = order.get("id")
                    otype = order.get("order_type") or "market_order"
                    logger.debug("Found SL child: id=%s type=%s stop=%s", oid, otype, order.get("stop_price"))
                    if product_id in self.sl_search_attempts:
                        self.sl_search_attempts[product_id] = 0
                    return int(oid), str(otype)
            
            logger.warning("No open Stop-Loss order found for product_id=%s (attempt %d/3).", 
                          product_id, self.sl_search_attempts.get(product_id, 1))
            return None
        logger.error("❌ Failed to fetch stop orders (HTTP %s): %s", status, data)
        return None

    async def update_stop_price(
        self,
        order_id: int,
        product_id: int,
        size: int | float,
        new_stop_price: float,
        order_type: str = "market_order",
    ) -> bool:
        # Check active status periodically
        update_count = self.active_positions.get(product_id, {}).get("update_count", 0)
        if update_count % 5 == 0:
            if not await self.validate_position_active(product_id):
                logger.warning("⚠️ Position not active for product_id=%s. Skipping stop-loss update.", product_id)
                return False
        
        if product_id in self.active_positions:
            self.active_positions[product_id]["update_count"] = update_count + 1
        
        path = "/v2/orders"
        sl_side = "sell" if float(size) > 0 else "buy"

        req = {
            "id": int(order_id),
            "product_id": int(product_id),
            "size": abs(int(size) if float(size) == int(size) else float(size)),
            "side": sl_side,
            "order_type": order_type,
            "stop_price": f"{float(new_stop_price):.4f}",
            "reduce_only": True,
        }
        if order_type == "limit_order":
            req["limit_price"] = f"{float(new_stop_price):.4f}"

        logger.info("PUT /v2/orders (stop edit) -> id=%s price=%s type=%s", order_id, req["stop_price"], order_type)
        status, resp = await self.api_client.put(path, payload=req)
        
        if status == 200 and resp and resp.get("success"):
            logger.info("✅ Stop order %s updated to %s", order_id, req["stop_price"])
            return True
        
        # ✅ FIX: Handle Ghost Order Errors Gracefully
        if status == 400:
            error_code = resp.get("error", {}).get("code") if resp else "unknown"
            
            if error_code == "stop_price_change_not_supported":
                logger.error(f"🛑 Order {order_id} triggered/closed. Stopping TSL.")
                if product_id in self.active_positions:
                    self.active_positions[product_id]["active"] = False
                return False

            if error_code == "open_order_not_found":
                logger.warning("⚠️ Ghost Order detected: Stop-loss %s not found. Assuming position closed.", order_id)
                if product_id in self.active_positions:
                    self.active_positions[product_id]["active"] = False
                return False
                
            else:
                logger.error("❌ Failed to update stop order (HTTP 400, code=%s): %s", error_code, resp)
        else:
            logger.error("❌ Failed to update stop order (HTTP %s): %s", status, resp)
        
        return False

    async def _trailing_loop(
        self,
        product_id: int,
        symbol: str,
        direction: str,
        size: int | float,
        entry_price: float
    ):
        base_multiplier = float(self.tsl_config["trail_multiplier"])
        min_trail_amount = float(self.tsl_config["min_trail_amount"])
        activation_pct = float(self.tsl_config["activation_pct"]) 
        check_interval = float(self.tsl_config["check_interval"])

        best_price_seen: float = float(entry_price)
        sl_tuple: Optional[Tuple[int, str]] = None
        consecutive_order_errors = 0
        max_consecutive_errors = 3
        
        loop_counter = 0

        logger.info("TSL Loop for %s started @ entry=%.4f (base trail x%.2f activation=%.2f%%)",
                    symbol, best_price_seen, base_multiplier, activation_pct*100)

        for attempt in range(3):
            sl_tuple = await self.fetch_open_stop_order_id(product_id, increment_attempt=(attempt == 0))
            if sl_tuple:
                break
            if attempt < 2:
                logger.info("Waiting for SL child (attempt %d/3)...", attempt + 1)
                await asyncio.sleep(1.0)
            else:
                logger.warning("Max attempts reached. No SL child found after 3 tries.")

        if not sl_tuple:
            logger.error("❌ No SL child found after 3 attempts for %s.", symbol)
            await self._cleanup_product(product_id)
            return

        stop_order_id, stop_order_type = sl_tuple
        logger.info("✅ TSL active for %s: order_id=%s type=%s", symbol, stop_order_id, stop_order_type)

        if product_id in self.sl_search_attempts:
            self.sl_search_attempts[product_id] = 0

        if product_id in self.active_positions:
            self.active_positions[product_id]["update_count"] = 0

        while True:
            try:
                loop_counter += 1
                
                if product_id not in self.active_positions or not self.active_positions[product_id].get("active", True):
                    logger.info("🛑 Position marked as inactive for %s. Stopping TSL.", symbol)
                    break

                mark_price, last_price = await self.fetch_ticker_data(symbol)
                live_price = mark_price if BRACKET_STOP_TRIGGER == "mark_price" else last_price

                if live_price is None:
                    await asyncio.sleep(check_interval)
                    continue

                profit_pct = 0.0
                if direction == "LONG":
                    profit_pct = (live_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - live_price) / entry_price

                current_multiplier = base_multiplier
                mode = "BASE"
                
                if profit_pct > 0.03: 
                    current_multiplier = 1.0 
                    mode = "TIGHT (3% gain)"
                elif profit_pct > 0.015: 
                    current_multiplier = 1.5
                    mode = "MEDIUM (1.5% gain)"
                
                if loop_counter % 12 == 0:
                    status_msg = "WAITING" if profit_pct < activation_pct else f"ACTIVE ({mode})"
                    logger.info(
                        f"💓 TSL Monitor [{symbol}]: PnL={profit_pct*100:.2f}% | "
                        f"Mult={current_multiplier} | Status={status_msg} | "
                        f"Price={live_price:.2f} | Best={best_price_seen:.2f}"
                    )

                if profit_pct < activation_pct:
                    await asyncio.sleep(check_interval)
                    continue

                latest_atr = await self._get_latest_atr(symbol)
                if latest_atr is not None and latest_atr > 0:
                    trail_amt = max(latest_atr * current_multiplier, min_trail_amount)
                else:
                    trail_amt = min_trail_amount

                should_update = False
                new_stop = 0.0

                if direction == "LONG":
                    if live_price > best_price_seen:
                        best_price_seen = live_price
                        new_stop = best_price_seen - trail_amt
                        should_update = True 
                else:
                    if live_price < best_price_seen:
                        best_price_seen = live_price
                        new_stop = best_price_seen + trail_amt
                        should_update = True

                if should_update:
                    success = await self.update_stop_price(
                        order_id=stop_order_id,
                        product_id=product_id,
                        size=size,
                        new_stop_price=new_stop,
                        order_type=stop_order_type,
                    )
                    
                    if success:
                        consecutive_order_errors = 0
                    else:
                        if product_id in self.active_positions and not self.active_positions[product_id].get("active", True):
                             break
                        consecutive_order_errors += 1
                        if consecutive_order_errors >= max_consecutive_errors:
                            logger.error("❌ Max consecutive errors reached for %s. Stopping TSL.", symbol)
                            break

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                logger.info("TSL loop cancelled for %s", symbol)
                break
            except Exception as e:
                logger.error("❌ Error in TSL loop for %s: %s", symbol, e, exc_info=True)
                consecutive_order_errors += 1
                if consecutive_order_errors >= max_consecutive_errors:
                    logger.error("❌ Max consecutive errors reached for %s. Stopping TSL.", symbol)
                    break
                await asyncio.sleep(check_interval)
        
        await self._cleanup_product(product_id)
        logger.info("✅ TSL stopped for %s", symbol)

    async def _cleanup_product(self, product_id: int):
        if product_id in self.tsl_tasks:
            del self.tsl_tasks[product_id]
        if product_id in self.sl_search_attempts:
            del self.sl_search_attempts[product_id]
        if product_id in self.active_positions:
            del self.active_positions[product_id]
        if product_id in self.product_to_symbol:
            del self.product_to_symbol[product_id]

    async def close(self):
        for pid in list(self.tsl_tasks.keys()):
            if pid in self.tsl_tasks and not self.tsl_tasks[pid].done():
                self.tsl_tasks[pid].cancel()
            await self._cleanup_product(pid)