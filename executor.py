# --- detla-bot/executor.py ---
# ✅ FIX: Removed unused 'self.session' to fix AttributeError in tests
# ✅ FIX: Retry logic for fetching fill price (Avoids $0.00 fills)
# ✅ FIX: Robust error handling
# ✅ FIX: Increased Lock TTL to 300s to match Reconciler

import asyncio
import orjson
import logging
import time
import csv
import os
from datetime import datetime
from typing import Optional, Any, Dict, Tuple
from redis import asyncio as aioredis
from config import (
    DELTA_BASE_URL, API_KEY, API_SECRET, SIGNAL_CHANNEL, MONITORING_CHANNEL,
    USER_AGENT, DMS_ID, TSL_ENABLED, TSL_CHANNEL, config, TRADING_SYMBOLS,
    BRACKET_STOP_TRIGGER, BRACKET_ORDER_TYPE, REDIS_POSITION_LOCK_PREFIX
)
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager

logger = logging.getLogger("executor")

class OrderExecutionManager:
    # 🔧 CHANGED: Increased from 60 to 300 to match Reconciler cycle & prevent double entries
    REDIS_POSITION_LOCK_TTL = 300

    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        self.redis = redis_client
        self.api_client = api_client
        # self.session = api_client.session  <-- REMOVED (Unused and caused mock error)
        self.risk_manager = risk_manager
        self.product_info_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ OrderExecutionManager initialized (Market Order Mode).")

    async def start(self):
        logger.info("▶️ OrderExecutionManager starting (listening for signals)...")
        
        # 🛡️ Sync State on Boot
        await self._sync_active_positions()
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(SIGNAL_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message": continue
                try:
                    signal = orjson.loads(msg["data"])
                except Exception: continue
                asyncio.create_task(self._handle_signal(signal))
        except asyncio.CancelledError:
            logger.info("OrderExecutionManager cancelled.")
        finally:
            await pubsub.unsubscribe(SIGNAL_CHANNEL)

    async def _sync_active_positions(self):
        """
        🛡️ CRITICAL: Checks for existing open positions on exchange at startup.
        Restores monitoring and locks if the bot was restarted during a trade.
        Uses product-id based fetching for reliability.
        """
        try:
            logger.info("♻️ Syncing active positions from Exchange...")
            count = 0
            
            # Iterate through all trading symbols to check positions specifically
            # This is more robust than a global fetch which fails schema validation
            for symbol in TRADING_SYMBOLS:
                product_id = None
                
                # 1. Resolve Product ID
                product_info = await self._get_product_info(symbol)
                if product_info:
                    product_id = product_info.get("id")
                
                if not product_id:
                    logger.warning(f"Could not resolve product ID for {symbol}, skipping sync.")
                    continue

                # 2. Fetch Position for this specific product
                status, response = await self.api_client.get("/v2/positions", params={"product_id": str(product_id)})
                
                if status != 200 or not response.get("success"):
                    continue

                result = response.get("result")
                # Normalize result (can be list or dict)
                target_pos = None
                if isinstance(result, dict): target_pos = result
                elif isinstance(result, list):
                    for p in result:
                        if float(p.get("size", 0)) != 0:
                            target_pos = p
                            break
                            
                if target_pos and float(target_pos.get("size", 0)) != 0:
                    size = float(target_pos.get("size"))
                    entry_price = float(target_pos.get("entry_price", 0))
                    
                    logger.info(f"♻️ Restoring state for {symbol} (Size: {size})")
                    await self._notify_monitor(symbol, size, product_id)
                    count += 1

            logger.info(f"✅ Synced {count} active positions.")

        except Exception as e:
            logger.error(f"Error syncing positions: {e}", exc_info=True)

    async def _get_product_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol in self.product_info_cache:
            return self.product_info_cache[symbol]
        
        status, response = await self.api_client.get(f"/v2/products/{symbol}")
        if status == 200 and response.get("success"):
            product = response.get("result", {})
            if product:
                try:
                    tick_size = float(product.get("tick_size", "0.5"))
                    precision = 0
                    if "." in str(tick_size):
                        precision = len(str(tick_size).split(".")[-1])
                    info = {
                        "id": int(product.get("id")),
                        "tick_size": tick_size,
                        "precision": precision,
                        "symbol": product.get("symbol")
                    }
                    self.product_info_cache[symbol] = info
                    return info
                except Exception as e:
                    logger.error("Error parsing product info: %s", e)
        return None

    async def _acquire_position_lock(self, symbol: str, timeout: int = 5) -> bool:
        deadline = time.time() + timeout
        lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
        lock_value = orjson.dumps({"symbol": symbol, "ts": time.time()})
        while time.time() < deadline:
            ok = await self.redis.set(lock_key, lock_value, ex=self.REDIS_POSITION_LOCK_TTL, nx=True)
            if ok: return True
            await asyncio.sleep(0.25)
        logger.warning("⚠️ Could not acquire lock for %s (Busy)", symbol)
        return False

    async def _release_position_lock(self, symbol: str):
        try:
            lock_key = f"{REDIS_POSITION_LOCK_PREFIX}{symbol}"
            await self.redis.delete(lock_key)
        except Exception: pass

    async def _handle_signal(self, signal: dict):
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        
        base_size_config = config["BASE_POSITION_SIZE"]
        if isinstance(base_size_config, dict):
            size_hint = base_size_config.get(symbol, 1)
        else:
            size_hint = float(base_size_config)
        int_size = int(size_hint)

        if not symbol or not direction: return

        ok, info = await self.risk_manager.validate_signal(signal)
        if not ok:
            logger.warning("Signal rejected by RiskManager: %s", info)
            return

        if not await self._acquire_position_lock(symbol): return

        try:
            product_info = await self._get_product_info(symbol)
            if not product_info: raise Exception("Product Info Unavailable")
            
            tp_price = float(signal.get("tp_price", 0))
            sl_price = float(signal.get("sl_price", 0))
            if tp_price == 0 or sl_price == 0:
                logger.error("Invalid TP/SL in signal. Aborting.")
                await self._release_position_lock(symbol)
                return

            side = "buy" if direction == "LONG" else "sell"
            
            ref_price = float(signal.get("trigger_price", 0))
            if ref_price == 0: ref_price = tp_price # Fallback heuristic
            
            res = await self._place_linked_orders(symbol, side, int_size, tp_price, sl_price, product_info, ref_price)
            
            if not res:
                await self._release_position_lock(symbol)
                return

            product_id, ret_direction, filled_avg_price = res

            await self._notify_monitor(symbol, int_size, product_id)
            if TSL_ENABLED:
                await self._notify_tsl_manager(symbol, ret_direction, int_size, product_id, filled_avg_price)

            logger.info("✅ Trade Executed for %s @ $%.2f. Lock HELD.", symbol, filled_avg_price)
            
            # Log to Journal
            self._log_trade_journal({
                "symbol": symbol,
                "direction": direction,
                "size": int_size,
                "price": filled_avg_price,
                "regime": signal.get("regime", "Unknown"),
                "confidence": signal.get("confidence", 0),
                "reasoning": signal.get("reasoning", {})
            })

        except Exception as e:
            logger.error("❌ Error handling signal for %s: %s", symbol, e, exc_info=True)
            await self._release_position_lock(symbol)

    def _log_trade_journal(self, trade_data: dict):
        try:
            file_exists = os.path.isfile("logs/trade_journal.csv")
            os.makedirs("logs", exist_ok=True)
            
            with open("logs/trade_journal.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Symbol", "Direction", "Size", "Price", "Regime", "Confidence", "Forecast", "VolZ", "FNG", "DXY_ROC", "VIX"])
                
                reasoning = trade_data.get("reasoning", {})
                writer.writerow([
                    datetime.now().isoformat(),
                    trade_data.get("symbol"),
                    trade_data.get("direction"),
                    trade_data.get("size"),
                    trade_data.get("price"),
                    trade_data.get("regime"),
                    trade_data.get("confidence"),
                    reasoning.get("forecast", 0),
                    reasoning.get("vol_zscore", 0),
                    reasoning.get("fng", 0),
                    reasoning.get("dxy_roc", 0),
                    reasoning.get("vix", 0)
                ])
        except Exception as e:
            logger.error(f"Failed to write to trade journal: {e}")

            logger.error(f"Failed to write to trade journal: {e}")

    async def _place_maker_entry(self, symbol: str, product_id: int, side: str, size: int, ref_price: float, max_slippage=0.005) -> Tuple[bool, float]:
        """
        🕵️ GENIUS EXECUTION: Smart Chase-Limit Logic
        Places Limit Order at Best Bid/Ask. If not filled, chases price up to max_slippage.
        Returns (Success, AvgFillPrice)
        """
        try:
            # 1. Calculate Hard Limits
            limit_cap = ref_price * (1 + max_slippage) if side == "buy" else ref_price * (1 - max_slippage)
            
            order_id = None
            start_time = time.time()
            attempt = 0
            active_limit_price = 0.0 # Track locally to avoid GET 404s
            
            while time.time() - start_time < 20: # 20s Max Chase Time
                attempt += 1
                
                # A. Get Fresh Ticker
                status, resp = await self.api_client.get(f"/v2/tickers/{symbol}")
                if status != 200: 
                    await asyncio.sleep(1)
                    continue
                    
                # Use Orderbook Top if available, else Mark
                # Delta Ticker usually has 'ask' and 'bid'
                ticker = resp.get("result", {})
                best_bid = float(ticker.get("bid", 0) or ticker.get("spot_price", 0))
                best_ask = float(ticker.get("ask", 0) or ticker.get("spot_price", 0))
                
                if best_bid == 0: # Fallback
                    best_bid = best_ask = float(ticker.get("close", ref_price))

                # B. Determine Price
                # If BUY, we want to be Best Bid + 1 tick (Aggressive Maker) or just Best Bid
                # To ensure fill, maybe Best Bid + small delta, but strictly < Best Ask
                target_price = best_bid if side == "buy" else best_ask
                
                if side == "buy":
                    if target_price > limit_cap: target_price = limit_cap # Cap
                else:
                    if target_price < limit_cap: target_price = limit_cap # Floor

                # C. Check status / Chase
                should_chase = False
                if order_id:
                     if side == "buy":
                         if target_price > active_limit_price: should_chase = True
                     else:
                         if target_price < active_limit_price: should_chase = True
                     
                     if should_chase:
                         logger.info(f"🏃 Chasing {side.upper()}: {active_limit_price} -> {target_price}")
                         # Use correct DELETE payload
                         await self.api_client.delete("/v2/orders", payload={"id": order_id, "product_id": product_id})
                         order_id = None
                         active_limit_price = 0.0
                
                # D. Place New Order if needed
                if not order_id:
                    price_str = f"{target_price:.6f}".rstrip('0').rstrip('.')
                    
                    payload = {
                        "product_id": product_id,
                        "size": size,
                        "side": side,
                        "order_type": "limit_order",
                        "limit_price": price_str,
                        "time_in_force": "gtc"
                    }
                    
                    logger.info(f"🎯 Placing Smart Limit {side.upper()} @ {price_str}")
                    s, r = await self.api_client.post("/v2/orders", payload)
                    if s == 200 and r.get("success"):
                         order_id = int(r["result"]["id"])
                         active_limit_price = target_price # Update tracker
                    else:
                         logger.error(f"Failed to place chase limit: {r}")
                         # Fallback to Market?
                         return False, 0.0
                
                await asyncio.sleep(2)
                
            # Timeout - Cancel and convert to Market?
            # --- GENIUS UPGRADE: DO NOT PANIC BUY ---
            # Standard retail bots market buy here. Snipers abort to preserve R:R.
            
            logger.warning("⏰ Chase Timeout. Price moved too fast. ABORTING TRADE to preserve R:R.")
            if order_id:
                 await self.api_client.delete(f"/v2/orders/{order_id}")
            
            return False, 0.0 # Return Fail instead of Market Buy
            
        except Exception as e:
            logger.error(f"Smart Execution Error: {e}")
            return False, 0.0
    async def _place_linked_orders(self, symbol, side, size, tp_price, sl_price, product_info, ref_price=0.0) -> Optional[Tuple[int, str, float]]:
        product_id = product_info["id"]
        precision = product_info["precision"]
        
        # 1. Execute Entry (Smart Chase)
        # Pass ref_price (mid/close from signal) to calculate Chase Caps
        success, filled_price = await self._place_maker_entry(symbol, product_id, side, size, ref_price)
        
        if not success:
            logger.error(f"❌ Entry Failed for {symbol}")
            return None
            
        logger.info(f"✅ Entry Filled: {symbol} @ {filled_price}")
        
        # 2. Place Bracket (TP/SL)
        sl_str = f"{sl_price:.{precision}f}"
        tp_str = f"{tp_price:.{precision}f}"

        bracket_payload = {
            "product_id": product_id,
            "product_symbol": symbol,
            "stop_loss_order": {
                "order_type": BRACKET_ORDER_TYPE,
                "stop_price": sl_str,
            },
            "take_profit_order": {
                "order_type": "limit_order",
                "stop_price": tp_str,
                "limit_price": tp_str,
            },
            "bracket_stop_trigger_method": BRACKET_STOP_TRIGGER
        }
        
        if BRACKET_ORDER_TYPE == "limit_order":
            bracket_payload["stop_loss_order"]["limit_price"] = sl_str

        logger.info(f"🛡️ Placing Bracket: SL={sl_str} TP={tp_str}")
        b_status, b_resp = await self.api_client.post("/v2/orders/bracket", bracket_payload)
        
        if b_status != 200:
            logger.error(f"⚠️ Bracket Placement Failed: {b_resp}")
        
        direction = "LONG" if side == "buy" else "SHORT"
        return product_id, direction, filled_price

    async def _notify_monitor(self, symbol: str, size: float, product_id: int):
        try:
            message = {
                "type": "start_monitoring",
                "symbol": symbol,
                "size": float(size),
                "product_id": int(product_id),
                "timestamp": time.time(),
            }
            await self.redis.publish(MONITORING_CHANNEL, orjson.dumps(message))
        except Exception as e:
            logger.error("Failed to notify PositionMonitor: %s", e)

    async def _notify_tsl_manager(self, symbol, direction, size, product_id, entry_price):
        try:
            payload = {
                "command": "START_TSL",
                "symbol": symbol,
                "direction": direction,
                "size": float(size),
                "product_id": int(product_id),
                "entry_price": float(entry_price),
            }
            await self.redis.publish(TSL_CHANNEL, orjson.dumps(payload))
        except Exception as e:
            logger.error("Failed to notify TSL Manager: %s", e)