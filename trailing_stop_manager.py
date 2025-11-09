# --- trailing_stop_manager.py ---
# UPDATED: Trails the bracket SL child (preserves its order_type)
# UPDATED: Uses centralized DeltaAPIClient for signed calls
# UPDATED: Initializes best_price_seen from entry_price
# UPDATED: Live mark/last price fetching; dynamic ATR trail with floor

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
    BRACKET_STOP_TRIGGER,   # to select which price to trust conceptually
)
from utils.api_client import DeltaAPIClient

logger = logging.getLogger("tsl_manager")


class TrailingStopManager:
    def __init__(self, redis_client: aioredis.Redis, http_session: aiohttp.ClientSession, api_client: DeltaAPIClient):
        self.redis = redis_client
        self.session = http_session     # for unauth GETs
        self.api_client = api_client    # for signed GET/PUT

        self.tsl_tasks: Dict[int, asyncio.Task] = {}  # key: product_id

        # Dynamic TSL config
        self.tsl_config = {
            "trail_multiplier": config["TSL_ATR_MULTIPLIER"],
            "min_trail_amount": config["TSL_MIN_TRAIL_AMOUNT"],
            "check_interval": config["TSL_CHECK_INTERVAL"],
            "atr_timeframe": ATR_TIMEFRAME,
        }

        self._runner_task: Optional[asyncio.Task] = None

    # ---------------------------
    # Lifecycle
    # ---------------------------
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

                if channel == TSL_CHANNEL:
                    if data.get("command") == "START_TSL":
                        product_id = int(data["product_id"])
                        # prevent duplicate task per product
                        if product_id in self.tsl_tasks and not self.tsl_tasks[product_id].done():
                            logger.info("TSL already active for product_id=%s", product_id)
                            continue
                        task = asyncio.create_task(
                            self._trailing_loop(
                                product_id=product_id,
                                symbol=data["symbol"],
                                direction=data["direction"],
                                size=int(data["size"]) if float(data["size"]) == int(data["size"]) else float(data["size"]),
                                entry_price=float(data["entry_price"]),
                            ),
                            name=f"TSL-{product_id}"
                        )
                        self.tsl_tasks[product_id] = task

                elif channel == MONITORING_CHANNEL:
                    # stop TSL when position is closed
                    if data.get("type") == "position_closed":
                        # in a more connected system we'd also get product_id; here we stop all tasks on that symbol
                        symbol = data.get("symbol")
                        # Best-effort: cancel all tasks (safe if multiple symbols rarely overlap)
                        to_cancel = [pid for pid, t in self.tsl_tasks.items() if not t.done()]
                        for pid in to_cancel:
                            self.tsl_tasks[pid].cancel()
                            logger.info("🛑 Stopping TSL (position closed). product_id=%s", pid)
        except asyncio.CancelledError:
            logger.info("TSLManager cancelled.")
        finally:
            await pubsub.unsubscribe(TSL_CHANNEL, MONITORING_CHANNEL)

    async def close(self):
        for _, task in list(self.tsl_tasks.items()):
            if not task.done():
                task.cancel()

    # ---------------------------
    # Cached ATR from FeatureEngine (Redis)
    # ---------------------------
    async def _get_latest_atr(self, symbol: str) -> Optional[float]:
        """Fetch the latest ATR value from FeatureEngine’s LATEST_ENRICHED_KEY cache."""
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

    # ---------------------------
    # Live price
    # ---------------------------
    async def fetch_ticker_data(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch live mark & last traded price. Returns (mark_price, last_price)."""
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

    # ---------------------------
    # Orders: find & edit stop-loss child
    # ---------------------------
    async def fetch_open_stop_order_id(self, product_id: int) -> Optional[Tuple[int, str]]:
        """
        Return (order_id, order_type) for the active stop-loss child order
        created by the bracket or standalone SL.
        """
        path = "/v2/orders"
        params = {
            "product_ids": str(product_id),
            "states": "open,pending",
            "stop_order_type": "stop_loss_order",
        }
        status, data = await self.api_client.get(path, params=params)
        if status == 200 and data and data.get("success"):
            for order in data.get("result", []):
                if order.get("stop_order_type") == "stop_loss_order" and order.get("state") in ("open", "pending"):
                    oid = order.get("id")
                    otype = order.get("order_type") or "market_order"
                    logger.debug("Found SL child: id=%s type=%s stop=%s", oid, otype, order.get("stop_price"))
                    return int(oid), str(otype)
            logger.warning("No open Stop-Loss order found for product_id=%s. Waiting...", product_id)
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
        """Edit the stop order preserving its original order_type; set limit_price if needed."""
        path = "/v2/orders"
        # Determine side from position: for LONG (size>0) SL is sell; for SHORT SL is buy
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
        logger.error("❌ Failed to update stop order (HTTP %s): %s", status, resp)
        return False

    # ---------------------------
    # Core trailing loop
    # ---------------------------
    async def _trailing_loop(
        self,
        product_id: int,
        symbol: str,
        direction: str,
        size: int | float,
        entry_price: float
    ):
        """
        Continuous trailing stop logic:
          - wait for the stop-loss child to be visible
          - compute dynamic trail using ATR with floor
          - move stop only in favorable direction
          - preserve original order_type (limit/market)
        """
        trail_multiplier = float(self.tsl_config["trail_multiplier"])
        min_trail_amount = float(self.tsl_config["min_trail_amount"])
        check_interval = float(self.tsl_config["check_interval"])

        best_price_seen: float = float(entry_price)
        sl_tuple: Optional[Tuple[int, str]] = None  # (order_id, order_type)

        logger.info("TSL Loop for %s started @ entry=%.4f (trail x%.2f floor=%.4f)",
                    symbol, best_price_seen, trail_multiplier, min_trail_amount)

        # Wait for SL child order
        for attempt in range(10):
            sl_tuple = await self.fetch_open_stop_order_id(product_id)
            if sl_tuple:
                break
            logger.info("Waiting for SL child (attempt %d/10)...", attempt + 1)
            await asyncio.sleep(1.0)

        if not sl_tuple:
            logger.error("❌ No SL child found; cannot trail %s", symbol)
            return

        stop_order_id, stop_order_type = sl_tuple
        logger.info("TSL active for %s: order_id=%s type=%s", symbol, stop_order_id, stop_order_type)

        while True:
            try:
                mark_price, last_price = await self.fetch_ticker_data(symbol)
                live_price = mark_price if BRACKET_STOP_TRIGGER == "mark_price" else last_price

                if live_price is None:
                    await asyncio.sleep(check_interval)
                    continue

                # Trail distance: ATR*multiplier with minimum floor
                latest_atr = await self._get_latest_atr(symbol)
                if latest_atr is not None and latest_atr > 0:
                    trail_amt = max(latest_atr * trail_multiplier, min_trail_amount)
                else:
                    trail_amt = min_trail_amount

                if direction == "LONG":
                    if live_price > best_price_seen:
                        best_price_seen = live_price
                    new_stop = best_price_seen - trail_amt
                else:
                    if live_price < best_price_seen:
                        best_price_seen = live_price
                    new_stop = best_price_seen + trail_amt

                # Push update (preserve order type)
                await self.update_stop_price(
                    order_id=stop_order_id,
                    product_id=product_id,
                    size=size,
                    new_stop_price=new_stop,
                    order_type=stop_order_type,
                )

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                logger.info("TSL loop cancelled for %s", symbol)
                break
            except Exception as e:
                logger.error("❌ Error in TSL loop for %s: %s", symbol, e, exc_info=True)
                await asyncio.sleep(check_interval)
