# --- executor.py ---
# UPDATED: Bracket Order placement + dynamic live-price SL/TP fallback
# UPDATED: Uses centralized DeltaAPIClient
# UPDATED: TSL Manager now trails the bracket SL child (via existing TSL manager)
# UPDATED: Auto-correct invalid TP/SL sides based on current entry anchor (mark/last)
# FIXED: Indentation bug in _get_product_info when adding symbol to cache
# NEW: Dynamic SL/TP based on SNR (support/resistance) + ATR + ML confidence (safe fallback to old 2%/3%)

import aiohttp
import asyncio
import json
import logging
import time
import urllib.parse
from typing import Optional, Any, Dict, Tuple, List

from redis import asyncio as aioredis

from config import (
    DELTA_BASE_URL,
    API_KEY,
    API_SECRET,
    SIGNAL_CHANNEL,
    MONITORING_CHANNEL,
    USER_AGENT,
    TRADING_SYMBOLS,
    DMS_ID,
    TSL_CHANNEL,
    TSL_ENABLED,
    config,
    BRACKET_STOP_TRIGGER,
    BRACKET_ORDER_TYPE,
)
from utils.api_client import DeltaAPIClient
from risk_manager import RiskManager

# Optional import of FeatureEngine for SNR/ATR (code works even if not present)
try:
    from feature_engine import FeatureEngine  # type: ignore
    _FEATURE_ENGINE_AVAILABLE = True
except Exception:
    FeatureEngine = None  # type: ignore
    _FEATURE_ENGINE_AVAILABLE = False

logger = logging.getLogger("executor")


class OrderExecutionManager:
    REDIS_POSITION_LOCK_KEY = "active_position"
    REDIS_POSITION_LOCK_TTL = 60

    def __init__(self, redis_client: aioredis.Redis, api_client: DeltaAPIClient, risk_manager: RiskManager):
        # Shared clients
        self.redis = redis_client
        self.api_client = api_client
        self.session = api_client.session

        self._process_lock = asyncio.Lock()
        self.risk_manager = risk_manager

        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.dms_id = DMS_ID

        self.product_info_cache: Dict[str, Dict[str, Any]] = {}

        self._signal_task: Optional[asyncio.Task] = None
        logger.info("✅ OrderExecutionManager initialized (using DeltaAPIClient).")

    # ---------------------------
    # Lifecycle
    # ---------------------------
    async def start(self):
        """Main signal-consumer loop."""
        logger.info("▶️ OrderExecutionManager starting (listening for signals)...")
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(SIGNAL_CHANNEL)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    signal = json.loads(msg["data"])
                except Exception:
                    logger.warning("Ignoring non-JSON signal: %s", msg.get("data"))
                    continue
                asyncio.create_task(self._handle_signal(signal))
        except asyncio.CancelledError:
            logger.info("OrderExecutionManager cancelled.")
        finally:
            await pubsub.unsubscribe(SIGNAL_CHANNEL)

    async def close(self):
        logger.info("🔒 Executor connections closed by main.")

    # ---------------------------
    # Helpers: product & price
    # ---------------------------
    async def _get_product_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches product details (id, tick_size, precision) by symbol and caches them.
        GET /v2/products/{symbol}
        """
        if symbol in self.product_info_cache:
            return self.product_info_cache[symbol]

        path = f"/v2/products/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"

        try:
            async with self.session.get(url, headers={'User-Agent': USER_AGENT}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    product = data.get("result", {})
                    if product:
                        product_id = product.get("id")
                        tick_size_str = product.get("tick_size")
                        if product_id and tick_size_str:
                            try:
                                precision = len(str(tick_size_str).split(".")[-1]) if "." in str(tick_size_str) else 0
                                info = {
                                    "id": int(product_id),
                                    "tick_size": float(tick_size_str),
                                    "precision": int(precision),
                                }
                                # also expose symbol if present
                                if product.get("symbol"):
                                    info["symbol"] = product.get("symbol")
                                self.product_info_cache[symbol] = info
                                return info
                            except Exception as ve:
                                logger.error("Invalid tick_size for %s: %s | %s", symbol, tick_size_str, ve)
                                return None
                logger.error("❌ Product info not found for %s (HTTP %s)", symbol, resp.status)
                return None
        except Exception as e:
            logger.error("❌ Error fetching product info for %s: %s", symbol, e, exc_info=True)
            return None

    async def _get_ticker_prices(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch current mark_price and last traded ('close') price.
        GET /v2/tickers/{symbol}
        Returns (mark_price, last_price)
        """
        path = f"/v2/tickers/{symbol}"
        url = f"{DELTA_BASE_URL}{path}"
        try:
            async with self.session.get(url, headers={'User-Agent': USER_AGENT}) as resp:
                data = await resp.json()
                if resp.status == 200 and data.get("success"):
                    res = data.get("result", {})
                    mark_price = res.get("mark_price")
                    last_price = res.get("close")
                    return (
                        float(mark_price) if mark_price is not None else None,
                        float(last_price) if last_price is not None else None,
                    )
                logger.error("❌ Ticker fetch failed for %s (HTTP %s): %s", symbol, resp.status, data)
                return (None, None)
        except Exception as e:
            logger.error("❌ Error fetching ticker for %s: %s", symbol, e)
            return (None, None)

    # ---------------------------
    # Dynamic SL/TP (SNR + ATR + confidence)
    # ---------------------------
    def _detect_snr_levels(self, candles: List[Dict[str, float]], lookback: int = 60) -> Tuple[List[float], List[float]]:
        """
        Simple swing-high / swing-low detector on recent candles to produce support/resistance levels.
        Non-intrusive: if candles missing/too short, returns empty lists.
        """
        if not candles or len(candles) < max(lookback, 7):
            return [], []

        seg = candles[-lookback:]
        highs = [float(c["high"]) for c in seg]
        lows = [float(c["low"]) for c in seg]

        resistances: List[float] = []
        supports: List[float] = []

        # local peaks/troughs with a 2-candle neighborhood
        for i in range(2, len(seg) - 2):
            h = highs[i]
            l = lows[i]
            if highs[i - 2] < h > highs[i + 2]:
                resistances.append(h)
            if lows[i - 2] > l < lows[i + 2]:
                supports.append(l)

        # Deduplicate (basic) and sort
        resistances = sorted(set(resistances))
        supports = sorted(set(supports))
        return supports, resistances

    def _safe_get_latest_candles(self, symbol: str) -> List[Dict[str, float]]:
        if not _FEATURE_ENGINE_AVAILABLE:
            return []
        # Accept either static or instance-style getters if user implemented differently
        try:
            if hasattr(FeatureEngine, "get_latest_candles"):
                return list(FeatureEngine.get_latest_candles(symbol) or [])
        except Exception as e:
            logger.debug("get_latest_candles not available or failed: %s", e)
        return []

    def _safe_get_latest_atr(self, symbol: str, default_from_price: float) -> float:
        if not _FEATURE_ENGINE_AVAILABLE:
            return max(default_from_price * 0.005, 0.0001)  # 0.5% fallback
        try:
            if hasattr(FeatureEngine, "get_latest_atr"):
                v = FeatureEngine.get_latest_atr(symbol)
                if v is not None and v > 0:
                    return float(v)
        except Exception as e:
            logger.debug("get_latest_atr not available or failed: %s", e)
        # safe fallback
        return max(default_from_price * 0.005, 0.0001)

    def _try_dynamic_sl_tp(
        self,
        symbol: str,
        direction: str,
        entry: float,
        confidence: float,
        precision: int,
    ) -> Optional[Tuple[float, float]]:
        """
        Compute dynamic SL/TP using SNR + ATR + confidence.
        Returns (sl_price, tp_price) or None if data insufficient.
        Never throws; purely additive—doesn't change legacy flow if it fails.
        """
        candles = self._safe_get_latest_candles(symbol)
        if not candles:
            return None

        atr = self._safe_get_latest_atr(symbol, entry)

        # Base buffers from ATR; scale with confidence
        # Lower confidence => tighter TP, slightly wider SL
        tp_mult = 2.5 if confidence < 0.8 else 3.0
        sl_mult = 1.5 if confidence < 0.6 else 1.0

        tp_buffer = atr * tp_mult
        sl_buffer = atr * sl_mult

        supports, resistances = self._detect_snr_levels(candles, lookback=60)

        if direction == "LONG":
            # nearest levels around entry
            higher_res = [r for r in resistances if r > entry]
            lower_sup = [s for s in supports if s < entry]
            nearest_res = min(higher_res) if higher_res else entry + tp_buffer
            nearest_sup = max(lower_sup) if lower_sup else entry - sl_buffer

            tp_price = min(nearest_res, entry + tp_buffer)
            sl_price = max(nearest_sup, entry - sl_buffer)

        else:  # SHORT
            lower_res = [r for r in resistances if r < entry]
            higher_sup = [s for s in supports if s > entry]
            nearest_res_below = max(lower_res) if lower_res else entry - tp_buffer
            nearest_sup_above = min(higher_sup) if higher_sup else entry + sl_buffer

            tp_price = max(nearest_res_below, entry - tp_buffer)
            sl_price = min(nearest_sup_above, entry + sl_buffer)

        sl_price = round(float(sl_price), precision)
        tp_price = round(float(tp_price), precision)

        # Sanity: ensure correct orientation; if not, return None to let legacy flow handle
        if direction == "LONG" and not (sl_price < entry < tp_price):
            return None
        if direction == "SHORT" and not (tp_price < entry < sl_price):
            return None

        logger.info(
            "🤖 Dynamic SL/TP for %s via SNR+ATR (conf=%.2f): SL=%.*f TP=%.*f (ATR=%.6f)",
            symbol,
            confidence,
            precision,
            sl_price,
            precision,
            tp_price,
            atr,
        )
        return sl_price, tp_price

    # ---------------------------
    # Redis position lock
    # ---------------------------
    async def _acquire_position_lock(self, symbol: str, timeout: int = 5) -> bool:
        deadline = time.time() + timeout
        lock_value = json.dumps({"symbol": symbol, "ts": time.time()})
        while time.time() < deadline:
            ok = await self.redis.set(
                self.REDIS_POSITION_LOCK_KEY,
                lock_value,
                ex=self.REDIS_POSITION_LOCK_TTL,
                nx=True,
            )
            if ok:
                logger.info("🔒 Acquired distributed lock for %s", symbol)
                return True
            await asyncio.sleep(0.25)
        logger.warning("⚠️ Could not acquire lock for %s", symbol)
        return False

    async def _release_position_lock(self):
        try:
            await self.redis.delete(self.REDIS_POSITION_LOCK_KEY)
            logger.info("🔓 Released distributed position lock")
        except Exception as e:
            logger.error("❌ Error releasing lock: %s", e)

    # ---------------------------
    # Public signal handler
    # ---------------------------
    async def _handle_signal(self, signal: dict):
        """
        Expected signal keys (typical):
          symbol, direction ("LONG"/"SHORT"), size_hint (contracts), sl_price, tp_price
        """
        symbol = signal.get("symbol")
        direction = signal.get("direction")
        size_hint = signal.get("size_hint", 0)

        if not symbol or not direction or not size_hint:
            logger.warning("Ignoring invalid signal: %s", signal)
            return

        ok, info = await self.risk_manager.validate_signal(signal)
        if not ok:
            logger.warning("Signal rejected by RiskManager: %s", info)
            return

        side = "buy" if direction == "LONG" else "sell"

        if not await self._acquire_position_lock(symbol):
            logger.warning("Another position in progress; skipping signal for %s", symbol)
            return

        try:
            # Calculate prices: prefer signal-provided SL/TP; else compute from live price (fallback)
            product_info = await self._get_product_info(symbol)
            if not product_info:
                logger.error("Missing product info for %s", symbol)
                return

            precision = int(product_info["precision"])
            product_id = int(product_info["id"])

            # get live prices
            mark_price, last_price = await self._get_ticker_prices(symbol)
            # choose anchor for checks/fallback calc
            anchor_price = mark_price if BRACKET_STOP_TRIGGER == "mark_price" else last_price

            # Get SL/TP from signal or compute
            sl_price = signal.get("sl_price")
            tp_price = signal.get("tp_price")
            confidence = float(signal.get("confidence", 0.7))

            # ---- NEW: try dynamic computation first (without breaking old behavior) ----
            if (sl_price is None or tp_price is None) and anchor_price is not None:
                dyn = self._try_dynamic_sl_tp(
                    symbol=symbol,
                    direction=direction,
                    entry=float(anchor_price),
                    confidence=confidence,
                    precision=precision,
                )
                if dyn is not None:
                    sl_price, tp_price = dyn

            # Legacy fallback (unchanged) if still missing
            if sl_price is None or tp_price is None:
                # Fallback logic similar to your PowerShell example: 2% SL / 3% TP
                if anchor_price is None:
                    logger.error("Cannot compute fallback SL/TP: no live anchor price for %s", symbol)
                    return
                entry_reference = float(anchor_price)

                sl_pct = 0.02  # 2%
                tp_pct = 0.03  # 3%

                if direction == "LONG":
                    sl_price = entry_reference * (1.0 - sl_pct)
                    tp_price = entry_reference * (1.0 + tp_pct)
                else:
                    sl_price = entry_reference * (1.0 + sl_pct)
                    tp_price = entry_reference * (1.0 - tp_pct)

                sl_price = round(sl_price, precision)
                tp_price = round(tp_price, precision)
                logger.info(
                    "🧮 Fallback SL/TP computed from live %s: SL=%s TP=%s",
                    "mark_price" if BRACKET_STOP_TRIGGER == "mark_price" else "last_traded_price",
                    sl_price,
                    tp_price,
                )

            # ✅ AUTO-CORRECT invalid TP/SL sides before bracket placement (kept)
            if anchor_price is not None:
                entry = float(anchor_price)
                if direction == "LONG" and not (sl_price < entry < tp_price):
                    logger.warning(
                        "⚠️ Invalid LONG bracket detected (SL>=Entry or TP<=Entry). Auto-fixing around entry."
                    )
                    sl_price = round(entry * 0.98, precision)
                    tp_price = round(entry * 1.03, precision)
                elif direction == "SHORT" and not (tp_price < entry < sl_price):
                    logger.warning(
                        "⚠️ Invalid SHORT bracket detected (TP>=Entry or SL<=Entry). Auto-fixing around entry."
                    )
                    tp_price = round(entry * 0.97, precision)
                    sl_price = round(entry * 1.02, precision)

            # Place entry + bracket (with fallback to standalone SL if needed)
            res = await self._place_linked_orders(
                symbol, side, float(size_hint), float(tp_price), float(sl_price)
            )
            if not res:
                logger.error("Order placement failed for %s", symbol)
                return

            product_id, ret_direction = res

            # Notify monitor & TSL manager
            # Entry price basis: use the best available live price
            entry_price = anchor_price if anchor_price is not None else (mark_price or last_price or 0.0)
            entry_price = float(entry_price or 0.0)

            await self._notify_monitor(symbol, float(size_hint), product_id)
            if TSL_ENABLED:
                await self._notify_tsl_manager(symbol, ret_direction, float(size_hint), product_id, entry_price)

            logger.info("✅ Signal executed for %s (size=%s)", symbol, size_hint)

        except Exception as e:
            logger.error("❌ Error handling signal: %s", e, exc_info=True)
        finally:
            await self._release_position_lock()

    # ---------------------------
    # Core order placement
    # ---------------------------
    async def _send_order(self, method: str, path: str, payload: dict) -> dict:
        """Helper to map order sending to the centralized API client."""
        method = method.upper()
        if method == "POST":
            status, data = await self.api_client.post(path, payload)
        elif method == "PUT":
            status, data = await self.api_client.put(path, payload)
        else:
            status, data = await self.api_client.get(path, params=payload)

        if status == 200:
            return data
        return {"success": False, "error": data}

    async def _place_linked_orders(
        self,
        symbol: str,
        side: str,
        size: float,
        tp_price: float,
        sl_price: float
    ) -> Optional[Tuple[int, str]]:
        """
        Places:
          1) Market entry
          2) Bracket (TP + SL) attached to position, with config-driven trigger & order types
        Falls back to standalone SL if bracket fails.
        Returns (product_id, direction) on success.
        """
        product_info = await self._get_product_info(symbol)
        if not product_info:
            logger.error(f"❌ Product Info missing for {symbol}. Blocking trade.")
            return None

        product_id = int(product_info["id"])
        precision = int(product_info["precision"])
        direction = "LONG" if side == "buy" else "SHORT"

        # 1) Market Entry
        entry_payload = {
            "product_id": product_id,
            "size": abs(size),
            "side": side,
            "order_type": "market_order",
        }
        logger.info(f"📦 Placing Market Entry Order: {entry_payload}")
        entry_resp = await self._send_order("POST", "/v2/orders", entry_payload)
        if not entry_resp.get("success"):
            logger.error(f"❌ Market Entry Order failed: {entry_resp}")
            return None
        logger.info(f"🎯 Entry Order Placed. ID: {entry_resp.get('result', {}).get('id')}")

        # 2) Bracket order
        final_sl_price = round(float(sl_price), precision)
        final_tp_price = round(float(tp_price), precision)
        sl_str = f"{final_sl_price:.{precision}f}"
        tp_str = f"{final_tp_price:.{precision}f}"

        bracket_payload = {
            "product_id": product_id,
            "product_symbol": symbol,
            "stop_loss_order": {
                "order_type": BRACKET_ORDER_TYPE,      # "limit_order" or "market_order"
                "stop_price": sl_str,
            },
            "take_profit_order": {
                "order_type": "limit_order",
                "stop_price": tp_str,
                "limit_price": tp_str,
            },
            "bracket_stop_trigger_method": BRACKET_STOP_TRIGGER  # "mark_price" or "last_traded_price"
        }

        if BRACKET_ORDER_TYPE == "limit_order":
            bracket_payload["stop_loss_order"]["limit_price"] = sl_str

        logger.info(f"📦 Placing Bracket Order: {bracket_payload}")
        status, bracket_resp = await self.api_client.post("/v2/orders/bracket", bracket_payload)

        if status == 200 and bracket_resp and bracket_resp.get("success"):
            logger.info(f"✅ Bracket Order Placed. Resp: {bracket_resp.get('result')}")
        else:
            logger.error(f"❌ Bracket Order failed (HTTP {status}). Response: {bracket_resp}")
            logger.warning("↩️ Falling back to standalone Stop Market order to protect position.")
            sl_side = "sell" if side == "buy" else "buy"
            fallback_sl_payload = {
                "product_id": product_id,
                "size": abs(size),
                "side": sl_side,
                "order_type": "market_order",  # stop market
                "stop_price": sl_str,
                "reduce_only": True,
                "stop_order_type": "stop_loss_order",
            }
            logger.info(f"📦 Placing Standalone Stop Market Order (Fallback): {fallback_sl_payload}")
            sl_resp = await self._send_order("POST", "/v2/orders", fallback_sl_payload)
            if not sl_resp.get("success"):
                logger.error(f"❌ Fallback Stop Loss Order failed: {sl_resp}. WARNING: Position may be unprotected!")
            else:
                logger.info(f"🎯 Fallback Stop Loss Order Placed. ID: {sl_resp.get('result', {}).get('id')}")

        return product_id, direction

    # ---------------------------
    # Notifications
    # ---------------------------
    async def _notify_monitor(self, symbol: str, size: float, product_id: int):
        try:
            message = {
                "type": "position_opened",
                "symbol": symbol,
                "size": float(size),
                "product_id": int(product_id),
                "timestamp": time.time(),
            }
            await self.redis.publish(MONITORING_CHANNEL, json.dumps(message))
            logger.info("📢 Notified PositionMonitor: %s", message)
        except Exception as e:
            logger.error("Failed to notify PositionMonitor: %s", e)

    async def _notify_tsl_manager(self, symbol: str, direction: str, size: float, product_id: int, entry_price: float):
        """
        Kick off TSL on the bracket's SL child. The TSL Manager will:
         - locate the child SL order
         - preserve its order type (limit/market)
         - trail it using ATR distance
        """
        try:
            payload = {
                "command": "START_TSL",
                "symbol": symbol,
                "direction": direction,
                "size": float(size),
                "product_id": int(product_id),
                "entry_price": float(entry_price),
            }
            await self.redis.publish(TSL_CHANNEL, json.dumps(payload))
            logger.info("📢 Notified TSL Manager: %s", payload)
        except Exception as e:
            logger.error("Failed to notify TSL Manager: %s", e)
