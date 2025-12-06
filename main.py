# --- detla-bot/main.py ---
# ✅ FIXED: Redis Keepalive enabled to prevent Windows 10054 Connection Reset errors
# ✅ FIXED: Health Check Server included

import asyncio
import signal
import logging
import logging.handlers
import os
import aiohttp
from aiohttp import web
from redis import asyncio as aioredis
import queue
import time

from ws_manager import WebSocketManager
from feature_engine import FeatureEngine
from ml_strategy import MLForecastingStrategy
from executor import OrderExecutionManager
from monitor import PositionMonitor
from risk_manager import RiskManager
from trailing_stop_manager import TrailingStopManager

from utils.api_client import DeltaAPIClient
from reconciler import StateReconciler
from utils.signing import sync_time_offset
from config import REDIS_URL, config, API_KEY, API_SECRET, HEALTH_CHECK_KEY_FE

# --- Systematic Logging Setup ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "bot.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")

console_level = config.get("LOG_LEVEL", "INFO")
console_handler = logging.StreamHandler()
console_handler.setLevel(console_level)
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(log_file_path, mode='a', encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
root_logger.addHandler(queue_handler)

listener = logging.handlers.QueueListener(log_queue, console_handler, file_handler)

logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aioredis").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("main")

# --- Health Check ---
async def health_check_handler(request: web.Request) -> web.Response:
    try:
        components = request.app['components']
        redis_client = components['redis']
        ws_manager = components['ws_manager']
        
        await redis_client.ping()
        
        if not ws_manager.is_authenticated:
            raise Exception("WebSocket is not authenticated.")
            
        last_fe_ts_str = await redis_client.get(HEALTH_CHECK_KEY_FE)
        if last_fe_ts_str:    
            last_fe_ts = int(last_fe_ts_str)
            if (time.time() * 1_000_000 - last_fe_ts) > (5 * 60 * 1_000_000):
                raise Exception(f"FeatureEngine data is stale.")
        return web.json_response({"status": "healthy"}, status=200)
    except Exception as e:
        return web.json_response({"status": "unhealthy", "reason": str(e)}, status=503)

async def start_health_server(components: dict) -> web.AppRunner:
    app = web.Application()
    app['components'] = components 
    app.router.add_get("/healthz", health_check_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 9090) 
    await site.start()
    logger.info("🩺 Health check server started on http://0.0.0.0:9090/healthz")
    return runner

async def run_bot():
    logger.info("🚀 Starting Delta Institutional Trading Bot")
    assert API_KEY is not None and API_SECRET is not None, "FATAL: Credentials missing."
    
    listener.start()
    logger.info("📄 Log listener started for thread-safe logging.")

    redis_client = None
    http_session = None
    health_runner = None 
    reconciler = None 
    tasks = [] 
    
    try:
        # ✅ FIX APPLIED: Socket Keepalive + Health Check Interval
        # This forces Windows to keep the TCP connection open.
        redis_client = await aioredis.from_url(
            REDIS_URL, 
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30
        )
        logger.info(f"📡 Connected to Redis at {REDIS_URL}")
        
        http_session = aiohttp.ClientSession()
        logger.info("🔗 HTTP ClientSession created")

        await sync_time_offset(http_session)
        api_client = DeltaAPIClient(http_session, API_KEY, API_SECRET)
        logger.info("🔐 Centralized DeltaAPIClient created")

        ws_manager = WebSocketManager(redis_client, http_session)
        feature_engine = FeatureEngine(redis_client, http_session) 
        
        risk_manager = RiskManager(redis_client)
        await risk_manager._load_state_from_redis() 
        await risk_manager.start() 

        strategy = MLForecastingStrategy(redis_client) 
        executor = OrderExecutionManager(redis_client, api_client, risk_manager)
        position_monitor = PositionMonitor(redis_client, api_client, risk_manager)
        tsl_manager = TrailingStopManager(redis_client, http_session, api_client) 
        reconciler = StateReconciler(redis_client, api_client)
        
        components = {"redis": redis_client, "ws_manager": ws_manager}
        health_runner = await start_health_server(components)

        tasks = [
            asyncio.create_task(ws_manager.start(), name="WebSocketHandler"),
            asyncio.create_task(feature_engine.start(), name="FeatureEngine"),
            asyncio.create_task(strategy.start(risk_manager), name="MLStrategy"), 
            asyncio.create_task(executor.start(), name="OrderExecutor"),
            asyncio.create_task(position_monitor.start(), name="PositionMonitor"),
            asyncio.create_task(tsl_manager.start(), name="TSLManager"), 
            asyncio.create_task(reconciler.start(), name="StateReconciler"),
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            if task.exception():
                logger.error(f"💥 Task {task.get_name()} crashed: {task.exception()}", exc_info=task.exception())

    except Exception as e:
        logger.error(f"💥 Fatal error in main startup: {e}", exc_info=True)
    finally:
        logger.info("🔻 Shutting down all services...")
        
        if reconciler: 
            await reconciler.stop()
            logger.info("🔻 Reconciler stopped")
        
        for task in tasks: 
            if not task.done(): task.cancel()
        if tasks: 
            await asyncio.gather(*tasks, return_exceptions=True)
            
        if health_runner: 
            await health_runner.cleanup() 
            logger.info("🔻 Health check server stopped")
            
        if http_session: 
            await http_session.close()
            logger.info("🔻 HTTP session closed")
            
        if redis_client: 
            await redis_client.aclose()
            logger.info("🔻 Redis connection closed")
        
        listener.stop()
        logger.info("📄 Log listener stopped.")
        logger.info("✅ All services stopped cleanly.")

async def supervisor():
    while True:
        await run_bot()
        logger.info("🟢 Bot exited. Restarting in 10s...")
        await asyncio.sleep(10)

async def shutdown_supervisor():
    logger.info("🔻 Received shutdown signal. Stopping supervisor...")
    for task in asyncio.all_tasks():
        if task.get_coro().__name__ == 'supervisor':
            task.cancel()

def handle_interrupt(sig, frame):
    signal_name = signal.Signals(sig).name 
    logger.info(f"🛑 Main loop stopped by user ({signal_name}).")
    asyncio.create_task(shutdown_supervisor())

if __name__ == "__main__":
    logger.info("Application starting up.")
    try:
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)
    except ValueError:
        logger.warning("⚠️ Signal handlers not supported on this platform. Use Ctrl+C to stop.")
    
    try: 
        asyncio.run(supervisor())
    except asyncio.CancelledError: 
        pass
    except Exception as e:
        logger.error(f"💥 Fatal error during application run: {e}", exc_info=True)
    finally:
        logger.info("Application shutting down.")