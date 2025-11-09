# --- main.py ---
# UPDATED: To instantiate and inject the new DeltaAPIClient
# UPDATED: Injects RiskManager into PositionMonitor for PnL reporting
# FIX: Calls risk_manager.start() to enable the daily reset loop
# ✅ NEW: Adds API key assertion on startup
# ✅ NEW: Calls server time sync on startup
# ✅ NEW: Adds /healthz endpoint for production monitoring

import asyncio
import signal
import logging
import logging.handlers  
import os                
import aiohttp
from aiohttp import web # ✅ NEW: Import for health check
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

# NEW: Import the centralized API client
from utils.api_client import DeltaAPIClient
# ✅ NEW: Import time sync
from utils.signing import sync_time_offset 
from config import REDIS_URL, config, API_KEY, API_SECRET, HEALTH_CHECK_KEY_FE

# --- Systematic Logging Setup (FIXED FOR THREAD-SAFETY) ---

# 1. Define Log Directory and ensure it exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "bot.log")

# 2. Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) 

# 3. Create a consistent formatter
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)

# 4. Create the target handlers (used by the listener)
console_level = config.get("LOG_LEVEL", "INFO")
console_handler = logging.StreamHandler()
console_handler.setLevel(console_level)
console_handler.setFormatter(log_formatter)

file_handler = logging.handlers.TimedRotatingFileHandler(
    log_file_path,
    when="midnight", 
    backupCount=7,    
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

# 5. Setup Queue and QueueHandler for thread-safe logging
log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
# Add ONLY the QueueHandler to the root logger
root_logger.addHandler(queue_handler)

# 6. Create the listener to process the queue in a separate thread
listener = logging.handlers.QueueListener(log_queue, console_handler, file_handler)

# 7. Get the logger for this main file
logger = logging.getLogger("main")

# --- End of Logging Setup ---

# ✅ --- NEW: Health Check Handler ---
async def health_check_handler(request: web.Request) -> web.Response:
    """
    Checks the health of all critical components.
    Returns HTTP 200 if healthy, HTTP 503 if unhealthy.
    """
    try:
        components = request.app['components']
        redis_client = components['redis']
        ws_manager = components['ws_manager']
        
        # 1. Check Redis Ping
        await redis_client.ping()
        
        # 2. Check WebSocket Authentication
        if not ws_manager.is_authenticated:
            raise Exception("WebSocket is not authenticated.")
            
        # 3. Check FeatureEngine (is it processing data?)
        last_fe_ts_str = await redis_client.get(HEALTH_CHECK_KEY_FE)
        if not last_fe_ts_str:
            raise Exception("FeatureEngine has not processed any data.")
            
        last_fe_ts = int(last_fe_ts_str)
        # Check if timestamp (in microseconds) is older than 5 minutes
        if (time.time() * 1_000_000 - last_fe_ts) > (5 * 60 * 1_000_000):
            raise Exception(f"FeatureEngine data is stale ({(time.time() * 1_000_000 - last_fe_ts) / 1_000_000:.0f}s old).")

        # All checks passed
        return web.json_response({"status": "healthy"}, status=200)

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        return web.json_response({"status": "unhealthy", "reason": str(e)}, status=503)

async def start_health_server(components: dict) -> web.AppRunner:
    """Initializes and starts the lightweight health check web server."""
    app = web.Application()
    app['components'] = components # Make components accessible to the handler
    app.router.add_get("/healthz", health_check_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    # 9090 is a common port for health checks
    site = web.TCPSite(runner, '0.0.0.0', 9090) 
    await site.start()
    logger.info("🩺 Health check server started on http://0.0.0.0:9090/healthz")
    return runner
# --- END HEALTH CHECK ---


async def run_bot():
    logger.info("🚀 Starting Delta Institutional Trading Bot")
    
    # ✅ --- FIX: Assert API keys are present ---
    assert API_KEY is not None and API_SECRET is not None, \
        "FATAL: DELTA_API_KEY and DELTA_API_SECRET not found in environment. Please check your .env file."
    logger.info("✅ API Credentials loaded.")
    # --- END FIX ---

    redis_client = None
    http_session = None
    health_runner = None # ✅ NEW
    tasks = [] 
    
    # 💥 FIX: Start the listener immediately 
    listener.start()
    logger.info("📄 Log listener started for thread-safe logging.")
    
    try:
        redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"📡 Connected to Redis at {REDIS_URL}")
        
        http_session = aiohttp.ClientSession()
        logger.info("🔗 HTTP ClientSession created")
        
        # ✅ --- FIX: Sync time with server on startup ---
        await sync_time_offset(http_session)
        # --- END FIX ---

        # --- NEW: Instantiate Centralized API Client ---
        api_client = DeltaAPIClient(http_session, API_KEY, API_SECRET)
        logger.info("🔐 Centralized DeltaAPIClient created")

        # --- Component Initialization and Startup (UPDATED) ---
        ws_manager = WebSocketManager(redis_client, http_session)
        feature_engine = FeatureEngine(redis_client, http_session) 
        
        risk_manager = RiskManager(redis_client)
        await risk_manager._load_state_from_redis() # Load equity state
        await risk_manager.start() # ✅ --- FIX: Start the daily reset loop ---

        strategy = MLForecastingStrategy(redis_client) 
        
        # UPDATED: Inject api_client
        executor = OrderExecutionManager(redis_client, api_client, risk_manager)
        
        # ✅ --- FIX: Inject RiskManager into PositionMonitor ---
        position_monitor = PositionMonitor(redis_client, api_client, risk_manager)
        # --- END FIX ---
        
        # UPDATED: Inject both session (for unauth) and api_client (for auth)
        tsl_manager = TrailingStopManager(redis_client, http_session, api_client) 

        # ✅ --- NEW: Start health check server ---
        components = {
            "redis": redis_client,
            "ws_manager": ws_manager
        }
        health_runner = await start_health_server(components)
        # --- END NEW ---

        # Start all services concurrently
        tasks = [
            asyncio.create_task(ws_manager.start(), name="WebSocketHandler"),
            asyncio.create_task(feature_engine.start(), name="FeatureEngine"),
            asyncio.create_task(strategy.start(risk_manager), name="MLStrategy"), # Pass risk_manager to start()
            asyncio.create_task(executor.start(), name="OrderExecutor"),
            asyncio.create_task(position_monitor.start(), name="PositionMonitor"),
            asyncio.create_task(tsl_manager.start(), name="TSLManager"), # NEW: Start TSL Manager
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            try:
                if task.exception():
                    logger.error(f"💥 Task {task.get_name()} crashed: {task.exception()}", exc_info=task.exception())
            except asyncio.CancelledError:
                logger.info(f"Task {task.get_name()} was cancelled.")

    except Exception as e:
        logger.error(f"💥 Fatal error in main startup: {e}", exc_info=True)
    finally:
        logger.info("🔻 Shutting down all services...")
        
        for task in tasks:
            if not task.done():
                task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        if health_runner:
            await health_runner.cleanup() # ✅ NEW
            logger.info("🔻 Health check server stopped")
            
        if http_session:
            await http_session.close()
            logger.info("🔻 HTTP session closed")
        if redis_client:
            await redis_client.aclose()
            logger.info("🔻 Redis connection closed")
            
        # 💥 FIX: Stop the listener thread cleanly after all tasks finish
        listener.stop()
        logger.info("📄 Log listener stopped.")
            
        logger.info("✅ All services stopped cleanly.")

async def supervisor():
    """Keeps the bot running; restarts automatically after clean exit."""
    try:
        while True:
            await run_bot()
            logger.info("🟢 Bot exited. Restarting in 10s...")
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        logger.info("🛑 Supervisor stopping.")
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"💥 Fatal error in supervisor: {e}")
        await asyncio.sleep(5)

async def shutdown_supervisor():
    """A helper to gracefully shut down the supervisor task."""
    logger.info("🔻 Received shutdown signal. Stopping supervisor...")
    # Cancel the supervisor task, which is the parent of everything
    for task in asyncio.all_tasks():
        if task.get_coro().__name__ == 'supervisor':
            task.cancel()

def handle_interrupt(sig, frame):
    """Handles Ctrl+C and OS signals gracefully."""
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