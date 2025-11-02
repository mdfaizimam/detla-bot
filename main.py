import asyncio
import signal
import logging
import logging.handlers  # <-- Added for file rotation
import os                # <-- Added to create 'logs' directory
import aiohttp
from redis import asyncio as aioredis

from ws_manager import WebSocketManager
from feature_engine import FeatureEngine
from ml_strategy import MLForecastingStrategy
from executor import OrderExecutionManager
from monitor import PositionMonitor
from config import REDIS_URL, config

# --- Systematic Logging Setup ---

# 1. Define Log Directory and ensure it exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "bot.log")

# 2. Get the root logger
# We configure the root logger so all modules (executor, monitor, etc.) inherit this setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Set root level to DEBUG to capture everything

# 3. Create a consistent formatter
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)

# 4. Console Handler (prints to screen)
# Use the LOG_LEVEL from your config for the console
console_level = config.get("LOG_LEVEL", logging.INFO) # Use .get for safety
console_handler = logging.StreamHandler()
console_handler.setLevel(console_level)
console_handler.setFormatter(log_formatter)

# 5. File Handler (saves to file, rotates daily)
# This handler will write ALL messages (DEBUG and up) to the file
file_handler = logging.handlers.TimedRotatingFileHandler(
    log_file_path,
    when="midnight",  # Rotate at midnight
    backupCount=7,    # Keep 7 days of logs
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

# 6. Add both handlers to the root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# 7. Get the logger for this main file
logger = logging.getLogger("main")

# --- End of Logging Setup ---


async def run_bot():
    logger.info("🚀 Starting Delta Institutional Trading Bot")

    redis_client = None
    http_session = None
    tasks = [] 

    try:
        redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info(f"📡 Connected to Redis at {REDIS_URL}")
        
        http_session = aiohttp.ClientSession()
        logger.info("🔗 HTTP ClientSession created")

        # ✅ MODIFICATION: Pass http_session to FeatureEngine
        ws_manager = WebSocketManager(redis_client, http_session)
        feature_engine = FeatureEngine(redis_client, http_session) # <-- ADDED http_session
        strategy = MLForecastingStrategy(redis_client)
        executor = OrderExecutionManager(redis_client, http_session)
        position_monitor = PositionMonitor(redis_client, http_session)

        # Start all services concurrently
        tasks = [
            asyncio.create_task(ws_manager.start(), name="WebSocketHandler"),
            asyncio.create_task(feature_engine.start(), name="FeatureEngine"),
            asyncio.create_task(strategy.start(), name="MLStrategy"),
            asyncio.create_task(executor.start(), name="OrderExecutor"),
            asyncio.create_task(position_monitor.start(), name="PositionMonitor"),
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
            
        if http_session:
            await http_session.close()
            logger.info("🔻 HTTP session closed")
        if redis_client:
            await redis_client.aclose()
            logger.info("🔻 Redis connection closed")
            
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


# --- main.py (FIXED Loop Management) ---

# ... (Keep all functions above, including run_bot and supervisor)

async def shutdown_supervisor():
    """A helper to gracefully shut down the supervisor task."""
    logger.info("🔻 Received shutdown signal. Stopping supervisor...")
    # Cancel the supervisor task, which is the parent of everything
    for task in asyncio.all_tasks():
        if task.get_coro().__name__ == 'supervisor':
            task.cancel()

# --- main.py (Corrected Signal Handler) ---

def handle_interrupt(sig, frame):
    """Handles Ctrl+C and OS signals gracefully."""
    # ✅ FIX: Use signal.Signals to get the name from the integer signal number
    signal_name = signal.Signals(sig).name 
    logger.info(f"🛑 Main loop stopped by user ({signal_name}).")
    asyncio.create_task(shutdown_supervisor())

if __name__ == "__main__":
# ... (rest of the code remains the same)
    logger.info("Application starting up.")
    
    # Set up signal handlers gracefully for interrupt and termination
    try:
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)
    except ValueError:
        # This handles the Windows/Jupyter environment issue
        logger.warning("⚠️ Signal handlers not supported on this platform. Use Ctrl+C to stop.")
    
    try:
        # ✅ FIX: Use asyncio.run() to create and manage the loop
        asyncio.run(supervisor())
    except asyncio.CancelledError:
        # Expected exception when supervisor is
        pass
    except Exception as e:
        logger.error(f"💥 Fatal error during application run: {e}", exc_info=True)
    finally:
        logger.info("Application shutting down.")