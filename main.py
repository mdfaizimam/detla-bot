import asyncio
import signal
import logging
import logging.handlers  
import os                
import aiohttp
from redis import asyncio as aioredis
import queue 

from ws_manager import WebSocketManager
from feature_engine import FeatureEngine
from ml_strategy import MLForecastingStrategy
from executor import OrderExecutionManager
from monitor import PositionMonitor
from config import REDIS_URL, config

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
console_level = config.get("LOG_LEVEL", logging.INFO)
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


async def run_bot():
    logger.info("🚀 Starting Delta Institutional Trading Bot")

    redis_client = None
    http_session = None
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

        # The rest of your component initialization remains here.
        ws_manager = WebSocketManager(redis_client, http_session)
        feature_engine = FeatureEngine(redis_client, http_session) 
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