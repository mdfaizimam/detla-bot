import asyncio
import signal
import logging
import aiohttp
from redis import asyncio as aioredis

from ws_manager import WebSocketManager
from feature_engine import FeatureEngine
from ml_strategy import MLForecastingStrategy
from executor import OrderExecutionManager
from monitor import PositionMonitor
from config import REDIS_URL, config

logging.basicConfig(
    level=config["LOG_LEVEL"] if config["LOG_LEVEL"] else logging.DEBUG, # <-- Use config dictionary
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("main")


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
        # Expected exception when supervisor is cancelled cleanly
        pass
    except Exception as e:
        logger.error(f"💥 Fatal error during application run: {e}", exc_info=True)
    finally:
        logger.info("Application shutting down.")