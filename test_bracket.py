
import logging
import asyncio
import sys
import os

# Adjust path to import modules
sys.path.append(os.getcwd())

from executor import Executor

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TEST]: %(message)s")
log = logging.getLogger("test_bracket")

def test_bracket():
    log.info("🧪 Starting Bracket Order Test...")
    
    # Init Executor
    exe = Executor()
    
    symbol = "BTCUSD"
    
    # 1. Sync Position to 0 (Cleaner)
    exe.sync_position(symbol, 0.0)
    
    # 2. Place Test Buy with Bracket
    log.info("🚀 Placing TEST BUY with Bracket...")
    exe.place_order(symbol, "BUY", 1, bracket=True)
    
    log.info("✅ Test Command Sent. Check Delta Exchange Dashboard for Bracket Orders.")

if __name__ == "__main__":
    test_bracket()
