# --- config.py ---
# Complete Updated File (with Dynamic TSL Parameters)

import os
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# ✅ Base Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

# ----------------------------------------------------------------------
# ✅ Core Environment Config
# ----------------------------------------------------------------------
API_KEY = os.getenv("DELTA_API_KEY", "")
API_SECRET = os.getenv("DELTA_API_SECRET", "")
WS_URL = "wss://socket.india.delta.exchange"
DELTA_BASE_URL = "https://api.india.delta.exchange" 
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ----------------------------------------------------------------------
# ✅ Deadman Switch Configuration 
# ----------------------------------------------------------------------
DMS_ID = os.getenv("DMS_ID", "default_trading_bot_dms")

# ----------------------------------------------------------------------
# ✅ Spot Index Symbols (For v2/spot_price WS subscription)
# ----------------------------------------------------------------------
SPOT_INDEX_SYMBOLS = {
    "BTCUSD": ".DEXBTUSD",
    "ETHUSD": ".DEETHUSD", 
    "SOLUSD": ".DESOLUSD"
}

# ----------------------------------------------------------------------
# ✅ Trading Symbols & Markets
# ----------------------------------------------------------------------
TRADING_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]  
PRIORITY_LIST = ["SOLUSD", "ETHUSD", "BTCUSD"]

# ----------------------------------------------------------------------
# ✅ Risk Management Parameters
# ----------------------------------------------------------------------
BASE_POSITION_SIZE = 1       # Base position size in contracts (Static size: 1)
MAX_DRAWDOWN_PERCENT = 0.15  # 15% Max Portfolio Drawdown (for RiskManager)
MAX_DAILY_LOSS_PERCENT = 0.05 # 5% Max Daily Loss (for RiskManager)
MAX_CONCURRENT_TRADES = 1    

# ----------------------------------------------------------------------
# ✅ GLOBAL CONSTANTS (For use in the config dictionary calculation)
# ----------------------------------------------------------------------
MAX_POSITION_SIZE = 0.1      # 10% max position (used for reporting)

# ----------------------------------------------------------------------
# ✅ Smart TP/SL & R/R Parameters 
# ----------------------------------------------------------------------
ATR_TIMEFRAME = "5m"       # Use 5m ATR for volatility
SL_ATR_MULTIPLIER = 2.0    
TP_BUFFER_PCT = 0.001      
MIN_RISK_REWARD_RATIO = 1.5 

# ----------------------------------------------------------------------
# ✅ Trailing Stop Loss (TSL) Parameters (DYNAMIC UPDATE)
# ----------------------------------------------------------------------
TSL_ENABLED = True
TSL_TRAIL_AMOUNT = 2.00    # Static fallback/Original config (used as fallback)
TSL_CHECK_INTERVAL = 5     # Poll frequency in seconds
TSL_CHANNEL = "delta:tsl_control" 

TSL_ATR_MULTIPLIER = 1.0     # NEW: Multiplier applied to ATR for dynamic trail distance
TSL_MIN_TRAIL_AMOUNT = 0.5   # NEW: Minimum dollar value trail floor

# ----------------------------------------------------------------------
# ✅ Heuristic Strategy Parameters (Entry Signal)
# ----------------------------------------------------------------------
OBI_THRESHOLD = 0.3  
TFI_THRESHOLD = 0.1  
SIGNAL_CONFIDENCE = 0.9 

# ----------------------------------------------------------------------
# ✅ Heuristic Strategy Filters (All 4 Filters)
# ----------------------------------------------------------------------
# 1. Trend Filter
TREND_CHECK_ENABLED = True
TREND_TIMEFRAME = "1h" 

# 2. Funding Rate Filter
FUNDING_CHECK_ENABLED = True
FUNDING_RATE_THRESHOLD = 0.0005 # 0.05%

# 3. Volume Filter
VOLUME_CHECK_ENABLED = True
VOLUME_TIMEFRAME = "5m"       
VOLUME_SMA_PERIOD = 20         
VOLUME_SURGE_MULTIPLIER = 2.0 

# 4. S/R Filter
SNR_CHECK_ENABLED = True
SNR_PROXIMITY_PCT = 0.002 # 0.2%

# ----------------------------------------------------------------------
# ✅ Order Execution Parameters
# ----------------------------------------------------------------------
ORDER_TIMEOUT = 30
MAX_ORDER_RETRIES = 3
RETRY_DELAY = 2
BRACKET_STOP_TRIGGER = "last_traded_price"
BRACKET_ORDER_TYPE = "limit_order"

# ----------------------------------------------------------------------
# ✅ System & Connection Parameters
# ----------------------------------------------------------------------
USER_AGENT = "DeltaInstitutionalBot/1.0"
WS_RECONNECT_BASE = 3
WS_RECONNECT_MAX = 60
WS_HEARTBEAT_INTERVAL = 30
RATE_LIMIT_REST = 100
RATE_LIMIT_WS = 150

# ----------------------------------------------------------------------
# ✅ Redis Channels & Data Management
# ----------------------------------------------------------------------
RAW_CHANNEL = "delta:raw:ws"
ENRICHED_CHANNEL = "delta:enriched"
SIGNAL_CHANNEL = "delta:signals"
EXECUTION_CHANNEL = "delta:executions"
ERROR_CHANNEL = "delta:errors"
MONITORING_CHANNEL = "delta:monitoring"
CONTROL_CHANNEL = "delta:control" 
TSL_CHANNEL = "delta:tsl_control" 

LATEST_ENRICHED_KEY = "latest:enriched:" # NEW: Redis Key Prefix for caching FE output

REDIS_DATA_TTL = 3600
CACHE_EXPIRY = 300

# ----------------------------------------------------------------------
# ✅ Logging
# ----------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
LOG_PATH = BASE_DIR / "logs"
LOG_FILE = LOG_PATH / "bot.log"
AUDIT_LOG_FILE = LOG_PATH / "audit.log"
os.makedirs(LOG_PATH, exist_ok=True)

# ----------------------------------------------------------------------
# ✅ Derived Config Object (for easy passing)
# ----------------------------------------------------------------------
config = {
    # Risk
    "max_position_pct": MAX_POSITION_SIZE * 100,
    "BASE_POSITION_SIZE": BASE_POSITION_SIZE,
    "MIN_RISK_REWARD_RATIO": MIN_RISK_REWARD_RATIO,
    "PRODUCT_SPECS": {}, 

    # Smart TP/SL
    "ATR_TIMEFRAME": ATR_TIMEFRAME,
    "SL_ATR_MULTIPLIER": SL_ATR_MULTIPLIER,
    "TP_BUFFER_PCT": TP_BUFFER_PCT,
    
    # TSL (Dynamic & Static)
    "TSL_ENABLED": TSL_ENABLED,
    "TSL_TRAIL_AMOUNT": TSL_TRAIL_AMOUNT, 
    "TSL_CHECK_INTERVAL": TSL_CHECK_INTERVAL,
    "TSL_CHANNEL": TSL_CHANNEL,
    "TSL_ATR_MULTIPLIER": TSL_ATR_MULTIPLIER, 
    "TSL_MIN_TRAIL_AMOUNT": TSL_MIN_TRAIL_AMOUNT, 

    # Heuristic Params
    "OBI_THRESHOLD": OBI_THRESHOLD,
    "TFI_THRESHOLD": TFI_THRESHOLD,
    
    # Filter Params
    "TREND_CHECK_ENABLED": TREND_CHECK_ENABLED,
    "TREND_TIMEFRAME": TREND_TIMEFRAME,
    "FUNDING_CHECK_ENABLED": FUNDING_CHECK_ENABLED,
    "FUNDING_RATE_THRESHOLD": FUNDING_RATE_THRESHOLD,
    "VOLUME_CHECK_ENABLED": VOLUME_CHECK_ENABLED,
    "VOLUME_TIMEFRAME": VOLUME_TIMEFRAME,
    "VOLUME_SMA_PERIOD": VOLUME_SMA_PERIOD,
    "VOLUME_SURGE_MULTIPLIER": VOLUME_SURGE_MULTIPLIER,
    "SNR_CHECK_ENABLED": SNR_CHECK_ENABLED,
    "SNR_PROXIMITY_PCT": SNR_PROXIMITY_PCT,

    # Execution Params
    "BRACKET_STOP_TRIGGER": BRACKET_STOP_TRIGGER,
    "USER_AGENT": USER_AGENT,
    
    "LOG_LEVEL": LOG_LEVEL, 
}

# ----------------------------------------------------------------------
# ✅ Startup Prints
# ----------------------------------------------------------------------
print(f"✅ Configuration loaded successfully")
print(f"📊 Trading Symbols: {TRADING_SYMBOLS}")
print(f"🎯 Priority List: {' -> '.join(PRIORITY_LIST)}")
print(f"🛡️ Risk Management: {MAX_DRAWDOWN_PERCENT*100}% max drawdown, {MAX_DAILY_LOSS_PERCENT*100}% daily loss")
print(f"🤖 Strategy: Heuristic (OBI > {OBI_THRESHOLD}, TFI > {TFI_THRESHOLD})")
print(f"--- FILTERS ---")
print(f"📈 Trend: {TREND_CHECK_ENABLED} on {TREND_TIMEFRAME} candle")
print(f"💵 Funding: {FUNDING_CHECK_ENABLED} (Threshold: {FUNDING_RATE_THRESHOLD * 100}%)")
print(f"🌊 Volume: {VOLUME_CHECK_ENABLED} ({VOLUME_TIMEFRAME} vol > {VOLUME_SURGE_MULTIPLIER}x SMA({VOLUME_SMA_PERIOD}))")
print(f"⛰️ S/R: {SNR_CHECK_ENABLED} (Avoid {SNR_PROXIMITY_PCT * 100}% proximity to Daily/Weekly/Monthly levels)")
print(f"⚖️ R/R: Required Ratio > {MIN_RISK_REWARD_RATIO}:1")
print(f"🎛️ Single Position Mode: Active (Max {MAX_CONCURRENT_TRADES} concurrent trade)")
print(f"🔀 Trailing Stop Loss: {'Enabled' if TSL_ENABLED else 'Disabled'} (Trail: {TSL_ATR_MULTIPLIER}x ATR, Min: {TSL_MIN_TRAIL_AMOUNT} USD)")