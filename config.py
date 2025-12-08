# --- detla-bot/config.py ---
# COMPLETE PRODUCTION CONFIGURATION
# ✅ FIX: Widened TSL settings to prevent "suffocation"
# ✅ FIX: Added Activation Buffer for TSL
# ✅ FIX: Increased Confidence & Cooldown settings
# ✅ FIX: Defined 'TSL_TRAIL_AMOUNT' to resolve Pylance error

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
API_KEY = os.getenv("DELTA_API_KEY")
API_SECRET = os.getenv("DELTA_API_SECRET")
WS_URL = "wss://socket.india.delta.exchange"
DELTA_BASE_URL = "https://api.india.delta.exchange" 
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BINANCE_FUTURES_URL = "https://fapi.binance.com" 
DMS_ID = os.getenv("DMS_ID", "default_trading_bot_dms")

# ----------------------------------------------------------------------
# ✅ Spot Index Symbols
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
PUBLIC_SYMBOL_MAPPING = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT"
}
PRIORITY_LIST = ["SOLUSD", "ETHUSD", "BTCUSD"]

# ----------------------------------------------------------------------
# ✅ Risk Management Parameters
# ----------------------------------------------------------------------
# 🔧 FIX: Per-Symbol Sizing. 
BASE_POSITION_SIZE = {
    "BTCUSD": 1, 
    "ETHUSD": 1, 
    "SOLUSD": 1   
}

MAX_DRAWDOWN_PERCENT = 0.15  
MAX_DAILY_LOSS_PERCENT = 0.05 
MAX_CONCURRENT_TRADES = 3    
MAX_POSITION_SIZE = 0.1      

# 🛑 GATEKEEPER SETTINGS
GATEKEEPER_ENABLED = True
GATEKEEPER_VOL_THRESHOLD = 0.25  
GATEKEEPER_VOLATILITY_MIN = 0.0005 

# 🧠 Position Sizing
ENABLE_SMART_SIZING = False     
MIN_SIZE_MULTIPLIER = 1.0      
MAX_SIZE_MULTIPLIER = 1.0      
CONFIDENCE_FLOOR = 0.65
CONFIDENCE_CEILING = 0.90

# ----------------------------------------------------------------------
# ✅ Smart TP/SL & R/R Parameters 
# ----------------------------------------------------------------------
ATR_TIMEFRAME = "5m"       
SL_ATR_MULTIPLIER = 2.5     # ⬆️ INCREASED: Give trade room to breathe (was 2.0)
TP_BUFFER_PCT = 0.001      
MIN_RISK_REWARD_RATIO = 1.5 

# ----------------------------------------------------------------------
# ✅ Model & Training Config
# ----------------------------------------------------------------------
ATR_LABEL_MULTIPLIER = 1.5     
LAG_PERIODS = [1, 3, 5]        
USE_STACKING_ENSEMBLE = True 

# ----------------------------------------------------------------------
# ✅ Trailing Stop Loss (TSL) Parameters - 🔧 CRITICAL FIXES
# ----------------------------------------------------------------------
TSL_ENABLED = True
TSL_CHECK_INTERVAL = 5     
TSL_CHANNEL = "delta:tsl_control" 

# 🔧 FIX: Relaxed TSL to match Initial SL (Prevents immediate stop out)
TSL_ATR_MULTIPLIER = 2.0     # ⬆️ INCREASED from 1.0 to 2.0
TSL_MIN_TRAIL_AMOUNT = 0.5   
TSL_TRAIL_AMOUNT = 2.0       # ✅ FIXED: Added missing variable definition

# 🔧 FIX: Activation Buffer (Wait for 0.5% profit before trailing starts)
TSL_ACTIVATION_PCT = 0.005   # ✅ NEW: Prevents TSL from killing trade at entry

# ----------------------------------------------------------------------
# ✅ Dynamic Confidence Strategy
# ----------------------------------------------------------------------
DYNAMIC_CONFIDENCE_ENABLED = True
BASE_CONFIDENCE = 0.65        # ⬆️ INCREASED from 0.55/0.50 to filter noise
MIN_CONFIDENCE = 0.60          
VOLATILITY_SCALER = 2.0        

# ----------------------------------------------------------------------
# ✅ Mean Reversion Strategy
# ----------------------------------------------------------------------
MEAN_REVERSION_ENABLED = True
MR_BB_LENGTH = 20              
MR_BB_STD = 2.0                
MR_RSI_OVERSOLD = 30           
MR_RSI_OVERBOUGHT = 70         
MR_KER_THRESHOLD = 0.25        
MR_RISK_REWARD = 1.2           

# ----------------------------------------------------------------------
# ✅ Data Granularity (Volume Bars)
# ----------------------------------------------------------------------
VOLUME_BAR_SIZE = {
    "BTCUSD": 5.0,    
    "ETHUSD": 50.0,   
    "SOLUSD": 500.0   
}
VOLUME_BAR_CHANNEL = "delta:volume_bars"

# ----------------------------------------------------------------------
# ✅ Heuristic Strategy Parameters
# ----------------------------------------------------------------------
OBI_THRESHOLD = 0.3  
TFI_THRESHOLD = 0.1  
SIGNAL_CONFIDENCE = BASE_CONFIDENCE 

# ----------------------------------------------------------------------
# ✅ Strategy Filters
# ----------------------------------------------------------------------
TREND_CHECK_ENABLED = True
TREND_TIMEFRAME = "1h" 
FUNDING_CHECK_ENABLED = True
FUNDING_RATE_THRESHOLD = 0.0005 
VOLUME_CHECK_ENABLED = True
VOLUME_TIMEFRAME = "5m"       
VOLUME_SMA_PERIOD = 20         
VOLUME_SURGE_MULTIPLIER = 2.0 
SNR_CHECK_ENABLED = True
SNR_PROXIMITY_PCT = 0.002 

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
USER_AGENT = "DeltaInstitutionalBot/2.0"
WS_RECONNECT_BASE = 3
WS_RECONNECT_MAX = 60
WS_HEARTBEAT_INTERVAL = 30
RATE_LIMIT_REST = 100
RATE_LIMIT_WS = 150
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0

# ----------------------------------------------------------------------
# ✅ Redis Channels & Data Management
# ----------------------------------------------------------------------
RAW_CHANNEL = "delta:raw:ws"
PRIVATE_CHANNEL = "delta:private:ws" 
ENRICHED_CHANNEL = "delta:enriched"
SIGNAL_CHANNEL = "delta:signals"
EXECUTION_CHANNEL = "delta:executions"
ERROR_CHANNEL = "delta:errors"
MONITORING_CHANNEL = "delta:monitoring"
CONTROL_CHANNEL = "delta:control" 
TSL_CHANNEL = "delta:tsl_control" 
REDIS_POSITION_LOCK_PREFIX = "active_position:" 
LATEST_ENRICHED_KEY = "latest:enriched:" 
HEALTH_CHECK_KEY_FE = "health:fe:last_ts"
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
# ✅ Config Dictionary
# ----------------------------------------------------------------------
config = {
    "BASE_POSITION_SIZE": BASE_POSITION_SIZE, 
    "MIN_RISK_REWARD_RATIO": MIN_RISK_REWARD_RATIO,
    "ATR_TIMEFRAME": ATR_TIMEFRAME,
    "SL_ATR_MULTIPLIER": SL_ATR_MULTIPLIER,
    "TP_BUFFER_PCT": TP_BUFFER_PCT,
    "ATR_LABEL_MULTIPLIER": ATR_LABEL_MULTIPLIER,
    "LAG_PERIODS": LAG_PERIODS,
    "USE_STACKING_ENSEMBLE": USE_STACKING_ENSEMBLE,
    "DYNAMIC_CONFIDENCE_ENABLED": DYNAMIC_CONFIDENCE_ENABLED,
    "BASE_CONFIDENCE": BASE_CONFIDENCE,
    "MIN_CONFIDENCE": MIN_CONFIDENCE,
    "VOLATILITY_SCALER": VOLATILITY_SCALER,
    "MEAN_REVERSION_ENABLED": MEAN_REVERSION_ENABLED,
    "MR_RSI_OVERSOLD": MR_RSI_OVERSOLD,
    "MR_RSI_OVERBOUGHT": MR_RSI_OVERBOUGHT,
    "MR_KER_THRESHOLD": MR_KER_THRESHOLD,
    "MR_RISK_REWARD": MR_RISK_REWARD,
    "VOLUME_BAR_SIZE": VOLUME_BAR_SIZE,
    "ENABLE_SMART_SIZING": ENABLE_SMART_SIZING,
    "GATEKEEPER_ENABLED": GATEKEEPER_ENABLED,
    "GATEKEEPER_VOL_THRESHOLD": GATEKEEPER_VOL_THRESHOLD,
    "GATEKEEPER_VOLATILITY_MIN": GATEKEEPER_VOLATILITY_MIN,
    "TSL_ENABLED": TSL_ENABLED,
    "TSL_TRAIL_AMOUNT": TSL_TRAIL_AMOUNT, # ✅ Now defined
    "TSL_CHECK_INTERVAL": TSL_CHECK_INTERVAL,
    "TSL_CHANNEL": TSL_CHANNEL,
    "TSL_ATR_MULTIPLIER": TSL_ATR_MULTIPLIER, 
    "TSL_MIN_TRAIL_AMOUNT": TSL_MIN_TRAIL_AMOUNT,
    "TSL_ACTIVATION_PCT": TSL_ACTIVATION_PCT, 
    "OBI_THRESHOLD": OBI_THRESHOLD,
    "TFI_THRESHOLD": TFI_THRESHOLD,
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
    "BRACKET_STOP_TRIGGER": BRACKET_STOP_TRIGGER,
    "USER_AGENT": USER_AGENT,
    "LOG_LEVEL": LOG_LEVEL, 
    "API_MAX_RETRIES": API_MAX_RETRIES,
    "API_RETRY_DELAY": API_RETRY_DELAY,
    "CONFIDENCE_FLOOR": CONFIDENCE_FLOOR,
    "CONFIDENCE_CEILING": CONFIDENCE_CEILING,
    "MIN_SIZE_MULTIPLIER": MIN_SIZE_MULTIPLIER,
    "MAX_SIZE_MULTIPLIER": MAX_SIZE_MULTIPLIER
}