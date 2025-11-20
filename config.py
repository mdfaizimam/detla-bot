# --- detla-bot/config.py ---
# COMPLETE PRODUCTION CONFIGURATION
# ✅ Includes: Risk, Strategy, Filters, TSL, and System Settings
# ✅ NEW: Volume Bars, Mean Reversion, Dynamic Confidence

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
BASE_POSITION_SIZE = 1       
MAX_DRAWDOWN_PERCENT = 0.15  
MAX_DAILY_LOSS_PERCENT = 0.05 
MAX_CONCURRENT_TRADES = 1    
MAX_POSITION_SIZE = 0.1      

# ----------------------------------------------------------------------
# ✅ Smart TP/SL & R/R Parameters 
# ----------------------------------------------------------------------
ATR_TIMEFRAME = "5m"       
SL_ATR_MULTIPLIER = 2.0    
TP_BUFFER_PCT = 0.001      
MIN_RISK_REWARD_RATIO = 1.5 

# ----------------------------------------------------------------------
# ✅ Model & Training Config
# ----------------------------------------------------------------------
ATR_LABEL_MULTIPLIER = 1.5     
LAG_PERIODS = [1, 3, 5]        
USE_STACKING_ENSEMBLE = True 

# ----------------------------------------------------------------------
# ✅ Trailing Stop Loss (TSL) Parameters
# ----------------------------------------------------------------------
TSL_ENABLED = True
TSL_TRAIL_AMOUNT = 2.00    
TSL_CHECK_INTERVAL = 5     
TSL_CHANNEL = "delta:tsl_control" 
TSL_ATR_MULTIPLIER = 1.0     
TSL_MIN_TRAIL_AMOUNT = 0.5   

# ----------------------------------------------------------------------
# ✅ Dynamic Confidence Strategy (The "Recall" Fix)
# ----------------------------------------------------------------------
DYNAMIC_CONFIDENCE_ENABLED = True
BASE_CONFIDENCE = 0.90         # Target confidence for calm markets
MIN_CONFIDENCE = 0.65          # Floor confidence for exploding markets
VOLATILITY_SCALER = 2.0        # How aggressively to lower confidence

# ----------------------------------------------------------------------
# ✅ Mean Reversion Strategy (The "Chop" Fix)
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
    "BTCUSD": 5.0,    # Generate bar every 5 BTC traded
    "ETHUSD": 50.0,   # Generate bar every 50 ETH traded
    "SOLUSD": 500.0   # Generate bar every 500 SOL traded
}
VOLUME_BAR_CHANNEL = "delta:volume_bars"

# ----------------------------------------------------------------------
# ✅ Heuristic Strategy Parameters
# ----------------------------------------------------------------------
OBI_THRESHOLD = 0.3  
TFI_THRESHOLD = 0.1  
SIGNAL_CONFIDENCE = BASE_CONFIDENCE # Default fallback

# ----------------------------------------------------------------------
# ✅ Strategy Filters (All 4 Filters)
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
REDIS_POSITION_LOCK_KEY = "active_position"
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
# ✅ Derived Config Object
# ----------------------------------------------------------------------
config = {
    "BASE_POSITION_SIZE": BASE_POSITION_SIZE,
    "MIN_RISK_REWARD_RATIO": MIN_RISK_REWARD_RATIO,
    "ATR_TIMEFRAME": ATR_TIMEFRAME,
    "SL_ATR_MULTIPLIER": SL_ATR_MULTIPLIER,
    "TP_BUFFER_PCT": TP_BUFFER_PCT,
    
    # Training Params
    "ATR_LABEL_MULTIPLIER": ATR_LABEL_MULTIPLIER,
    "LAG_PERIODS": LAG_PERIODS,
    "USE_STACKING_ENSEMBLE": USE_STACKING_ENSEMBLE,
    
    # Dynamic & MR Params
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
    
    # API Client Config
    "API_MAX_RETRIES": API_MAX_RETRIES,
    "API_RETRY_DELAY": API_RETRY_DELAY,
}

# ----------------------------------------------------------------------
# ✅ Startup Prints
# ----------------------------------------------------------------------
print(f"✅ Configuration loaded successfully")
print(f"📊 Trading Symbols: {TRADING_SYMBOLS}")
print(f"🎯 Priority List: {' -> '.join(PRIORITY_LIST)}")
print(f"🛡️ Risk Management: {MAX_DRAWDOWN_PERCENT*100}% max drawdown, {MAX_DAILY_LOSS_PERCENT*100}% daily loss")
print(f"🤖 Strategy: Hybrid (ML Trend + Mean Reversion)")
print(f"--- FILTERS ---")
print(f"📈 Trend: {TREND_CHECK_ENABLED} on {TREND_TIMEFRAME} candle")
print(f"💵 Funding: {FUNDING_CHECK_ENABLED} (Threshold: {FUNDING_RATE_THRESHOLD * 100}%)")
print(f"🌊 Volume: {VOLUME_CHECK_ENABLED} ({VOLUME_TIMEFRAME} vol > {VOLUME_SURGE_MULTIPLIER}x SMA({VOLUME_SMA_PERIOD}))")
print(f"⛰️ S/R: {SNR_CHECK_ENABLED} (Avoid {SNR_PROXIMITY_PCT * 100}% proximity to Daily/Weekly/Monthly levels)")
print(f"⚖️ R/R: Required Ratio > {MIN_RISK_REWARD_RATIO}:1")
print(f"🎛️ Single Position Mode: Active (Max {MAX_CONCURRENT_TRADES} concurrent trade)")
print(f"𔀀 Trailing Stop Loss: {'Enabled' if TSL_ENABLED else 'Disabled'} (Trail: {TSL_ATR_MULTIPLIER}x ATR, Min: {TSL_MIN_TRAIL_AMOUNT} USD)")
print(f"🧠 Dynamic Confidence: Enabled (Base: {BASE_CONFIDENCE}, Min: {MIN_CONFIDENCE})")