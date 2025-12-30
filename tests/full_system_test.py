import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

# --- 1. SETUP ENVIRONMENT & MOCKS ---
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock key dependencies BEFORE importing modules
# Comprehensive Mock Suite
mock_redis_pkg = MagicMock()
sys.modules['redis'] = mock_redis_pkg
sys.modules['redis.asyncio'] = MagicMock()
sys.modules['redis.exceptions'] = MagicMock()

sys.modules['utils.api_client'] = MagicMock()
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt.async_support'] = MagicMock()
sys.modules['yfinance'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['orjson'] = MagicMock()

# AI/ML Mocks (Heavy libs that might not be installed or slow)
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.autograd'] = MagicMock()

sys.modules['pytorch_lightning'] = MagicMock()
sys.modules['pytorch_lightning.callbacks'] = MagicMock()
sys.modules['pytorch_lightning.loggers'] = MagicMock()

sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['sklearn.mixture'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("SYSTEM_TEST")

# Import Modules (now that mocks are in place)
try:
    from feature_engine import FeatureEngine
    from ml_strategy import MLForecastingStrategy
    from confluence_engine import CouncilOfElders
    from risk_manager import RiskManager
    from config import TRADING_SYMBOLS
except ImportError as e:
    logger.critical(f"Import Failed: {e}")
    sys.exit(1)

# --- 2. THE SIMULATION ---

# --- 2. THE SIMULATION ---

def run_system_check():
    logger.info("════════════════════════════════════════════════════")
    logger.info("🛡️  STARTING WORLD GENIUS BOT SYSTEM INTEGRATION TEST")
    logger.info("════════════════════════════════════════════════════")

    # A. Mock Infrastructure
    mock_redis = MagicMock()
    mock_api = MagicMock()
    mock_api.get.return_value = (200, {"success": True, "result": []}) # Wallet balance

    # B. Initialize Components
    logger.info("\n--- 1. INITIALIZING COMPONENTS ---")
    
    # Risk Manager
    risk_manager = RiskManager(mock_redis, mock_api)
    logger.info("✅ RiskManager Initialized")

    # Feature Engine
    mock_http_session = MagicMock()
    feature_engine = FeatureEngine(mock_redis, mock_http_session)
    feature_engine.start = MagicMock() # Prevent loop
    logger.info("✅ FeatureEngine Initialized")

    # Strategy (The Core)
    strategy = MLForecastingStrategy(mock_redis)
    strategy.config = {"BASE_POSITION_SIZE": {"ETHUSD": 10}, "RISK": {}} # Inject Config
    strategy.risk_manager = risk_manager # Inject Risk Manager
    logger.info("✅ MLForecastingStrategy Initialized")
    
    # C. Simulate Data Ingest (The Eyes)
    logger.info("\n--- 2. SIMULATING 'EYES' (DATA INGEST) ---")
    symbol = "ETHUSD"
    price = 3000.0
    
    # 1. Update Feature Engine State (Manual Injection)
    # feature_engine._handle_trade_update(...) skipped
    
    # 2. Inject Mock TAS (Technical Analysis) + Genius Features
    mock_tas = {
        "5m": {
            "close": price,
            "rsi_14": 45.0,
            "bb_width": 0.05,
            "atr": 10.0,
            "vol_zscore": 2.5, # High Vol
            "adx": 30.0,
            "dist_to_poc": 0.005 # ✅ 0.5% away (Safe)
        },
        "1h": {
            "dist_to_poc": 0.005 # Backup
        }
    }
    feature_engine.features[symbol] = {
        "mid_price": price,
        "mark_price": price,
        "tas": mock_tas,
        "timestamp": 1234567890,
        "funding_rate": 0.01,
        "order_book": {"bids": {}, "asks": {}},
        "trades": [],
        "last_trade_price": price,
        "tfi": 0.5,
        "obi": 0.2
    }
    
    # 3. Simulate Enriched Payload Publication
    enriched_payload = {
        "symbol": symbol,
        "mid_price": price,
        "mark_price": price,
        "tas": mock_tas,
        "dist_to_long_liq": 0.005, 
        "dist_to_short_liq": 0.05,
        "funding_roc": 0.0001,
        "corr_BTCUSD": 0.95,
        "regime": "Med Vol (Trend)" 
    }
    
    logger.info(f"✅ Data Packet Constructed for {symbol}")
    logger.info(f"   - Price: {price}")
    logger.info(f"   - POC Distance: {mock_tas['5m']['dist_to_poc']*100}%")

    # D. Simulate Strategy Processing (The Brain)
    logger.info("\n--- 3. SIMULATING 'BRAIN' (STRATEGY & COUNCIL) ---")
    
    strategy.tft_predictor = MagicMock()
    strategy.tft_predictor.predict.return_value = 0.002 # Bullish
    
    strategy.rl_agent = MagicMock()
    strategy.rl_agent.get_action.return_value = ("LONG", 0.8) 
    
    strategy.regime_classifier = MagicMock()
    strategy.regime_classifier.is_fitted = True
    strategy.regime_classifier.prepare_features.return_value = [[1,2,3]] 
    strategy.regime_classifier.scaler.transform.return_value = [[1,2,3]]
    strategy.regime_classifier.model.predict_proba.return_value = [[0.1, 0.8, 0.1]]
    strategy.regime_classifier.regime_map = {0: "Low Vol", 1: "Med Vol (Trend)", 2: "High Vol"}

    # Manually Call Council
    logger.info("🧠 Convening Council of Elders...")
    
    regime_probs = {"Med Vol (Trend)": 0.8}
    liquidity_state = {
        "dist_to_long_liq": 0.005, 
        "dist_to_short_liq": 0.05, 
        "dist_to_poc": 0.005 
    }
    
    council_result = strategy.council.evaluate(
        symbol=symbol,
        tft_forecast=0.002,     
        rl_action="LONG",       
        rl_confidence=0.8,
        regime="Med Vol (Trend)", 
        regime_probs=regime_probs,
        vol_zscore=2.5,
        liquidity_state=liquidity_state
    )
    
    logger.info(f"🧙 Council Verdict: {council_result['decision']} | Conf: {council_result['confidence']:.2f}")
    logger.info(f"   Reason: {council_result['reason']}")
    
    if council_result['decision'] == "LONG":
        logger.info("✅ System Sync Check 1: Council correctly fused Trend + Tactician + Regime.")
    else:
        logger.error("❌ System Sync Check 1 Failed: Council should have approved LONG.")

    # E. Simulate Signal Generation
    logger.info("\n--- 4. SIMULATING SIGNAL OUTPUT ---")
    
    size = risk_manager.calculate_dynamic_size(symbol, council_result['confidence'], "Med Vol (Trend)", 10)
    logger.info(f"📏 Dynamic Size: {size}")
    
    sl_mult = risk_manager.get_adaptive_sl_multiplier("Med Vol (Trend)")
    logger.info(f"🛑 Adaptive SL Multiplier: {sl_mult}x ATR")
    
    if size > 10 and sl_mult > 1.0:
        logger.info("✅ System Sync Check 2: Risk Manager actively adjusting Size and Stops.")
    else:
         logger.warning("⚠️ System Sync Check 2: Risk adjustments seem flat.")
         
    logger.info("\n════════════════════════════════════════════════════")
    logger.info("🎉 SYSTEM HEALTH CHECK COMPLETE")
    logger.info("════════════════════════════════════════════════════")

if __name__ == "__main__":
    run_system_check()
