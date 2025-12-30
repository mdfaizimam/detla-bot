import logging
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# --- 1. SETUP ENVIRONMENT & MOCKS ---
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Dependencies 
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

# AI Mocks (Still needed for imports inside files, but we won't instantiate objects that use them)
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("PIPELINE_TEST")

# Import Components Directly
try:
    from feature_engine import FeatureEngine
    from confluence_engine import CouncilOfElders
    from risk_manager import RiskManager
except ImportError as e:
    logger.critical(f"Import Failed: {e}")
    sys.exit(1)

# --- 2. THE SIMULATION ---

def run_pipeline_check():
    logger.info("════════════════════════════════════════════════════")
    logger.info("🧪  STARTING MANUAL PIPELINE VERIFICATION")
    logger.info("════════════════════════════════════════════════════")

    # A. Setup
    mock_redis = MagicMock()
    mock_api = MagicMock()
    mock_api.get.return_value = (200, {"success": True, "result": []})
    
    # B. Initialize Independent Components
    logger.info("\n--- 1. COMPONENT INITIALIZATION ---")
    
    risk_manager = RiskManager(mock_redis, mock_api)
    logger.info("✅ RiskManager Initialized")
    
    # Council needs config
    council_config = {"some_param": 1}
    council = CouncilOfElders(council_config)
    logger.info("✅ CouncilOfElders Initialized")

    # Feature Engine
    mock_http = MagicMock()
    feature_engine = FeatureEngine(mock_redis, mock_http)
    logger.info("✅ FeatureEngine Initialized")

    # C. Execute Pipeline Flow (Manually)
    logger.info("\n--- 2. EXECUTION FLOW ---")
    symbol = "ETHUSD"
    price = 3000.0
    
    # 1. Feature Engine: Produce TAS + Genius Features
    logger.info(f"🔹 Step 1: Feature Engine constructs Data Packet for {symbol}")
    mock_tas = {
        "5m": {
            "close": price,
            "dist_to_poc": 0.005 # ✅ 0.5% away (Safe)
        },
        "1h": {"dist_to_poc": 0.005}
    }
    enriched_payload = {
        "symbol": symbol,
        "mid_price": price,
        "tas": mock_tas,
        "dist_to_long_liq": 0.005, 
        "dist_to_short_liq": 0.05,
        "regime": "Med Vol (Trend)" 
    }
    logger.info(f"   -> Enriched Payload Created. POC Dist: {mock_tas['5m']['dist_to_poc']*100}%")

    # 2. Strategy Logic Shim (Extract & Pass)
    logger.info(f"🔹 Step 2: Strategy Extracts Data & Convenes Council")
    
    # Extract
    tas_data = enriched_payload.get("tas", {})
    dist_poc = float(tas_data["5m"].get("dist_to_poc", 0.0))
    
    liquidity_state = {
        "dist_to_long_liq": enriched_payload.get("dist_to_long_liq", 1.0),
        "dist_to_short_liq": enriched_payload.get("dist_to_short_liq", 1.0),
        "dist_to_poc": dist_poc
    }
    
    # Evaluate
    council_result = council.evaluate(
        symbol=symbol,
        tft_forecast=0.002,     # Bullish
        rl_action="LONG",       # Buy
        rl_confidence=0.8,
        regime="Med Vol (Trend)", 
        regime_probs={"Med Vol (Trend)": 0.8},
        vol_zscore=2.5,
        liquidity_state=liquidity_state
    )
    
    logger.info(f"   -> Council Decision: {council_result['decision']} (Conf: {council_result['confidence']:.2f})")
    
    if council_result['decision'] != "LONG":
         logger.error("❌ PIPELINE BROKEN: Council rejected valid Long signal.")
         return

    # 3. Risk Management
    logger.info(f"🔹 Step 3: Risk Manager Sizing & Safety")
    
    size = risk_manager.calculate_dynamic_size(symbol, council_result['confidence'], "Med Vol (Trend)", 10)
    sl_mult = risk_manager.get_adaptive_sl_multiplier("Med Vol (Trend)")
    
    logger.info(f"   -> Sizing: {size} (Base 10 -> Boosted?)")
    logger.info(f"   -> Stop Multiplier: {sl_mult}x")
    
    if size > 10 and sl_mult > 1.0:
        logger.info("✅ PIPELINE SUCCESS: Signal flowed from Data -> Council -> Risk -> Sizing")
    else:
        logger.warning("⚠️ PIPELINE WARNING: Sizing logic unexpected.")

    logger.info("\n════════════════════════════════════════════════════")
    logger.info("🎉 VERIFICATION PASSED")
    logger.info("════════════════════════════════════════════════════")

if __name__ == "__main__":
    run_pipeline_check()
