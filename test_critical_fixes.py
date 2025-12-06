# --- tests/test_critical_fixes.py ---
# ✅ VERIFICATION SUITE for Critical Bug Fixes
# Tests: Ghost Equity, PnL Erasure, Executor Retry, API Delete, and Async ML

import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from aiohttp import ClientSession

# Import your actual classes
from risk_manager import RiskManager
from monitor import PositionMonitor
from executor import OrderExecutionManager
from utils.api_client import DeltaAPIClient
from ml_strategy import MLForecastingStrategy

# --- FIX 1: API Client DELETE Payload ---
@pytest.mark.asyncio
async def test_api_client_delete_sends_payload():
    """Verify that DELETE requests send JSON body (payload), not query params."""
    mock_session = AsyncMock(spec=ClientSession)
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"success": True}
    # Context manager mock for session.request
    mock_session.request.return_value.__aenter__.return_value = mock_response

    client = DeltaAPIClient(mock_session, "fake_key", "fake_secret")
    
    # Action
    await client.delete("/v2/orders", payload={"id": 123})

    # Assertion: Check arguments passed to session.request
    args, kwargs = mock_session.request.call_args
    assert args[0] == "DELETE"  # Method
    assert kwargs['params'] is None  # Params should be None
    assert kwargs['data'] == '{"id":123}'  # Payload should be in 'data' as JSON string

# --- FIX 2: Ghost Equity (Risk Manager) ---
@pytest.mark.asyncio
async def test_risk_manager_syncs_equity_on_start():
    """Verify RiskManager fetches actual wallet balance on start, ignoring default 1.0."""
    mock_redis = AsyncMock()
    mock_api = AsyncMock(spec=DeltaAPIClient)
    
    # Mock Redis returning None (simulating fresh start)
    mock_redis.get.return_value = None
    
    # Mock API returning $5000 USDT balance
    mock_api.get.return_value = (200, {
        "success": True,
        "result": [
            {"asset_symbol": "BTC", "balance": "0.1"},
            {"asset_symbol": "USDT", "balance": "5000.0"} # Target
        ]
    })

    rm = RiskManager(mock_redis, mock_api)
    
    # Action
    await rm.start()

    # Assertion
    assert rm.current_equity == 5000.0
    assert rm.peak_equity == 5000.0
    # Ensure it saved to Redis
    mock_redis.mset.assert_called()

# --- FIX 3: PnL Erasure (Position Monitor) ---
@pytest.mark.asyncio
async def test_monitor_syncs_equity_on_close():
    """Verify Monitor forces an equity sync when a position is closed."""
    mock_redis = AsyncMock()
    mock_api = AsyncMock(spec=DeltaAPIClient)
    mock_risk = AsyncMock(spec=RiskManager)
    
    monitor = PositionMonitor(mock_redis, mock_api, mock_risk)
    monitor.active_symbols.add("BTCUSD")

    # Mock API returning empty positions list (Closed)
    mock_api.get.return_value = (200, {"success": True, "result": []})

    # Action: Trigger the check
    await monitor._verify_flat_status("BTCUSD")

    # Assertion: Must call risk_manager.sync_equity()
    mock_risk.sync_equity.assert_awaited_once()
    # Lock must be released
    mock_redis.delete.assert_called_with("active_position:BTCUSD")

# --- FIX 4: Executor Fill Price Retry ---
@pytest.mark.asyncio
async def test_executor_retries_on_zero_fill_price():
    """Verify Executor waits/retries if API returns 0.0 fill price initially."""
    mock_redis = AsyncMock()
    mock_api = AsyncMock(spec=DeltaAPIClient)
    mock_api.session = AsyncMock() # Mock session just in case
    mock_risk = AsyncMock(spec=RiskManager)
    
    # Mock successful Market Order placement
    mock_api.post.side_effect = [
        (200, {"success": True, "result": {"id": 100}}), # 1. Place Entry
        (200, {"success": True}) # 2. Place Bracket
    ]

    # Mock GET Order responses: 
    # Call 1: price 0.0 (too fast)
    # Call 2: price 0.0 (still processing)
    # Call 3: price 60000.0 (Filled)
    mock_api.get.side_effect = [
        (200, {"result": {"avg_fill_price": "0.0"}}),
        (200, {"result": {"avg_fill_price": "0.0"}}),
        (200, {"result": {"avg_fill_price": "60000.0"}}),
    ]

    mock_risk.validate_signal.return_value = (True, {})
    mock_redis.set.return_value = True # Lock acquired

    executor = OrderExecutionManager(mock_redis, mock_api, mock_risk)
    
    # Mock product info
    executor.product_info_cache["BTCUSD"] = {"id": 27, "tick_size": 0.5, "precision": 1}

    signal = {
        "symbol": "BTCUSD", "direction": "LONG", 
        "tp_price": 61000, "sl_price": 59000
    }

    # Action
    with patch("asyncio.sleep", return_value=None): # Skip actual sleep
        await executor._handle_signal(signal)

    # Assertion
    # Ensure api.get was called multiple times (retry logic worked)
    assert mock_api.get.call_count == 3 

# --- FIX 5: ML Strategy Non-Blocking ---
@pytest.mark.asyncio
async def test_ml_strategy_runs_in_executor():
    """Verify CPU-intensive predict_proba is offloaded to loop executor."""
    mock_redis = AsyncMock()
    
    # Patch joblib.load to avoid loading the actual model file (prevents XGBoost warnings and speeds up test)
    with patch("joblib.load") as mock_load:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.1, 0.1, 0.8]]
        mock_load.return_value = mock_model
        
        strategy = MLForecastingStrategy(mock_redis)
        
        strategy.config = {
            "BASE_POSITION_SIZE": {"BTCUSD": 1},
            "BASE_CONFIDENCE": 0.5,
            "SL_ATR_MULTIPLIER": 1.0,
            "MIN_RISK_REWARD_RATIO": 1.5,
            "MIN_SIZE_MULTIPLIER": 1.0,
            "MAX_SIZE_MULTIPLIER": 1.0,
            "CONFIDENCE_FLOOR": 0.6,
            "CONFIDENCE_CEILING": 0.9,
            # Feature flags to skip other checks
            "TREND_CHECK_ENABLED": False,
            "FUNDING_CHECK_ENABLED": False,
            "GATEKEEPER_ENABLED": False,
            "MEAN_REVERSION_ENABLED": False,
            "DYNAMIC_CONFIDENCE_ENABLED": False
        }

        # Dummy Data
        data = {
            "symbol": "BTCUSD",
            "mid_price": 50000,
            "imbalance": 0.5,
            "funding_rate": 0.0001,
            "tas": {
                "5m": {
                    "ema_20": 50000, "ema_50": 49000, "atr": 100, 
                    "rsi_14": 50, "ker": 0.8, "adx": 25, 
                    "bb_width": 0.1, "macd_hist": 10, "obv": 1000
                },
                "1m": {}
            }
        }

        # We mock asyncio.get_running_loop to check if run_in_executor is called
        with patch("asyncio.get_running_loop") as mock_loop_getter:
            mock_loop = MagicMock()
            # Make run_in_executor return an awaitable (Future)
            f = asyncio.Future()
            f.set_result([[0.1, 0.1, 0.8]])
            mock_loop.run_in_executor.return_value = f
            mock_loop_getter.return_value = mock_loop

            # Action
            await strategy._handle_enriched_event(data, MagicMock(circuit_open=False))

            # Assertion
            # Verify run_in_executor was called with the model's predict_proba function
            mock_loop.run_in_executor.assert_called()
            args = mock_loop.run_in_executor.call_args
            # args[0] is None (default executor), args[1] is the function
            assert args[0][1] == strategy.model.predict_proba