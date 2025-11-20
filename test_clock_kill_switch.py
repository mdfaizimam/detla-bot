# tests/test_clock_kill_switch.py
import pytest
import time
import asyncio
import aiohttp
from unittest.mock import AsyncMock
from utils.signing import sync_time_offset, MAX_ALLOWED_CLOCK_DRIFT_MS, CIRCUIT_OPEN_KEY

@pytest.mark.asyncio
async def test_clock_drift_kill_switch():
    """
    Deterministic 600 ms drift → kill-switch → Redis flag set.
    """
    redis_mock = AsyncMock()

    # ---- mock server Date header 600 ms ahead ----
    future_time = time.gmtime(time.time() + 0.6)  # 600 ms drift
    future_date_str = time.strftime("%a, %d %b %Y %H:%M:%S GMT", future_time)

    class FakeResponse:
        status = 200
        headers = {"Date": future_date_str}
        async def json(self): return {}
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass

    class FakeSession:
        async def get(self, url, headers=None): return FakeResponse()

    # ---- run ----
    with pytest.raises(RuntimeError):  # loop.stop() raises RuntimeError in pytest-asyncio runtime
        await sync_time_offset(FakeSession(), redis_mock)

    # ---- proof: Redis circuit flag set ----
    redis_mock.set.assert_called_once_with(CIRCUIT_OPEN_KEY, "1", ex=60)