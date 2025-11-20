# detla-bot/utils/binance_client.py
# (Create this new file)

import aiohttp
import logging

BINANCE_FUTURES_URL = "https://fapi.binance.com"
SYMBOL_MAPPING = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT"
}
log = logging.getLogger(__name__)

async def get_latest_funding_rate(session: aiohttp.ClientSession, symbol: str) -> float | None:
    """
    Fetches the most recent funding rate from Binance.
    """
    binance_symbol = SYMBOL_MAPPING.get(symbol, "BTCUSDT")
    params = {"symbol": binance_symbol, "limit": 1}
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
    
    try:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and isinstance(data, list):
                    return float(data[0]['fundingRate'])
            else:
                log.warning(f"Binance API (funding) returned status: {resp.status}")
    except Exception as e:
        log.error(f"Error fetching live funding rate: {e}")
    
    return None

async def get_latest_ls_ratio(session: aiohttp.ClientSession, symbol: str) -> float | None:
    """
    Fetches the most recent Long/Short ratio from Binance.
    """
    binance_symbol = SYMBOL_MAPPING.get(symbol, "BTCUSDT")
    params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
    url = f"{BINANCE_FUTURES_URL}/futures/data/globalLongShortAccountRatio"
    
    try:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and isinstance(data, list):
                    return float(data[0]['longShortRatio'])
            else:
                log.warning(f"Binance API (L/S) returned status: {resp.status}")
    except Exception as e:
        log.error(f"Error fetching live L/S ratio: {e}")
    
    return None