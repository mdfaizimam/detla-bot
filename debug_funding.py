import aiohttp
import asyncio
import time

URL = "https://api.delta.exchange/v2/history/candles"

async def test_funding():
    async with aiohttp.ClientSession() as session:
        # Test 1: Standard Request (BTCUSD) using FUNDING prefix convention
        # Based on search results: symbol="FUNDING:BTCUSD"
        print("\n--- Test 1: FUNDING:BTCUSD via Candles Endpoint ---")
        end = int(time.time())
        start = end - (86400 * 7) # 7 days
        # Resolution 1h (60) or 8h (480) usually for funding
        params = {
            "symbol": "FUNDING:BTCUSD", 
            "resolution": "1h", 
            "start": str(start), 
            "end": str(end)
        }
        
        async with session.get(URL, params=params) as resp:
            print(f"Status: {resp.status}")
            try:
                data = await resp.json()
                result = data.get('result', [])
                print(f"Result count: {len(result)}")
                if result:
                     print(f"Sample: {result[0]}")
                else:
                    print(f"Full Response: {data}")
            except Exception as e:
                print(f"Error decoding JSON: {e}")
                print(await resp.text())

        # Test 2: Try BTC-PERP or other symbol formats
        print("\n--- Test 2: Other Symbol Formats ---")
        for sym in ["BTC-PERP", "BTC_USD", "BTCUSDT"]:
            params["symbol"] = sym
            async with session.get(URL, params=params) as resp:
                 data = await resp.json()
                 count = len(data.get('result', []) or [])
                 print(f"Symbol '{sym}': {count} results")

if __name__ == "__main__":
    asyncio.run(test_funding())
