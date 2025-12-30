
import aiohttp
import asyncio
import json

async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.delta.exchange/v2/products") as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            products = data.get("result", [])
            print(f"Total products: {len(products)}")
            
            # Print first 3 products to see structure
            print("\n--- Sample Products ---")
            for p in products[:3]:
                print(json.dumps(p, indent=2))
                
            # Check for BTC products
            print("\n--- BTC Products Search ---")
            btc_prods = [p for p in products if "BTC" in p.get("symbol", "")]
            for p in btc_prods[:3]:
                print(f"Symbol: {p.get('symbol')}, Base: {p.get('base_currency')}, Quote: {p.get('quote_currency')}")

if __name__ == "__main__":
    asyncio.run(main())
