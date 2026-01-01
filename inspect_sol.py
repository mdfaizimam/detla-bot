import requests
import json

def inspect_sol():
    url = "https://api.india.delta.exchange/v2/products"
    try:
        resp = requests.get(url)
        data = resp.json()
        products = data.get("result", [])
        
        print(f"Total Products: {len(products)}")
        
        sol_prods = [p for p in products if "SOL" in p.get("symbol", "")]
        print(f"Found {len(sol_prods)} SOL-related products.")
        
        for p in sol_prods:
            if p.get("symbol") == "SOLUSD":
                print("\n--- TARGET FOUND: SOLUSD ---")
                print(json.dumps(p, indent=2))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_sol()
