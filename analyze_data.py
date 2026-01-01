import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
FILES = ["historical_candles.csv", "historical_spot_candles.csv", "historical_macro.csv", "historical_funding_rates.csv"]

def analyze_file(filename):
    path = DATA_DIR / filename
    print(f"\n--- Analyzing {filename} ---")
    if not path.exists():
        print("❌ File not found.")
        return

    try:
        df = pd.read_csv(path)
        if df.empty:
            print("⚠️ File is empty.")
            return

        print(f"✅ Loaded {len(df)} rows.")
        print(f"Columns: {list(df.columns)}")
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Check for duplicates
            if "symbol" in df.columns:
                dupes = df.duplicated(subset=["symbol", "timestamp"]).sum()
                print(f"Duplicates (Symbol+Time): {dupes}")
                
                # Per Symbol Stats
                print("Counts per Symbol:")
                print(df["symbol"].value_counts())
            else:
                dupes = df.duplicated(subset=["timestamp"]).sum()
                print(f"Duplicates (Time): {dupes}")
        
        # Check for NaNs
        nans = df.isna().sum().sum()
        if nans > 0:
            print(f"⚠️ Found {nans} missing values (NaNs).")
            print(df.isna().sum())
        else:
            print("✅ No missing values found.")

    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

if __name__ == "__main__":
    for f in FILES:
        analyze_file(f)
