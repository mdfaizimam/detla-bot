import pandas as pd
import numpy as np
from train_hybrid import DataLoader

# Create a dummy dataset
DATA_FILE = "fused_data_sample.csv"

def create_dummy_data():
    # Larger dataset to ensure stable quantiles, but massive outlier block at end
    N = 2000
    dates = pd.date_range("2023-01-01", periods=N, freq="1h")
    
    # Base data: Normal distribution
    data = np.random.randn(N)
    
    # Outlier injection: Last 20% are HUGE outliers
    # This will shift Q3 significantly if calculated globally
    data[-400:] = 100000.0 
    
    df = pd.DataFrame({
        "timestamp": dates,
        "close_log_ret": np.random.randn(N),
        "vol_zscore": np.random.randn(N),
        # Target column for clipping test
        "fear_greed_norm": data 
    })
    for col in DataLoader.REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    
    # Make sure we have some "borderline" values in the early part that might get clipped differently
    # if the bounds shift.
    # Partial (Normal Dist): Threshold ~6.5.
    # Full (Outliers): Threshold ~250,000.
    # Value 10.0 should be clipped in Partial, but NOT in Full.
    df.loc[500, "fear_greed_norm"] = 10.0 
    
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Created robust dummy data {DATA_FILE} with massive outliers.")

def test_leakage():
    print("--- Running Leakage Test ---")
    loader = DataLoader()
    
    # 1. Full Context
    df_full = loader.load_data(DATA_FILE)
    val_full = df_full.iloc[500]["fear_greed_norm"]
    
    # 2. Partial Context (First 1000 rows only - EXCLUDING the outliers)
    # If the logic is sound, processing the first 1000 rows alone should yield 
    # the SAME result for row 500 as processing them with the future outliers.
    df_partial_raw = pd.read_csv(DATA_FILE).iloc[:1000]
    df_partial_raw.to_csv("temp_partial.csv", index=False)
    
    df_partial = loader.load_data("temp_partial.csv")
    val_partial = df_partial.iloc[500]["fear_greed_norm"]
    
    print(f"\nValue at Index 500:")
    print(f"   Full Context (w/ Future Outliers): {val_full:.6f}")
    print(f"   Partial Context (No Future):       {val_partial:.6f}")
    
    diff = abs(val_full - val_partial)
    if diff > 1e-6:
        print(f"❌ LEAKAGE DETECTED! Difference: {diff:.6f}")
        print("   The outliers in the future changed how the past was processed.")
    else:
        print("✅ No Leakage Detected. Values are identical.")

if __name__ == "__main__":
    create_dummy_data()
    test_leakage()
