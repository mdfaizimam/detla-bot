
import pandas as pd
import json
import numpy as np
import os
import sys

# Ensure proper path ops
sys.path.append(os.getcwd())

from train_hybrid import DataLoader

def export_normalization_stats():
    print("📊 Calculating Normalization Statistics from Training Data...")
    
    # Load the EXACT data used for training
    data_file = "fused_data_real_FULL.csv"
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        return

    df = DataLoader.load_data(data_file)
    if df is None: 
        print("❌ Error: Failed to load data!")
        return

    # Calculate stats for REQUIRED columns only (excluding target)
    stats = {}
    
    # Columns that need normalization (Same as train_hybrid.py)
    features_to_norm = [c for c in DataLoader.REQUIRED_COLUMNS if c != 'close_log_ret']
    
    # Ensure model_institutional directory exists
    os.makedirs("model_institutional", exist_ok=True)

    for col in features_to_norm:
        if col in df.columns:
            # Handle potential infs before stats
            clean_col = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            stats[col] = {
                "mean": float(clean_col.mean()),
                "std": float(clean_col.std()) if clean_col.std() != 0 else 1.0
            }
            print(f"   ✅ {col}: Mean={stats[col]['mean']:.4f}, Std={stats[col]['std']:.4f}")
        else:
            print(f"   ⚠️ Warning: Column {col} not found in dataframe!")

    # Save to JSON
    output_file = "model_institutional/normalization_stats.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\n💾 Stats saved to {output_file}")

if __name__ == "__main__":
    export_normalization_stats()
