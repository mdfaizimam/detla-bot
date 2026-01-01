import pandas as pd
import numpy as np
from train_hybrid import load_data, train_layer_1_tft

def check():
    print("🕵️ Checking for Data Leakage...")
    
    # 1. Load Data
    df = load_data("fused_data_real_FULL.csv")
    
    # 2. Get Forecasts (Quick Inference)
    # We will just load the saved model and predict
    # (Assuming train_layer_2_rl logic logic is what you used)
    # Ideally, we load the .pth, but let's assume we run a 0 epoch train to get the object
    print("⚡ generating forecasts (fast)...")
    predictor = train_layer_1_tft(df, epochs=0)
    
    try:
        preds = predictor.predict(df)
        if preds.ndim > 1:
            preds = np.mean(preds[:, :3], axis=1)
            
        # Align
        if len(preds) < len(df):
            padding = np.zeros(len(df) - len(preds))
            preds = np.concatenate([padding, preds])
            
        df['forecast'] = preds
        
        # 3. Calculate Future Return
        df['future_return'] = df['close'].pct_change().shift(-1)
        
        # 4. Correlation
        # Shift forecast to match future?
        # If forecast at T predicts T+1, we compare df['forecast'] vs df['future_return']
        
        # Clean
        valid = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        corr = valid['forecast'].corr(valid['future_return'])
        
        print(f"\n📊 CORRELATION SCORE: {corr:.4f}")
        
        if corr > 0.8:
            print("🚨 CRITICAL LEAKAGE DETECTED! The model knows the future.")
            print("   Do NOT run live. The TFT target setup is wrong.")
        elif corr > 0.1:
            print("✅ Healthy Predictive Power. Good to trade.")
        else:
            print("⚠️ No Predictive Power. Model is guessing.")
            
    except Exception as e:
        print(f"Error checking leakage: {e}")

if __name__ == "__main__":
    check()