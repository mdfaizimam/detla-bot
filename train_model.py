# --- train_model.py ---
# Complete ML Training Script (Phase 2)
# FIX: Corrected target labeling for XGBoost (shifts -1, 0, 1 to 0, 1, 2)

import pandas as pd
# IMPORTANT: Use the installed name
import pandas_ta_classic as ta 
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Configuration
DATA_DIR = "data"
MODEL_DIR = "model"
INPUT_FILE = os.path.join(DATA_DIR, "historical_candles.csv")
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signal_classifier.joblib")

# Labeling parameters
FUTURE_LOOKBACK = 6     # Look ahead 6 candles (30 minutes for 5m TF)
PROFIT_TARGET_PCT = 0.001 # 0.1% move to label as success (1 / 1000)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
log = logging.getLogger("ML_TRAINER")

def calculate_features(df):
    """Calculates all technical indicators used as features (X)."""
    
    # --- Volatility/Trend Features ---
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['MACDh'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACDh_12_26_9']
    
    # --- Momentum/Volume Features ---
    # NOTE: The 'classic' fork may not have all ta functions named identically. 
    # We rely on the most common ones.
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    
    # --- Microstructure Features (from feature_engine concepts) ---
    df['Vol_Ratio'] = df['volume'] / ta.sma(df['volume'], length=20)
    # Ensure there is no division by zero when calculating percentage difference
    df['Close_vs_EMA20'] = (df['close'] - df['EMA_20']) / df['close'].replace(0, 1e-9) * 100
    
    # Drop NaNs created by TA calculations
    df = df.dropna()
    
    log.info("✅ Finished calculating 9 features.")
    return df

def create_labels(df):
    """Creates the target variable (y) for classification."""
    
    df['Future_Close'] = df['close'].shift(-FUTURE_LOOKBACK)
    df['Price_Change'] = (df['Future_Close'] - df['close']) / df['close']

    # --- Target Labeling: Uses -1, 0, 1 ---
    def label_trade(change):
        if change >= PROFIT_TARGET_PCT:
            return 1 # LONG success
        elif change <= -PROFIT_TARGET_PCT:
            return -1 # SHORT success
        return 0 # Sideways/No major move

    df['Target'] = df['Price_Change'].apply(label_trade)
    
    df = df.dropna(subset=['Future_Close', 'Target'])
    
    log.info("✅ Finished creating Target labels.")
    log.info("Target distribution: Longs: %d, Shorts: %d, Chop: %d", 
             (df['Target'] == 1).sum(), 
             (df['Target'] == -1).sum(), 
             (df['Target'] == 0).sum())
    
    return df

def train_and_save_model(df):
    """Splits data, trains XGBoost, and saves the model."""
    
    # --- Feature Selection ---
    feature_cols = ['EMA_20', 'EMA_50', 'RSI', 'ATR', 'MACDh', 'OBV', 'ADX', 'Vol_Ratio', 'Close_vs_EMA20']
    
    X = df[feature_cols]
    
    # ✅ FIX: Shift target labels from [-1, 0, 1] to [0, 1, 2] for XGBoost
    # 0: SHORT (-1), 1: CHOP (0), 2: LONG (1)
    y = df['Target'].replace({-1: 0, 0: 1, 1: 2})
    
    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    log.info("Starting XGBoost training...")

    # --- Train Model (Multi-class Classification: 0, 1, 2) ---
    model = XGBClassifier(
        objective='multi:softprob',  
        num_class=3,                 
        n_estimators=500,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        # Note: scale_pos_weight is for binary, for multi-class we keep it simple here
    )
    
    # Use NumPy arrays for XGBoost fitting
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    log.info("✅ XGBoost Training Complete.")

    # --- Evaluation ---
    y_pred = model.predict(X_test.to_numpy())
    
    log.info("\n--- Model Evaluation (Test Set) ---")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    log.info(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0, target_names=['SHORT (-1)', 'CHOP (0)', 'LONG (1)']).split('\n')[2:]}")
    
    y_proba = model.predict_proba(X_test.to_numpy())
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    log.info(f"AUC Score (One-vs-Rest): {auc:.4f}")
    
    # --- Saving ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, OUTPUT_MODEL_PATH)
    log.info(f"💾 Model successfully saved to {OUTPUT_MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    
    if not os.path.exists(INPUT_FILE):
        log.critical(f"❌ ERROR: Input file not found at {INPUT_FILE}")
        log.critical("Please run the 'historical_data_fetcher.py' script first and ensure it saves data.")
    else:
        try:
            raw_df = pd.read_csv(INPUT_FILE)
            log.info("Loaded %d rows of historical data.", len(raw_df))
            
            processed_data = []
            for symbol in raw_df['symbol'].unique():
                log.info(f"Processing features for {symbol}...")
                
                # --- Prepare DataFrame ---
                symbol_df = raw_df[raw_df['symbol'] == symbol].copy()
                symbol_df['time'] = pd.to_datetime(symbol_df['time'], unit='s')
                symbol_df = symbol_df.set_index('time').sort_index()

                # --- Feature Calculation ---
                symbol_df = calculate_features(symbol_df)
                
                # --- Label Creation ---
                symbol_df = create_labels(symbol_df)
                
                processed_data.append(symbol_df)
            
            final_processed_df = pd.concat(processed_data).reset_index(drop=True)
            log.info("Total processed data rows: %d", len(final_processed_df))

            # 3. Train and Save Model
            train_and_save_model(final_processed_df)
            
        except Exception as e:
            log.error(f"💥 An error occurred during training: {e}", exc_info=True)