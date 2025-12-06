# --- detla-bot/train_model.py ---
# 🧠 WORLD-CLASS ML MODEL: CALIBRATED STACKING ENSEMBLE + HYPERPARAMETER TUNING
# ✅ FEATURES: Regime Detection (KER, Fractal), Microstructure (OBI)
# ✅ OPTIMIZATION: RandomizedSearchCV to find best model settings
# ✅ ARCHITECTURE: Stacking Ensemble (LGBM + XGB + RF) -> Logistic Meta-Learner
# ✅ SAFETY: Probability Calibration (Crucial for Smart Sizing)
# ✅ DATA: SMOTE Augmentation for minority classes

import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from collections import deque
from scipy.stats import uniform, randint

# --- TUNING/VALIDATION ---
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    precision_score, 
    make_scorer,
    log_loss,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- ALGORITHMS ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- IMBALANCED LEARNING ---
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE 

# --- TA LIB ---
from ta import trend, volatility, momentum, volume

# --- CONFIG ---
from config import (
    ATR_LABEL_MULTIPLIER, 
    LAG_PERIODS, 
    USE_STACKING_ENSEMBLE
)

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = "data"
MODEL_DIR = "model"
CANDLES_INPUT_FILE = os.path.join(DATA_DIR, "historical_candles.csv")
FUNDING_INPUT_FILE = os.path.join(DATA_DIR, "historical_funding_rates.csv")
LSR_INPUT_FILE = os.path.join(DATA_DIR, "historical_long_short_ratio.csv") 
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signal_classifier.joblib")

# Training Params
FUTURE_LOOKBACK = 6     
PURGE_GAP = 12          
N_CV_SPLITS = 5         
N_ITER_SEARCH = 20 # Number of parameter combinations to try (Increase for better results)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRAINER]: %(message)s")
log = logging.getLogger("EnsembleTrainer")

# ----------------------------------------------------------------------
# 1. ADVANCED FEATURE ENGINEERING
# ----------------------------------------------------------------------
def calculate_features_enhanced(df):
    """Generates Regime-Aware Technical Indicators."""
    if len(df) < 50: return pd.DataFrame()
    df_ta = df.copy()

    try:
        # --- Volatility & Regime ---
        df_ta['ATR'] = volatility.average_true_range(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=True)
        
        # KER (Kaufman Efficiency Ratio)
        change = df_ta['close'].diff(10).abs()
        vol = df_ta['close'].diff().abs().rolling(10).sum()
        df_ta['KER'] = change / (vol + 1e-9)

        # Fractal Dimension (Proxy)
        df_ta['FRACTAL_DIM'] = df_ta['ATR'] / (df_ta['close'].rolling(20).std() + 1e-9)

        # Bollinger Bands Width
        bb = volatility.BollingerBands(df_ta['close'], window=20, window_dev=2)
        df_ta['BB_WIDTH'] = bb.bollinger_wband()

        # --- Trend ---
        for length in [8, 21, 50, 200]:
            df_ta[f'EMA_{length}'] = trend.ema_indicator(df_ta['close'], window=length, fillna=True)
        
        # --- Momentum ---
        df_ta['RSI'] = momentum.rsi(df_ta['close'], window=14, fillna=True)
        macd = trend.MACD(df_ta['close'])
        df_ta['MACDh'] = macd.macd_diff()
        df_ta['ADX'] = trend.adx(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=True)

        # --- Volume/Flow ---
        df_ta['OBV'] = volume.on_balance_volume(df_ta['close'], df_ta['volume'], fillna=True)
        df_ta['OBI_Proxy'] = (df_ta['close'] - df_ta['low']) / (df_ta['high'] - df_ta['low'] + 1e-9)
        
        # --- Interactions ---
        df_ta['RSI_x_KER'] = df_ta['RSI'] * df_ta['KER'] 
        df_ta['ADX_x_VOL'] = df_ta['ADX'] * (df_ta['volume'] / df_ta['volume'].rolling(20).mean())

        # --- Lags ---
        lag_cols = ['KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 'funding_rate', 'long_short_ratio']
        for col in lag_cols:
            if col in df_ta.columns:
                for lag in LAG_PERIODS:
                    df_ta[f'{col}_LAG{lag}'] = df_ta[col].shift(lag)

        df_ta = df_ta.fillna(0).replace([np.inf, -np.inf], 0).reset_index(drop=True)
        
    except Exception as e:
        log.error(f"Feature calculation error: {e}")
        return pd.DataFrame()

    return df_ta

# ----------------------------------------------------------------------
# 2. ADAPTIVE LABELING
# ----------------------------------------------------------------------
def create_labels(df):
    """Labels data based on Volatility-Adjusted forward returns."""
    if len(df) < FUTURE_LOOKBACK + 50: return df
    df = df.copy()
    
    df['Future_Close'] = df['close'].shift(-FUTURE_LOOKBACK)
    df['Return'] = (df['Future_Close'] - df['close']) / df['close']
    
    # Dynamic Threshold (1.5x ATR)
    atr_pct = df['ATR'] / df['close']
    threshold = atr_pct * 1.5
    min_thresh = 0.0025 
    
    final_thresh = np.maximum(threshold, min_thresh)

    conditions = [
        (df['Return'] > final_thresh),  # LONG
        (df['Return'] < -final_thresh)  # SHORT
    ]
    choices = [2, 0] 
    
    df['Target'] = np.select(conditions, choices, default=1)
    df = df.dropna(subset=['Target', 'Return'])
    
    dist = df['Target'].value_counts()
    log.info(f"Label Distribution :: Long: {dist.get(2,0)} | Short: {dist.get(0,0)} | Neutral: {dist.get(1,0)}")
    return df

# ----------------------------------------------------------------------
# 3. HYPERPARAMETER OPTIMIZATION
# ----------------------------------------------------------------------
def tune_lightgbm(X, y):
    """Finds best LightGBM parameters using Randomized Search."""
    log.info("🔎 Tuning LightGBM parameters...")
    
    param_dist = {
        'n_estimators': randint(500, 1500),
        'learning_rate': uniform(0.01, 0.1),
        'num_leaves': randint(20, 100),
        'max_depth': randint(5, 15),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }
    
    clf = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    # Use TimeSeriesSplit to prevent data leakage during tuning
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring='precision_weighted', # Optimize for precision (fewer false positives)
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    log.info(f"✅ Best LGBM Params: {search.best_params_}")
    return search.best_estimator_

# ----------------------------------------------------------------------
# 4. ENSEMBLE MODEL DEFINITION
# ----------------------------------------------------------------------
def build_stacking_ensemble(best_lgbm, class_weight=None):
    """
    Constructs a Voting/Stacking Ensemble using the tuned LGBM + others.
    """
    # 2. XGBoost (Standard strong params)
    xgb = XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # 3. Random Forest (The Stabilizer)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=5,
        max_features='sqrt', random_state=42, n_jobs=-1,
        class_weight=class_weight
    )
    
    # Meta-Learner
    estimators = [
        ('lgbm', best_lgbm), # Use the tuned model!
        ('xgb', xgb),
        ('rf', rf)
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
        passthrough=False 
    )
    
    return stack

# ----------------------------------------------------------------------
# 5. TRAINING & CALIBRATION LOOP
# ----------------------------------------------------------------------
def train_calibrated_ensemble(df):
    # --- A. Prepare Data ---
    feature_cols = [c for c in df.columns if c not in ['time', 'symbol', 'Target', 'Future_Close', 'Return', 'BB_UPPER', 'BB_LOWER']]
    X = df[feature_cols]
    y = df['Target']
    
    log.info(f"Training on {len(X)} samples with {len(feature_cols)} features.")
    
    # --- B. Tune Best Model ---
    # We tune LGBM on the raw data first
    best_lgbm = tune_lightgbm(X, y)

    # --- C. Handle Imbalance (SMOTE) ---
    log.info("Applying SMOTE for data augmentation...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    
    # --- D. Build Ensemble ---
    clf = build_stacking_ensemble(best_lgbm)
    
    # --- E. Calibration (The Truth Serum) ---
    calibrated_clf = CalibratedClassifierCV(
        estimator=clf,
        method='isotonic', 
        cv=3 
    )
    
    # --- F. Train ---
    log.info("🚀 Training Calibrated Ensemble with Tuned Components...")
    calibrated_clf.fit(X_res, y_res)
    
    # --- G. Validate ---
    y_pred = calibrated_clf.predict(X)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    log.info(f"✅ Model Training Complete. Weighted Precision (Training Set): {prec:.4f}")
    
    # --- H. Save ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(calibrated_clf, OUTPUT_MODEL_PATH)
    log.info(f"💾 Calibrated 'Genius' Model saved to {OUTPUT_MODEL_PATH}")

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
def load_and_process():
    log.info("Loading historical data...")
    if not os.path.exists(CANDLES_INPUT_FILE):
        log.error("Candle data missing. Run fetcher first.")
        return
        
    df = pd.read_csv(CANDLES_INPUT_FILE)
    df['time'] = pd.to_datetime(df['time'])
    
    # Process per symbol
    processed_dfs = []
    if 'symbol' in df.columns:
        for sym in df['symbol'].unique():
            sub_df = df[df['symbol'] == sym].sort_values('time')
            sub_df = calculate_features_enhanced(sub_df)
            sub_df = create_labels(sub_df)
            processed_dfs.append(sub_df)
    else:
        df = calculate_features_enhanced(df.sort_values('time'))
        processed_dfs.append(create_labels(df))
        
    full_df = pd.concat(processed_dfs).dropna().reset_index(drop=True)
    
    if len(full_df) > 100:
        train_calibrated_ensemble(full_df)
    else:
        log.error("Insufficient data to train model.")

if __name__ == "__main__":
    load_and_process()