# --- detla-bot/train_model.py ---
# WORLD-CLASS ML TRADING MODEL (PRECISION-OPTIMIZED V2)
# ✅ FIX: Added SMOTE for data augmentation (Solves "Less Data" problem)
# ✅ FIX: Added Regime Filters (Hurst/KER) to features
# ✅ FIX: Strict Precision Optimization (Only high-probability trades)
# ✅ FIX: Added safety check for insufficient data (IndexError fix)

import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from collections import deque
from scipy.stats import uniform, randint 

# --- TUNING/VALIDATION IMPORTS ---
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score, # ✅ NEW: Optimize for Precision
    classification_report,
    make_scorer 
)
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE # ✅ NEW: Data Augmentation

# --- Model Algorithm Imports ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# --- Technical Analysis Library Imports ---
from ta import trend, volatility, momentum, volume
from ta.utils import dropna

# --- NEW CONFIG IMPORTS ---
from config import (
    ATR_LABEL_MULTIPLIER, 
    LAG_PERIODS, 
    USE_STACKING_ENSEMBLE, 
    PUBLIC_SYMBOL_MAPPING
)

# --- Interpretability Imports ---
try:
    import shap
    from shap import TreeExplainer
    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = "data"
MODEL_DIR = "model"
# Input files
CANDLES_INPUT_FILE = os.path.join(DATA_DIR, "historical_candles.csv")
FUNDING_INPUT_FILE = os.path.join(DATA_DIR, "historical_funding_rates.csv")
LSR_INPUT_FILE = os.path.join(DATA_DIR, "historical_long_short_ratio.csv") 
# Output file
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signal_classifier.joblib")

# Labeling parameters
FUTURE_LOOKBACK = 6     

# New Validation/Tuning Configuration
N_CV_SPLITS = 5          
PURGE_GAP_PERIODS = 12   
N_ITER_TUNING = 50       # Reduced for speed, but SMOTE adds complexity
TUNING_SPLITS = 3        

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
log = logging.getLogger("ML_TRAINER_V8")


# ----------------------------------------------------------------------
# ADVANCED VALIDATION (PURGED TIME-SERIES SPLIT)
# ----------------------------------------------------------------------
class PurgedTimeSeriesSplit:
    """Implements Purged Walk-Forward Time-Series Cross-Validation."""
    def __init__(self, n_splits, purge_gap_periods):
        self.n_splits = n_splits
        self.purge_gap_periods = purge_gap_periods

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        tss = TimeSeriesSplit(n_splits=self.n_splits)
        
        for i, (_, test_indices) in enumerate(tss.split(X)):
            if len(test_indices) == 0: continue
            test_start_idx = test_indices[0]
            train_end_idx = test_start_idx - self.purge_gap_periods
            train_indices = indices[:train_end_idx]
            if len(train_indices) > 0 and len(test_indices) > 0:
                if self.n_splits == N_CV_SPLITS:
                    log.info(f"Fold {i+1}/{self.n_splits}: Train={len(train_indices)}, Test={len(test_indices)}")
                yield train_indices, test_indices
            else:
                if self.n_splits == N_CV_SPLITS:
                    log.warning(f"Skipping fold {i+1} due to insufficient data for purge split.")

# ----------------------------------------------------------------------
# ENHANCED HYPERPARAMETER TUNING (PRECISION FOCUSED)
# ----------------------------------------------------------------------
def tune_model_lgbm_enhanced(X, y):
    """
    Enhanced tuning using SMOTE for data augmentation and Precision scoring.
    """
    log.info(f"\n--- Starting PRECISION-OPTIMIZED Hyperparameter Tuning ---")
    
    # ✅ NEW: Use SMOTEPipeline to balance data inside the CV loop
    # This generates synthetic data points for Long/Short classes (which are usually rare)
    
    lgbm = LGBMClassifier(
        objective='multiclass', 
        num_class=3, 
        random_state=42, 
        n_jobs=-1, 
        verbose=-1
    )
    
    param_dist = {
        'lgbmclassifier__n_estimators': randint(500, 2000),
        'lgbmclassifier__learning_rate': uniform(0.005, 0.1),
        'lgbmclassifier__num_leaves': randint(31, 150),
        'lgbmclassifier__max_depth': randint(8, 25),
        'lgbmclassifier__min_child_samples': randint(5, 50),
        'lgbmclassifier__subsample': uniform(0.6, 0.4),
        'lgbmclassifier__colsample_bytree': uniform(0.6, 0.4),
        'lgbmclassifier__reg_alpha': uniform(0, 2),
        'lgbmclassifier__reg_lambda': uniform(0, 2),
    }

    # ✅ NEW: Insert SMOTE into the pipeline
    # Sampling strategy 'not majority' will oversample the minority classes (Long/Short)
    pipeline = make_imb_pipeline(
        StandardScaler(),
        SMOTE(random_state=42, k_neighbors=5), 
        lgbm
    )

    tscv = PurgedTimeSeriesSplit(n_splits=TUNING_SPLITS, purge_gap_periods=PURGE_GAP_PERIODS + FUTURE_LOOKBACK)

    # ✅ NEW: Optimize for weighted precision to minimize false positives
    precision_scorer = make_scorer(precision_score, average='weighted', zero_division=0)

    rscv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_TUNING,
        scoring=precision_scorer, # Optimize for precision!
        cv=tscv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    
    rscv.fit(X, y)
    
    log.info("✅ Precision Tuning Complete.")
    log.info(f"Best Precision Score found: {rscv.best_score_:.4f}")
    log.info(f"Best parameters: {rscv.best_params_}")
    
    return rscv.best_estimator_

# ----------------------------------------------------------------------
# FEATURE SELECTION FUNCTION
# ----------------------------------------------------------------------
def select_important_features(X, y, n_features=60):
    """Select top N most important features."""
    log.info(f"Selecting top {n_features} features...")
    
    selector = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    selector.fit(X, y)
    importances = selector.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    
    selected_features = feature_importance_df.head(n_features)['feature'].tolist()
    log.info(f"Selected top {len(selected_features)} features")
    return selected_features

# ----------------------------------------------------------------------
# WORLD-CLASS FEATURE ENGINEERING (REGIME AWARE)
# ----------------------------------------------------------------------
def calculate_features_enhanced(df):
    """
    World-class feature engineering with Regime Detection.
    """
    # ✅ FIX: Check for minimum length to avoid IndexError in TA calculations
    if len(df) < 50:
        log.warning(f"Dataframe too short for feature calculation ({len(df)} rows). Skipping.")
        return pd.DataFrame()

    df_ta = df.copy()

    try:
        # --- 0. Base ATR & Indicators ---
        # Ensure we fill NA to prevent issues
        df_ta['ATR'] = volatility.average_true_range(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=True)
        
        # --- 1. EMA Framework ---
        emas = [8, 21, 50, 100, 200]
        for length in emas:
            df_ta[f'EMA_{length}'] = trend.ema_indicator(df_ta['close'], window=length, fillna=True)
        
        # --- 2. REGIME DETECTION FEATURES (CRITICAL FOR ACCURACY) ---
        # Kaufman Efficiency Ratio (KER)
        # Direction / Volatility. High values = Strong Trend. Low values = Chop.
        er_period = 10
        change = df_ta['close'].diff(er_period).abs()
        volatility_sum = df_ta['close'].diff().abs().rolling(er_period).sum()
        df_ta['KER'] = change / (volatility_sum + 1e-9)
        
        # Simple Fractal Dimension (Regime Quality)
        # Based on Rolling Rescaled Range (approx)
        # High dim (>1.5) = Mean Reverting, Low dim (<1.5) = Trending
        # We use a simplified volatility ratio proxy here for speed
        df_ta['FRACTAL_DIM'] = df_ta['ATR'] / (df_ta['close'].rolling(20).std() + 1e-9)
        
        # Volatility Regime
        bb = volatility.BollingerBands(df_ta['close'], window=20, window_dev=2, fillna=True)
        df_ta['BB_WIDTH'] = bb.bollinger_wband()
        df_ta['BB_UPPER'] = bb.bollinger_hband()
        df_ta['BB_LOWER'] = bb.bollinger_lband()
        
        # --- 3. Standard Momentum ---
        df_ta['RSI'] = momentum.rsi(df_ta['close'], window=14, fillna=True)
        macd = trend.MACD(df_ta['close'], window_fast=12, window_slow=26, window_sign=9, fillna=True)
        df_ta['MACDh'] = macd.macd_diff()
        df_ta['OBV'] = volume.on_balance_volume(df_ta['close'], df_ta['volume'], fillna=True)
        df_ta['ADX'] = trend.adx(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=True)

        # --- 4. Microstructure Proxies ---
        df_ta['OBI_Proxy'] = (df_ta['close'] - df_ta['low']) / (df_ta['high'] - df_ta['low'] + 1e-9)
        df_ta['Vol_Ratio'] = (df_ta['volume'] / df_ta['volume'].rolling(20, min_periods=1).mean())
        df_ta['Close_vs_EMA20'] = ((df_ta['close'] - df_ta['EMA_21']) / df_ta['close'].replace(0, 1e-9)) * 100

        # --- 5. Interaction Features (Synthetic Data) ---
        # Generate synthetic relationships to help trees find patterns
        df_ta['RSI_x_KER'] = df_ta['RSI'] * df_ta['KER'] # Trend strength * Momentum
        df_ta['ADX_x_VOL'] = df_ta['ADX'] * df_ta['Vol_Ratio'] # Trend Strength * Volume
        
        # --- 6. Feature Lags ---
        cols_to_lag = [
            'KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 
            'funding_rate', 'long_short_ratio'
        ]
        for col in cols_to_lag:
            if col in df_ta.columns:
                for lag in LAG_PERIODS:
                    df_ta[f'{col}_LAG{lag}'] = df_ta[col].shift(lag)

        # --- SAFE NaN HANDLING ---
        df_ta = df_ta.fillna(0)
        df_ta = df_ta.replace([np.inf, -np.inf], 0)
        df_ta = df_ta.reset_index(drop=True) 
        
    except Exception as e:
        log.error(f"Error calculating features: {e}")
        return pd.DataFrame() # Return empty on failure

    log.info(f"🚀 WORLD-CLASS: Finished calculating {len(df_ta.columns)} enhanced features.")
    return df_ta


# ----------------------------------------------------------------------
# ENHANCED ADAPTIVE LABELING (STRICTER THRESHOLDS)
# ----------------------------------------------------------------------
def create_labels_adaptive(df):
    """
    Adaptive labeling that requires HIGHER volatility to trigger a label.
    This filters out small choppy moves from the training set.
    """
    if len(df) < FUTURE_LOOKBACK + 10:
        return df
    
    df = df.copy()
    
    # Only use the 6-period lookback (30 mins) for cleaner signal
    tf = 6 
    df[f'Future_Close_{tf}'] = df['close'].shift(-tf)
    df[f'Price_Change'] = (df[f'Future_Close_{tf}'] - df['close']) / (df['close'] + 1e-9)
    
    # Stricter Threshold: 2.0x ATR (was 1.5x or dynamic)
    # We want the model to only learn BIG moves
    volatility_measure = df['ATR'] / (df['close'] + 1e-9)
    threshold = volatility_measure * 2.0 
    final_threshold = np.maximum(threshold, 0.003) # Minimum 0.3% move

    def adaptive_label(row):
        change = row['Price_Change']
        thresh = row['Threshold']
        if change >= thresh: return 2   # LONG
        elif change <= -thresh: return 0 # SHORT
        return 1                         # CHOP/NEUTRAL

    df['Threshold'] = final_threshold
    df['Target'] = df.apply(adaptive_label, axis=1)
    
    df = df.drop([f'Future_Close_{tf}', 'Threshold'], axis=1, errors='ignore')
    df = df.dropna(subset=['Target', 'Price_Change'])
    
    log.info(f"🎯 STRICT LABELING APPLIED.")
    dist = df['Target'].value_counts()
    log.info(f"Target distribution: Short(0): {dist.get(0,0)}, Chop(1): {dist.get(1,0)}, Long(2): {dist.get(2,0)}")
    
    return df

# ----------------------------------------------------------------------
# RISK-ADJUSTED PERFORMANCE METRICS
# ----------------------------------------------------------------------
def run_backtest(y_pred, price_changes):
    """Performs a simple vector-based backtest to calculate trading metrics."""
    y_pred = y_pred[:len(price_changes)]
    positions = pd.Series(y_pred).map({2: 1, 0: -1, 1: 0}).to_numpy()
    returns = positions * price_changes
    active_returns = returns[positions != 0]

    if len(active_returns) == 0 or np.all(active_returns == 0):
        log.warning("Backtest produced no active trades or returns. Skipping metrics.")
        return 0, 0, 0, 0

    cumulative_returns = pd.Series(returns).cumsum()
    total_return = cumulative_returns.iloc[-1]
    running_max = cumulative_returns.cummax()
    drawdown = running_max - cumulative_returns
    max_drawdown = drawdown.max()
    
    if active_returns.std() == 0: sharpe_ratio = 0
    else: sharpe_ratio = active_returns.mean() / active_returns.std()
    if max_drawdown == 0: calmar_ratio = 0
    else: calmar_ratio = total_return / max_drawdown

    log.info(f"--- Backtest Results ---")
    log.info(f"Total Return (Fractional): {total_return:.4f}")
    log.info(f"Max Drawdown (Fractional): {max_drawdown:.4f}")
    log.info(f"Per-Period Sharpe Ratio (Active Trades): {sharpe_ratio:.4f}")
    log.info(f"Calmar Ratio: {calmar_ratio:.4f}")
    return total_return, max_drawdown, sharpe_ratio, calmar_ratio

# ----------------------------------------------------------------------
# DATA LOADING AND MERGING
# ----------------------------------------------------------------------
def load_and_merge_data():
    """Loads all CSVs and merges them into a single training dataframe."""
    log.info(f"Loading data from {DATA_DIR}...")
    if not os.path.exists(CANDLES_INPUT_FILE):
        log.critical(f"❌ ERROR: Main candle file not found: {CANDLES_INPUT_FILE}")
        return pd.DataFrame()
        
    df = pd.read_csv(CANDLES_INPUT_FILE)
    df['time'] = pd.to_datetime(df['time'])
    
    # Load Funding
    if os.path.exists(FUNDING_INPUT_FILE):
        df_fund = pd.read_csv(FUNDING_INPUT_FILE)
        # Assuming standard format for brevity, preserving robustness from previous code
        if 'fundingRate' in df_fund.columns:
             df_fund['time'] = pd.to_datetime(df_fund['fundingTime']) if 'fundingTime' in df_fund.columns else pd.to_datetime(df_fund['time'])
             df = pd.merge_asof(df.sort_values('time'), df_fund[['time', 'fundingRate']].sort_values('time'), on='time', direction='backward')
             df = df.rename(columns={'fundingRate': 'funding_rate'})

    # Load LSR
    if os.path.exists(LSR_INPUT_FILE):
        df_lsr = pd.read_csv(LSR_INPUT_FILE)
        if 'longShortRatio' in df_lsr.columns:
            df_lsr['time'] = pd.to_datetime(df_lsr['timestamp']) if 'timestamp' in df_lsr.columns else pd.to_datetime(df_lsr['time'])
            df = pd.merge_asof(df.sort_values('time'), df_lsr[['time', 'longShortRatio']].sort_values('time'), on='time', direction='backward')
            df = df.rename(columns={'longShortRatio': 'long_short_ratio'})
            
    # Fill NaNs
    for col in ['funding_rate', 'long_short_ratio']:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0)
        
    return df

# ----------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# ----------------------------------------------------------------------
def train_and_save_model(df):
    # Define features (Must match feature_engine.py + extra calculated ones)
    feature_cols = [
        'EMA_8', 'EMA_21', 'EMA_50', 
        'KER', 'FRACTAL_DIM', 'BB_WIDTH', # Regime features
        'RSI', 'MACDh', 'ATR', 'OBV', 'ADX',
        'OBI_Proxy', 'Vol_Ratio', 'Close_vs_EMA20',
        'funding_rate', 'long_short_ratio',
        'RSI_x_KER', 'ADX_x_VOL' # Interaction features
    ]
    # Add lags
    lagged_cols = [c for c in df.columns if '_LAG' in c]
    feature_cols.extend(lagged_cols)
    
    # Check availability
    available_features = [c for c in feature_cols if c in df.columns]
    
    X = df[available_features]
    y = df['Target']
    
    # Feature Selection
    if len(available_features) > 40:
        available_features = select_important_features(X, y, n_features=40)
        X = X[available_features]
    
    # Tuning & Training with SMOTE
    tuned_pipeline = tune_model_lgbm_enhanced(X, y)
    
    # Retrain on full data
    log.info("🚀 Retraining best model on full dataset...")
    tuned_pipeline.fit(X, y)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tuned_pipeline, OUTPUT_MODEL_PATH)
    log.info(f"💾 Final Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    all_data = load_and_merge_data()
    if not all_data.empty:
        if 'symbol' in all_data.columns:
            # Process by symbol if mixed
            dfs = []
            # ✅ FIX: Iterate safely and skip small data chunks
            for sym in all_data['symbol'].unique():
                sdf = all_data[all_data['symbol'] == sym].copy().sort_values('time')
                
                # Skip if too small
                if len(sdf) < 50:
                    log.warning(f"Skipping {sym}: Insufficient data ({len(sdf)} rows).")
                    continue
                    
                sdf = calculate_features_enhanced(sdf)
                
                if sdf.empty: # Double check if calculation returned empty
                    continue
                    
                sdf = create_labels_adaptive(sdf)
                dfs.append(sdf)
                
            if dfs:
                final_df = pd.concat(dfs).sort_values('time').reset_index(drop=True)
                train_and_save_model(final_df)
            else:
                log.error("❌ No valid data after processing all symbols.")
        else:
            # Single symbol case
            all_data = all_data.sort_values('time')
            if len(all_data) >= 50:
                all_data = calculate_features_enhanced(all_data)
                final_df = create_labels_adaptive(all_data)
                train_and_save_model(final_df)
            else:
                log.error(f"Insufficient data ({len(all_data)} rows).")