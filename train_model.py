# --- detla-bot/train_model.py ---
# WORLD-CLASS ML TRADING MODEL (ACCURACY ENHANCED VERSION)

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
    f1_score,
    classification_report,
    make_scorer 
)
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# --- Model Algorithm Imports ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 

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
# Open Interest and Liquidations files are expected to be missing/deprecated
OI_INPUT_FILE = os.path.join(DATA_DIR, "historical_open_interest.csv")
LIQ_INPUT_FILE = os.path.join(DATA_DIR, "historical_liquidations.csv")
# NEW INPUT FILE: Long/Short Ratio
LSR_INPUT_FILE = os.path.join(DATA_DIR, "historical_long_short_ratio.csv") 
# Output file
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signal_classifier_world_class.joblib")

# Labeling parameters
FUTURE_LOOKBACK = 6     

# New Validation/Tuning Configuration
N_CV_SPLITS = 5          
PURGE_GAP_PERIODS = 12   
N_ITER_TUNING = 80       # Increased for better tuning
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
# ENHANCED HYPERPARAMETER TUNING (IMPROVED VERSION)
# ----------------------------------------------------------------------
def tune_model_lgbm_enhanced(X, y):
    """
    Enhanced tuning with wider search space and feature engineering focus.
    """
    log.info(f"\n--- Starting ENHANCED LightGBM Hyperparameter Tuning (N_ITER={N_ITER_TUNING}) ---")
    
    # Calculate class weights for imbalance
    class_counts = y.value_counts().sort_index()
    total_samples = len(y)
    class_weights = {i: total_samples / (len(class_counts) * count) if count > 0 else 0 
                     for i, count in class_counts.items()}
    
    log.info(f"Class weights computed: {class_weights}")
    
    lgbm = LGBMClassifier(
        objective='multiclass', 
        num_class=3, 
        class_weight=class_weights,
        random_state=42, 
        n_jobs=-1, 
        verbose=-1
    )
    
    # Enhanced parameter distribution with feature engineering focus
    param_dist = {
        'lgbmclassifier__n_estimators': randint(500, 2000),           # More trees
        'lgbmclassifier__learning_rate': uniform(0.005, 0.1),         # Lower learning rate
        'lgbmclassifier__num_leaves': randint(31, 150),               # More leaves for complex patterns
        'lgbmclassifier__max_depth': randint(8, 25),                  # Deeper trees
        'lgbmclassifier__min_child_samples': randint(5, 50),          # Less overfitting
        'lgbmclassifier__subsample': uniform(0.6, 0.4),               # More randomness
        'lgbmclassifier__colsample_bytree': uniform(0.6, 0.4),        # Feature sampling
        'lgbmclassifier__reg_alpha': uniform(0, 2),                   # L1 regularization
        'lgbmclassifier__reg_lambda': uniform(0, 2),                  # L2 regularization
        'lgbmclassifier__min_split_gain': uniform(0.0, 0.1),          # Split improvement threshold
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lgbmclassifier', lgbm)
    ])

    tscv = PurgedTimeSeriesSplit(n_splits=TUNING_SPLITS, purge_gap_periods=PURGE_GAP_PERIODS + FUTURE_LOOKBACK)

    rscv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_TUNING,
        scoring=make_scorer(balanced_accuracy_score),
        cv=tscv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    
    rscv.fit(X, y)
    
    log.info("✅ Enhanced Tuning Complete.")
    log.info(f"Best Balanced Accuracy found: {rscv.best_score_:.4f}")
    log.info(f"Best parameters: {rscv.best_params_}")
    
    return rscv.best_estimator_.named_steps['lgbmclassifier']

# ----------------------------------------------------------------------
# FEATURE SELECTION FUNCTION
# ----------------------------------------------------------------------
def select_important_features(X, y, n_features=60):
    """
    Select top N most important features using LightGBM feature importance.
    """
    log.info(f"Selecting top {n_features} features...")
    
    # Quick feature importance with LightGBM
    selector = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    selector.fit(X, y)
    importances = selector.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance_df.head(n_features)['feature'].tolist()
    
    log.info(f"Selected top {len(selected_features)} features")
    log.info(f"Top 10 features: {selected_features[:10]}")
    
    return selected_features

# ----------------------------------------------------------------------
# WORLD-CLASS FEATURE ENGINEERING (ENHANCED VERSION)
# ----------------------------------------------------------------------
def calculate_features_enhanced(df):
    """
    World-class feature engineering with all phases implemented.
    Enhanced with additional predictive features.
    """
    df_ta = df.copy()

    # --- 0. Base ATR Calculation ---
    df_ta['ATR'] = volatility.average_true_range(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=True)

    # --- 1. EMA Framework & Market Regime ---
    emas = [8, 21, 50, 100, 200]
    for length in emas:
        df_ta[f'EMA_{length}'] = trend.ema_indicator(df_ta['close'], window=length, fillna=True)
    df_ta['EMA_50_SLOPE'] = df_ta['EMA_50'].diff(1)
    df_ta['EMA_50_ACCEL'] = df_ta['EMA_50_SLOPE'].diff(1)
    df_ta['REGIME_BULL'] = (df_ta['close'] > df_ta['EMA_8']) & (df_ta['EMA_8'] > df_ta['EMA_21']) & (df_ta['EMA_21'] > df_ta['EMA_50'])
    df_ta['REGIME_BEAR'] = (df_ta['close'] < df_ta['EMA_8']) & (df_ta['EMA_8'] < df_ta['EMA_21']) & (df_ta['EMA_21'] < df_ta['EMA_50'])
    df_ta['REGIME_CHOP'] = (~df_ta['REGIME_BULL']) & (~df_ta['REGIME_BEAR'])

    # --- 2. Bollinger Bands ---
    bb = volatility.BollingerBands(df_ta['close'], window=20, window_dev=2, fillna=True)
    df_ta['BB_UPPER'] = bb.bollinger_hband()
    df_ta['BB_LOWER'] = bb.bollinger_lband()
    df_ta['BB_MID'] = bb.bollinger_mavg()
    df_ta['BB_WIDTH'] = bb.bollinger_wband() 
    df_ta['BB_PCTB'] = bb.bollinger_pband() 
    bb_width_quantile = df_ta['BB_WIDTH'].rolling(50, min_periods=1).quantile(0.1)
    df_ta['BB_SQUEEZE'] = (df_ta['BB_WIDTH'] < bb_width_quantile).fillna(False)

    # --- 3. Stochastic Oscillator ---
    stoch = momentum.StochasticOscillator(df_ta['high'], df_ta['low'], df_ta['close'], window=14, smooth_window=3, fillna=True)
    df_ta['STOCH_K'] = stoch.stoch()
    df_ta['STOCH_D'] = stoch.stoch_signal()
    df_ta['STOCH_BULL_CROSS'] = (df_ta['STOCH_K'] > df_ta['STOCH_D']) & (df_ta['STOCH_K'].shift(1) < df_ta['STOCH_D'].shift(1)) & (df_ta['STOCH_K'] < 80)
    df_ta['STOCH_BEAR_CROSS'] = (df_ta['STOCH_K'] < df_ta['STOCH_D']) & (df_ta['STOCH_K'].shift(1) > df_ta['STOCH_D'].shift(1)) & (df_ta['STOCH_K'] > 20)

    # --- 4. Core Indicators ---
    df_ta['RSI'] = momentum.rsi(df_ta['close'], window=14, fillna=True)
    macd = trend.MACD(df_ta['close'], window_fast=12, window_slow=26, window_sign=9, fillna=True)
    df_ta['MACDh'] = macd.macd_diff()
    df_ta['OBV'] = volume.on_balance_volume(df_ta['close'], df_ta['volume'], fillna=True)
    df_ta['ADX'] = trend.adx(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=True)
    ichimoku = trend.IchimokuIndicator(df_ta['high'], df_ta['low'], fillna=True)
    df_ta['IC_TENKAN'] = ichimoku.ichimoku_conversion_line()
    df_ta['IC_KIJUN'] = ichimoku.ichimoku_base_line()
    df_ta['IC_SPAN_A'] = ichimoku.ichimoku_a()
    df_ta['IC_SPAN_B'] = ichimoku.ichimoku_b()

    # --- 5. Microstructure & Sentiment Features ---
    # Proxies (to match feature_engine.py)
    df_ta['OBI_Proxy'] = (df_ta['close'] - df_ta['low']) / (df_ta['high'] - df_ta['low'] + 1e-9)
    df_ta['TFI_Proxy'] = df_ta['volume'] * (df_ta['close'] - df_ta['open'])
    # Candle Features
    df_ta['Vol_Ratio'] = (df_ta['volume'] / df_ta['volume'].rolling(20, min_periods=1).mean())
    df_ta['Close_vs_EMA50'] = ((df_ta['close'] - df_ta['EMA_50']) / df_ta['close'].replace(0, 1e-9))

    # --- 6. Advanced Features (from User Script) ---
    df_ta['PRICE_MOMENTUM_3'] = df_ta['close'].pct_change(3)
    df_ta['PRICE_MOMENTUM_6'] = df_ta['close'].pct_change(6)
    price_change = df_ta['close'].pct_change().replace([np.inf, -np.inf], 0)
    volume_change = df_ta['volume'].pct_change().replace([np.inf, -np.inf], 0)
    df_ta['VOLUME_PRICE_DIVERGENCE'] = (price_change * volume_change < 0).astype(int)
    bb_width_quantile_70 = df_ta['BB_WIDTH'].rolling(50, min_periods=1).quantile(0.7)
    bb_width_quantile_30 = df_ta['BB_WIDTH'].rolling(50, min_periods=1).quantile(0.3)
    df_ta['VOLATILITY_HIGH'] = (df_ta['BB_WIDTH'] > bb_width_quantile_70).astype(int)
    df_ta['VOLATILITY_LOW'] = (df_ta['BB_WIDTH'] < bb_width_quantile_30).astype(int)
    df_ta['NEAR_RESISTANCE'] = ((df_ta['high'].rolling(20, min_periods=1).max() - df_ta['close']) / df_ta['close'].replace(0, 1e-9) < 0.01).astype(int)
    df_ta['NEAR_SUPPORT'] = ((df_ta['close'] - df_ta['low'].rolling(20, min_periods=1).min()) / df_ta['close'].replace(0, 1e-9) < 0.01).astype(int)

    # --- 7. Feature Engineering & Interactions (from User Script) ---
    df_ta['RSI_VOLUME_INTERACTION'] = df_ta['RSI'] * df_ta['Vol_Ratio']
    df_ta['EMA_TREND_STRENGTH'] = ((df_ta['EMA_8'] - df_ta['EMA_21']) / df_ta['EMA_21'].replace(0, 1e-9))
    df_ta['MEAN_REVERSION_RSI'] = (df_ta['RSI'] - 50).abs()
    df_ta['BB_MEAN_REVERSION'] = -df_ta['BB_PCTB']
    df_ta['PRICE_POSITION_BB'] = (df_ta['close'] - df_ta['BB_LOWER']) / (df_ta['BB_UPPER'] - df_ta['BB_LOWER']).replace(0, 1e-9)
    df_ta['PRICE_POSITION_BB'] = df_ta['PRICE_POSITION_BB'].clip(0, 1)

    # --- 8. Multi-Timeframe & Market Regime (from User Script) ---
    df_ta['HTF_TREND'] = (df_ta['EMA_50'] > df_ta['EMA_200']).astype(int)
    df_ta['HTF_MOMENTUM'] = (df_ta['close'] > df_ta['close'].rolling(50, min_periods=1).max()).astype(int)
    df_ta['RSI_7'] = momentum.rsi(df_ta['close'], window=7, fillna=True) 
    df_ta['RSI_21'] = momentum.rsi(df_ta['close'], window=21, fillna=True) 
    adx_threshold = 25
    df_ta['TRENDING_MARKET'] = (df_ta['ADX'] > adx_threshold).astype(int)
    volatility_median = df_ta['BB_WIDTH'].rolling(50, min_periods=1).quantile(0.5)
    df_ta['HIGH_VOL_REGIME'] = (df_ta['BB_WIDTH'] > volatility_median).astype(int)
    df_ta['REGIME_TRENDING_HIGH_VOL'] = (df_ta['TRENDING_MARKET'] & df_ta['HIGH_VOL_REGIME']).astype(int)
    df_ta['REGIME_TRENDING_LOW_VOL'] = (df_ta['TRENDING_MARKET'] & (~df_ta['HIGH_VOL_REGIME'].astype(bool))).astype(int)
    df_ta['REGIME_RANGING_HIGH_VOL'] = ((~df_ta['TRENDING_MARKET'].astype(bool)) & df_ta['HIGH_VOL_REGIME']).astype(int)
    df_ta['REGIME_RANGING_LOW_VOL'] = ((~df_ta['TRENDING_MARKET'].astype(bool)) & (~df_ta['HIGH_VOL_REGIME'].astype(bool))).astype(int)

    # --- 9. NEW: Advanced Price Action Features ---
    df_ta['GAP_OPENING'] = (df_ta['open'] - df_ta['close'].shift(1)) / df_ta['close'].shift(1).replace(0, 1e-9)
    df_ta['HIGH_LOW_RANGE'] = (df_ta['high'] - df_ta['low']) / df_ta['close'].replace(0, 1e-9)
    df_ta['CLOSE_POSITION'] = (df_ta['close'] - df_ta['low']) / (df_ta['high'] - df_ta['low'] + 1e-9)

    # --- 10. NEW: Advanced Volume Features ---
    df_ta['VOLUME_SURGE'] = (df_ta['volume'] > df_ta['volume'].rolling(20).mean() * 1.5).astype(int)
    df_ta['VOLUME_TREND'] = df_ta['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0)

    # --- 11. NEW: Momentum Acceleration ---
    df_ta['MOMENTUM_ACCEL'] = df_ta['PRICE_MOMENTUM_3'].diff()
    df_ta['RSI_ACCEL'] = df_ta['RSI'].diff()

    # --- 12. NEW: Volatility Regime Features ---
    volatility_bins = [0, df_ta['BB_WIDTH'].quantile(0.33), df_ta['BB_WIDTH'].quantile(0.66), np.inf]
    df_ta['VOLATILITY_REGIME'] = pd.cut(df_ta['BB_WIDTH'], bins=volatility_bins, labels=[0, 1, 2]).astype(float)

    # --- 13. NEW: Support/Resistance Strength ---
    df_ta['RESISTANCE_STRENGTH'] = (df_ta['high'].rolling(10).max() - df_ta['close']) / (df_ta['ATR'] + 1e-9)
    df_ta['SUPPORT_STRENGTH'] = (df_ta['close'] - df_ta['low'].rolling(10).min()) / (df_ta['ATR'] + 1e-9)

    # --- 14. NEW: Time-based Features ---
    if hasattr(df_ta.index, 'hour'):
        df_ta['HOUR_OF_DAY'] = df_ta.index.hour
        df_ta['DAY_OF_WEEK'] = df_ta.index.dayofweek
        df_ta['IS_WEEKEND'] = (df_ta['DAY_OF_WEEK'] >= 5).astype(int)
    else:
        df_ta['HOUR_OF_DAY'] = 0
        df_ta['DAY_OF_WEEK'] = 0
        df_ta['IS_WEEKEND'] = 0

    # --- 15. Feature Lags (Temporal Patterns) ---
    cols_to_lag = [
        'BB_WIDTH', 'ATR', 'EMA_8', 'close', 'OBI_Proxy', 'TFI_Proxy',
        'funding_rate', 'oi_pct_change', 'liq_long_vol', 'liq_short_vol',
        'RSI', 'MACDh', 'long_short_ratio', 'GAP_OPENING', 'HIGH_LOW_RANGE'
    ]
    for col in cols_to_lag:
        if col in df_ta.columns:
            for lag in LAG_PERIODS:
                df_ta[f'{col}_LAG{lag}'] = df_ta[col].shift(lag)

    # --- SAFE NaN HANDLING ---
    # Fill NaNs with 0 *after* all calculations and lags are done
    df_ta = df_ta.fillna(0)
    df_ta = df_ta.replace([np.inf, -np.inf], 0)
    df_ta = df_ta.reset_index(drop=True) 

    log.info(f"🚀 WORLD-CLASS: Finished calculating {len(df_ta.columns)} enhanced features.")
    return df_ta


# ----------------------------------------------------------------------
# ENHANCED ADAPTIVE LABELING (MULTI-TIMEFRAME VERSION)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ENHANCED ADAPTIVE LABELING (MULTI-TIMEFRAME VERSION - FIXED)
# ----------------------------------------------------------------------
def create_labels_adaptive(df):
    """
    Enhanced adaptive labeling with multiple timeframes and volatility regimes.
    """
    if len(df) < FUTURE_LOOKBACK + 10:
        log.warning(f"Insufficient data for labeling: {len(df)} rows")
        return df
    
    df = df.copy()
    
    # Multiple timeframes for labeling
    timeframes = [3, 6, 9]  # 15min, 30min, 45min ahead
    
    best_targets = []
    for tf in timeframes:
        df[f'Future_Close_{tf}'] = df['close'].shift(-tf)
        df[f'Price_Change_{tf}'] = (df[f'Future_Close_{tf}'] - df['close']) / (df['close'] + 1e-9)
        
        # Adaptive threshold based on multiple volatility measures
        volatility_1 = df['close'].pct_change().rolling(10, min_periods=1).std().fillna(0.01)
        volatility_2 = df['ATR'] / (df['close'] + 1e-9)
        combined_volatility = (volatility_1 + volatility_2) / 2
        
        adaptive_threshold = combined_volatility * 1.8  # Slightly more aggressive
        
        # Ensure minimum threshold
        final_threshold = np.maximum(adaptive_threshold, 0.0025)

        def adaptive_label_tf(row, tf_idx=tf):
            try:
                change = row[f'Price_Change_{tf_idx}']
                threshold = row[f'threshold_{tf_idx}']
                if pd.isna(change) or pd.isna(threshold): return 0
                if change >= threshold: return 1   # LONG
                elif change <= -threshold: return -1   # SHORT
                return 0   # CHOP
            except:
                return 0

        df[f'threshold_{tf}'] = final_threshold
        df[f'Target_{tf}'] = df.apply(adaptive_label_tf, axis=1)
        best_targets.append(df[f'Target_{tf}'])
    
    # Combine targets from multiple timeframes (majority voting)
    target_matrix = np.column_stack(best_targets)
    df['Target'] = pd.Series([np.bincount(row[row != 0] + 1).argmax() - 1 
                            if np.any(row != 0) else 0 
                            for row in target_matrix])
    
    # Keep the Price_Change from the primary timeframe (6 periods) for backtesting
    df['Price_Change'] = df['Price_Change_6']
    
    # Clean up temporary columns (keep Price_Change for backtesting)
    for tf in timeframes:
        df = df.drop([f'Future_Close_{tf}', f'threshold_{tf}', f'Target_{tf}'], 
                    axis=1, errors='ignore')
    
    initial_count = len(df)
    df = df.dropna(subset=['Target', 'Price_Change'])
    final_count = len(df)
    
    log.info(f"🎯 ENHANCED LABELING: Multi-timeframe adaptive thresholds.")
    dist = df['Target'].value_counts()
    log.info("Target distribution: Longs: %d, Shorts: %d, Chop: %d", 
             dist.get(1, 0), dist.get(-1, 0), dist.get(0, 0))
    log.info("Data retained: %d/%d rows (%.1f%%)", final_count, initial_count, 
             (final_count/initial_count)*100 if initial_count > 0 else 0)
    
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
# DATA LOADING AND MERGING (COMPLETELY FIXED VERSION)
# ----------------------------------------------------------------------
def load_and_merge_data():
    """Loads all CSVs and merges them into a single training dataframe."""
    
    log.info(f"Loading data from {DATA_DIR}...")
    if not os.path.exists(CANDLES_INPUT_FILE):
        log.critical(f"❌ ERROR: Main candle file not found: {CANDLES_INPUT_FILE}")
        log.critical("Please run 'historical_data_fetcher.py' first.")
        return pd.DataFrame()
        
    df = pd.read_csv(CANDLES_INPUT_FILE)
    df['time'] = pd.to_datetime(df['time'])  # Candles use 'time'
    
    # --- Load and merge Funding Rates ---
    if os.path.exists(FUNDING_INPUT_FILE):
        df_fund = pd.read_csv(FUNDING_INPUT_FILE)
        log.info(f"Funding file columns: {df_fund.columns.tolist()}")
        
        # Handle different column names for time
        if 'fundingTime' in df_fund.columns:
            df_fund['time'] = pd.to_datetime(df_fund['fundingTime'])
        elif 'time' in df_fund.columns:
            df_fund['time'] = pd.to_datetime(df_fund['time'])
        else:
            log.warning("Funding file has no recognizable time column. Using first available datetime column.")
            # Use the first datetime-like column
            for col in df_fund.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    try:
                        df_fund['time'] = pd.to_datetime(df_fund[col])
                        log.info(f"Using column '{col}' as time for funding data")
                        break
                    except:
                        continue
            if 'time' not in df_fund.columns:
                log.error("Could not find valid time column in funding data. Skipping funding merge.")
                df['funding_rate'] = 0.0
                df_fund = None
        
        if df_fund is not None:
            # Handle symbol mapping - if no symbol column, assume it's for all symbols
            if 'symbol' in df_fund.columns:
                public_to_delta = {v: k for k, v in PUBLIC_SYMBOL_MAPPING.items()}
                df_fund['symbol'] = df_fund['symbol'].map(public_to_delta)
                
                # Select only the columns we need for merging
                funding_cols = ['time', 'symbol']
                if 'fundingRate' in df_fund.columns:
                    funding_cols.append('fundingRate')
                elif 'funding_rate' in df_fund.columns:
                    funding_cols.append('funding_rate')
                else:
                    log.warning("No funding rate column found. Using 0.0")
                    df['funding_rate'] = 0.0
                    df_fund = None
                
                if df_fund is not None:
                    # Ensure both dataframes have the symbol column before merge
                    if 'symbol' not in df.columns:
                        log.error("Main dataframe missing 'symbol' column. Cannot merge.")
                        df['funding_rate'] = 0.0
                    else:
                        df = pd.merge_asof(
                            df.sort_values('time'),
                            df_fund[funding_cols].sort_values('time'),
                            on='time',
                            by='symbol',
                            direction='backward' 
                        )
                        # Rename column for consistency with feature list
                        if 'fundingRate' in df.columns:
                            df = df.rename(columns={'fundingRate': 'funding_rate'})
                        log.info("✅ Merged Funding Rate data.")
            else:
                log.warning("Funding file has no symbol column. Will merge without symbol matching.")
                # Simple time-based merge without symbol
                if 'fundingRate' in df_fund.columns:
                    df_fund_simple = df_fund[['time', 'fundingRate']].drop_duplicates('time')
                    df = pd.merge_asof(
                        df.sort_values('time'),
                        df_fund_simple.sort_values('time'),
                        on='time',
                        direction='backward'
                    )
                    if 'fundingRate' in df.columns:
                        df = df.rename(columns={'fundingRate': 'funding_rate'})
                    log.info("✅ Merged Funding Rate data (time-based only)")
                else:
                    log.warning("No funding rate column found. Using 0.0")
                    df['funding_rate'] = 0.0
    else:
        log.warning(f"Funding file not found: {FUNDING_INPUT_FILE}. Skipping.")
        df['funding_rate'] = 0.0

    # --- Load and merge Long/Short Ratio (LSR) ---
    if os.path.exists(LSR_INPUT_FILE):
        df_lsr = pd.read_csv(LSR_INPUT_FILE)
        log.info(f"LSR file columns: {df_lsr.columns.tolist()}")
        
        # Handle different column names for time
        if 'timestamp' in df_lsr.columns:
            df_lsr['time'] = pd.to_datetime(df_lsr['timestamp'])
        elif 'time' in df_lsr.columns:
            df_lsr['time'] = pd.to_datetime(df_lsr['time'])
        else:
            log.warning("LSR file has no recognizable time column. Using first available datetime column.")
            for col in df_lsr.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    try:
                        df_lsr['time'] = pd.to_datetime(df_lsr[col])
                        log.info(f"Using column '{col}' as time for LSR data")
                        break
                    except:
                        continue
            if 'time' not in df_lsr.columns:
                log.error("Could not find valid time column in LSR data. Skipping LSR merge.")
                df['long_short_ratio'] = 0.0
                df_lsr = None
        
        if df_lsr is not None:
            # Handle symbol mapping
            if 'symbol' in df_lsr.columns:
                public_to_delta = {v: k for k, v in PUBLIC_SYMBOL_MAPPING.items()}
                df_lsr['symbol'] = df_lsr['symbol'].map(public_to_delta)
                
                # Rename column for consistency with feature list
                if 'longShortRatio' in df_lsr.columns:
                    df_lsr = df_lsr.rename(columns={'longShortRatio': 'long_short_ratio'})
                elif 'long_short_ratio' not in df_lsr.columns:
                    log.warning("No long/short ratio column found. Using 0.0")
                    df['long_short_ratio'] = 0.0
                    df_lsr = None

                if df_lsr is not None:
                    # Select only the columns we need for merging
                    lsr_cols = ['time', 'symbol', 'long_short_ratio']

                    # Ensure both dataframes have the symbol column before merge
                    if 'symbol' not in df.columns:
                        log.error("Main dataframe missing 'symbol' column. Cannot merge.")
                        df['long_short_ratio'] = 0.0
                    else:
                        df = pd.merge_asof(
                            df.sort_values('time'),
                            df_lsr[lsr_cols].sort_values('time'),
                            on='time',
                            by='symbol',
                            direction='backward' 
                        )
                        log.info("✅ Merged Long/Short Ratio (LSR) data.")
            else:
                log.warning("LSR file has no symbol column. Will merge without symbol matching.")
                # Simple time-based merge without symbol
                if 'longShortRatio' in df_lsr.columns:
                    df_lsr_simple = df_lsr[['time', 'longShortRatio']].drop_duplicates('time')
                    df = pd.merge_asof(
                        df.sort_values('time'),
                        df_lsr_simple.sort_values('time'),
                        on='time',
                        direction='backward'
                    )
                    if 'longShortRatio' in df.columns:
                        df = df.rename(columns={'longShortRatio': 'long_short_ratio'})
                    log.info("✅ Merged LSR data (time-based only)")
                else:
                    log.warning("No long/short ratio column found. Using 0.0")
                    df['long_short_ratio'] = 0.0
    else:
        log.warning(f"LSR file not found: {LSR_INPUT_FILE}. Skipping. Setting LSR feature to 0.0.")
        df['long_short_ratio'] = 0.0
        
    # Fill NaNs from merges (for all sentiment features)
    sentiment_cols = ['funding_rate', 'long_short_ratio']
    for col in sentiment_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0)
        
    return df

# ----------------------------------------------------------------------
# MAIN TRAINING & VALIDATION ORCHESTRATOR (ENHANCED)
# ----------------------------------------------------------------------
def train_and_save_model(df):
    """Orchestrates tuning, CV, model selection, and saving."""
    
    # --- 1. Feature Selection (Now includes ALL features) ---
    base_feature_cols = [
        'EMA_8', 'EMA_21', 'EMA_50', 'EMA_100', 'EMA_200',
        'EMA_50_SLOPE', 'EMA_50_ACCEL',
        'REGIME_BULL', 'REGIME_BEAR', 'REGIME_CHOP',
        'BB_UPPER', 'BB_LOWER', 'BB_MID', 'BB_WIDTH', 'BB_PCTB', 'BB_SQUEEZE',
        'STOCH_K', 'STOCH_D', 'STOCH_BULL_CROSS', 'STOCH_BEAR_CROSS',
        'RSI', 'MACDh', 'ATR', 'OBV', 'ADX',
        'IC_TENKAN', 'IC_KIJUN', 'IC_SPAN_A', 'IC_SPAN_B',
        'Vol_Ratio', 'Close_vs_EMA50',
        'OBI_Proxy', 'TFI_Proxy', 
        'funding_rate', 'oi_pct_change', 'liq_long_vol', 'liq_short_vol',
        'long_short_ratio',  # ADDED LSR FEATURE
        # NEW FEATURES
        'GAP_OPENING', 'HIGH_LOW_RANGE', 'CLOSE_POSITION',
        'VOLUME_SURGE', 'VOLUME_TREND', 'MOMENTUM_ACCEL', 'RSI_ACCEL',
        'VOLATILITY_REGIME', 'RESISTANCE_STRENGTH', 'SUPPORT_STRENGTH',
        'HOUR_OF_DAY', 'DAY_OF_WEEK', 'IS_WEEKEND'
    ]
    phase_features = [
        'PRICE_MOMENTUM_3', 'PRICE_MOMENTUM_6', 'VOLUME_PRICE_DIVERGENCE',
        'VOLATILITY_HIGH', 'VOLATILITY_LOW', 'NEAR_RESISTANCE', 'NEAR_SUPPORT',
        'RSI_VOLUME_INTERACTION', 'EMA_TREND_STRENGTH',   
        'MEAN_REVERSION_RSI', 'BB_MEAN_REVERSION',
        'PRICE_POSITION_BB',
        'HTF_TREND', 'HTF_MOMENTUM', 'RSI_7', 'RSI_21',
        'TRENDING_MARKET', 'HIGH_VOL_REGIME',   
        'REGIME_TRENDING_HIGH_VOL', 'REGIME_TRENDING_LOW_VOL',
        'REGIME_RANGING_HIGH_VOL', 'REGIME_RANGING_LOW_VOL'
    ]
    lagged_feature_cols = [f'{col}_LAG{lag}' for col in [
        'BB_WIDTH', 'ATR', 'EMA_8', 'close', 'OBI_Proxy', 'TFI_Proxy',
        'funding_rate', 'oi_pct_change', 'liq_long_vol', 'liq_short_vol',
        'RSI', 'MACDh', 'long_short_ratio', 'GAP_OPENING', 'HIGH_LOW_RANGE'
    ] for lag in LAG_PERIODS]
    
    # Combine all feature sets
    feature_cols = list(set(base_feature_cols + phase_features + lagged_feature_cols))
    
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # Final check for missing columns
    available_features = []
    for col in feature_cols:
        if col not in df.columns:
            log.warning(f"Column '{col}' not found in final dataframe. Filling with 0.")
            df[col] = 0.0
        available_features.append(col)
            
    X = df[available_features]
    y = df['Target'].replace({-1: 0, 0: 1, 1: 2})
    price_changes_for_backtest = df['Price_Change'].to_numpy()
    
    # --- 1.5 Feature Selection ---
    if len(available_features) > 60:  # Only select if we have many features
        important_features = select_important_features(X, y, n_features=60)
        X = X[important_features]
        available_features = important_features
        log.info(f"🔍 Using {len(available_features)} most important features")
    
    # --- 2. Hyperparameter Tuning (LGBM Base Model) ---
    tuned_lgbm_classifier = tune_model_lgbm_enhanced(X, y)
    
    # --- 3. Define Models for Final Evaluation ---
    # Stacking ensemble is causing issues with custom CV - use LightGBM directly
    log.warning("Stacking Ensemble temporarily disabled due to CV compatibility issues. Using tuned LightGBM.")
    models = {
        "LightGBM_Tuned": Pipeline([
            ('scaler', StandardScaler()),
            ('model', tuned_lgbm_classifier)
        ])
    }

    # Purged Time-Series CV
    ptss = PurgedTimeSeriesSplit(
        n_splits=N_CV_SPLITS, 
        purge_gap_periods=PURGE_GAP_PERIODS + FUTURE_LOOKBACK
    )
    
    best_model_name = ""
    best_model_score = -np.inf 
    
    # --- 4. Run Purged Cross-Validation Loop ---
    for model_name, estimator in models.items(): 
        log.info(f"\n--- Starting Evaluation for: {model_name} ---")
        
        all_y_true_fold = []
        all_y_pred_fold = []
        all_price_changes_fold = []

        for fold, (train_idx, test_idx) in enumerate(ptss.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            price_changes_test = price_changes_for_backtest[test_idx]
            
            log.info(f"Fold {fold+1}: Fitting model...")
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            all_y_true_fold.extend(y_test)
            all_y_pred_fold.extend(y_pred)
            all_price_changes_fold.extend(price_changes_test)
            
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            log.info(f"Fold {fold+1}: BalAcc={bal_acc:.4f}, MCC={mcc:.4f}")

        # --- 5. Aggregate Metrics ---
        log.info(f"\n--- Aggregated Metrics for: {model_name} ---")
        overall_bal_acc = balanced_accuracy_score(all_y_true_fold, all_y_pred_fold)
        overall_mcc = matthews_corrcoef(all_y_true_fold, all_y_pred_fold)
        log.info(f"Overall Balanced Accuracy: {overall_bal_acc:.4f}")
        log.info(f"Overall Matthews Corr Coef: {overall_mcc:.4f}")
        
        target_names = ['SHORT (0)', 'CHOP (1)', 'LONG (2)']
        log.info(f"\nClassification Report (Overall):\n"
                 f"{classification_report(all_y_true_fold, all_y_pred_fold, zero_division=0, target_names=target_names)}")
        
        run_backtest(all_y_pred_fold, all_price_changes_fold)
        
        if overall_bal_acc > best_model_score:
            best_model_score = overall_bal_acc
            best_model_name = model_name

    # --- 6. Final Model Saving ---
    log.info(f"\n--- CV Complete. Best Model: {best_model_name} (BalAcc: {best_model_score:.4f}) ---")
    log.info(f"Retraining best model ({best_model_name}) on all data...")
    
    final_train_pipeline = models[best_model_name]
    
    # Fit the *entire* pipeline on *all* data
    final_train_pipeline.fit(X, y)
    
    # The 'final_train_pipeline' is now the production-ready inference pipeline
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_train_pipeline, OUTPUT_MODEL_PATH)
    log.info(f"💾 Final INFERENCE pipeline successfully saved to {OUTPUT_MODEL_PATH}")
    
    # --- 7. Feature Importance (SHAP) ---
    final_model = final_train_pipeline.named_steps['model']
    final_scaler = final_train_pipeline.named_steps['scaler']
    
    if SHAP_INSTALLED and hasattr(final_model, 'feature_importances_'): # e.g., LightGBM
        log.info("Calculating SHAP feature importance for base model...")
        try:
            X_scaled = final_scaler.transform(X)
            X_sample = pd.DataFrame(X_scaled, columns=available_features).sample(min(5000, X_scaled.shape[0]), random_state=42)
            explainer = TreeExplainer(final_model)
            shap_values = explainer(X_sample)
            log.info("\n--- Top 15 Feature Importance (SHAP Global Mean) ---")
            feature_imp = pd.Series(np.abs(shap_values.values).mean(axis=(0, 2)), index=available_features).sort_values(ascending=False).head(15)
            log.info(f"\n{feature_imp.to_string()}")
        except Exception as e:
            log.warning(f"SHAP analysis failed for base model: {e}")
        
    return final_train_pipeline

if __name__ == "__main__":
    
    # --- 1. Load and Merge All Data ---
    all_data_df = load_and_merge_data()
    
    if all_data_df.empty:
        log.critical("❌ No data loaded. Exiting.")
    else:
        try:
            processed_data = []
            # Check if we have symbol column for processing
            if 'symbol' not in all_data_df.columns:
                log.warning("No symbol column found. Processing as single symbol dataset.")
                symbol_df = all_data_df.copy()
                symbol_df = symbol_df.set_index('time').sort_index()

                # --- 2. Feature Calculation ---
                symbol_df = calculate_features_enhanced(symbol_df)
                
                # --- 3. Label Creation ---
                symbol_df = create_labels_adaptive(symbol_df)
                
                processed_data.append(symbol_df)
            else:
                # Use PUBLIC_SYMBOL_MAPPING to ensure we only process symbols we have sentiment data for
                for symbol in PUBLIC_SYMBOL_MAPPING.keys(): 
                    if symbol not in all_data_df['symbol'].unique():
                        log.warning(f"No candle data found for {symbol}, skipping.")
                        continue
                    
                    log.info(f"🚀 Processing {symbol} with WORLD-CLASS features...")
                    
                    symbol_df = all_data_df[all_data_df['symbol'] == symbol].copy()
                    symbol_df = symbol_df.set_index('time').sort_index()

                    # --- 2. Feature Calculation ---
                    symbol_df = calculate_features_enhanced(symbol_df)
                    
                    # --- 3. Label Creation ---
                    symbol_df = create_labels_adaptive(symbol_df)
                    
                    processed_data.append(symbol_df)
            
            if not processed_data:
                log.critical("❌ No data processed after filtering. Exiting.")
                exit()
                
            final_processed_df = pd.concat(processed_data).sort_index().reset_index(drop=True)
            log.info("🎯 Total processed data rows: %d", len(final_processed_df))

            if len(final_processed_df) < 500:
                log.critical(f"❌ Not enough data to train (need > 500, got {len(final_processed_df)}).")
            else:
                # --- 4. Train and Save Model ---
                log.info("🚀 STARTING WORLD-CLASS MODEL TRAINING...")
                train_and_save_model(final_processed_df)
            
        except Exception as e:
            log.error(f"💥 An error occurred during training: {e}", exc_info=True)