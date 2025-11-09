# --- train_model.py ---
# World-Class ML Training Pipeline (Phase 5: Final Corrected Code)
#
# Fixes included:
# 1.  Corrected 'lgbmclassifier__' prefix for RandomizedSearchCV.
# 2.  Added required 'get_n_splits' method to PurgedTimeSeriesSplit.
# 3.  FINAL FIX: Dynamic extraction of 'model' and 'scaler' to fix KeyError for tuned LightGBM.

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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# --- Model Algorithm Imports ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# --- Technical Analysis Library Imports ---
from ta import trend, volatility, momentum, volume
from ta.utils import dropna

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
INPUT_FILE = os.path.join(DATA_DIR, "historical_candles.csv")
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signal_classifier_v4_tuned.joblib")

# Labeling parameters
FUTURE_LOOKBACK = 6     
PROFIT_TARGET_PCT = 0.001 

# New Validation/Tuning Configuration
N_CV_SPLITS = 5          
PURGE_GAP_PERIODS = 12   
N_ITER_TUNING = 50       
TUNING_SPLITS = 3        

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
log = logging.getLogger("ML_TRAINER_V3")


# ----------------------------------------------------------------------
# ADVANCED VALIDATION (PURGED TIME-SERIES SPLIT)
# ----------------------------------------------------------------------

class PurgedTimeSeriesSplit:
    """
    Implements Purged Walk-Forward Time-Series Cross-Validation.
    """
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
# HYPERPARAMETER TUNING (FINAL CORRECTED FUNCTION)
# ----------------------------------------------------------------------

def tune_model_lgbm(X, y):
    """Performs Randomized Search CV for LightGBM tuning."""
    log.info(f"\n--- Starting LightGBM Hyperparameter Tuning (N_ITER={N_ITER_TUNING}) ---")
    
    lgbm = LGBMClassifier(objective='multiclass', num_class=3, random_state=42, n_jobs=-1, verbose=-1)
    scorer = make_scorer(balanced_accuracy_score)

    param_dist = {
        'lgbmclassifier__n_estimators': randint(200, 1000),      
        'lgbmclassifier__learning_rate': uniform(0.01, 0.1),     
        'lgbmclassifier__num_leaves': randint(10, 50),           
        'lgbmclassifier__max_depth': randint(3, 15),             
        'lgbmclassifier__min_child_samples': randint(10, 50),    
        'lgbmclassifier__subsample': uniform(0.6, 0.4),          
        'lgbmclassifier__colsample_bytree': uniform(0.6, 0.4)    
    }

    n_neighbors = min(5, y.value_counts().min() - 1)
    pipeline = make_imb_pipeline(
        StandardScaler(),
        SMOTE(random_state=42, k_neighbors=n_neighbors if n_neighbors > 0 else 1),
        lgbm
    )

    tscv = PurgedTimeSeriesSplit(n_splits=TUNING_SPLITS, purge_gap_periods=PURGE_GAP_PERIODS + FUTURE_LOOKBACK)

    rscv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_TUNING,
        scoring=scorer,
        cv=tscv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    
    rscv.fit(X, y)
    
    log.info("✅ Tuning Complete.")
    log.info(f"Best Balanced Accuracy found: {rscv.best_score_:.4f}")
    
    return rscv.best_estimator_

# ----------------------------------------------------------------------
# FEATURE ENGINEERING (Using 'ta' library)
# ----------------------------------------------------------------------

def calculate_features(df):
    """Calculates all advanced technical indicators and market regimes using the 'ta' package."""

    df_ta = df.copy()

    # --- 1. EMA Framework & Market Regime ---
    emas = [8, 21, 50, 100, 200]
    for length in emas:
        df_ta[f'EMA_{length}'] = trend.ema_indicator(df_ta['close'], window=length, fillna=False)

    # EMA Slope and Acceleration
    df_ta['EMA_50_SLOPE'] = df_ta['EMA_50'].diff(1)
    df_ta['EMA_50_ACCEL'] = df_ta['EMA_50_SLOPE'].diff(1)

    # Market Regime Classification
    df_ta['REGIME_BULL'] = (df_ta['close'] > df_ta['EMA_8']) & \
                       (df_ta['EMA_8'] > df_ta['EMA_21']) & \
                       (df_ta['EMA_21'] > df_ta['EMA_50'])

    df_ta['REGIME_BEAR'] = (df_ta['close'] < df_ta['EMA_8']) & \
                       (df_ta['EMA_8'] < df_ta['EMA_21']) & \
                       (df_ta['EMA_21'] < df_ta['EMA_50'])

    df_ta['REGIME_CHOP'] = (~df_ta['REGIME_BULL']) & (~df_ta['REGIME_BEAR'])

    # --- 2. Bollinger Bands ---
    bb = volatility.BollingerBands(df_ta['close'], window=20, window_dev=2, fillna=False)
    df_ta['BB_UPPER'] = bb.bollinger_hband()
    df_ta['BB_LOWER'] = bb.bollinger_lband()
    df_ta['BB_MID'] = bb.bollinger_mavg()
    df_ta['BB_WIDTH'] = bb.bollinger_wband()  
    df_ta['BB_PCTB'] = bb.bollinger_pband()   
    df_ta['BB_SQUEEZE'] = df_ta['BB_WIDTH'] < df_ta['BB_WIDTH'].rolling(50).quantile(0.1)

    # --- 3. Stochastic Oscillator ---
    stoch = momentum.StochasticOscillator(df_ta['high'], df_ta['low'], df_ta['close'], window=14, smooth_window=3, fillna=False)
    df_ta['STOCH_K'] = stoch.stoch()
    df_ta['STOCH_D'] = stoch.stoch_signal()

    # Crossovers (in non-extreme zones)
    df_ta['STOCH_BULL_CROSS'] = (df_ta['STOCH_K'] > df_ta['STOCH_D']) & \
                             (df_ta['STOCH_K'].shift(1) < df_ta['STOCH_D'].shift(1)) & \
                             (df_ta['STOCH_K'] < 80)

    df_ta['STOCH_BEAR_CROSS'] = (df_ta['STOCH_K'] < df_ta['STOCH_D']) & \
                             (df_ta['STOCH_K'].shift(1) > df_ta['STOCH_D'].shift(1)) & \
                             (df_ta['STOCH_K'] > 20)

    # --- 4. Additional Powerful Indicators ---
    df_ta['RSI'] = momentum.rsi(df_ta['close'], window=14, fillna=False)
    df_ta['RSI_HIDDEN_DIV'] = (df_ta['low'].rolling(14).min() > df_ta['low'].rolling(14).min().shift(14)) & \
                           (df_ta['RSI'].rolling(14).min() < df_ta['RSI'].rolling(14).min().shift(14))

    macd = trend.MACD(df_ta['close'], window_fast=12, window_slow=26, window_sign=9, fillna=False)
    df_ta['MACDh'] = macd.macd_diff()

    df_ta['ATR'] = volatility.average_true_range(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=False)

    df_ta['OBV'] = volume.on_balance_volume(df_ta['close'], df_ta['volume'], fillna=False)
    df_ta['ADX'] = trend.adx(df_ta['high'], df_ta['low'], df_ta['close'], window=14, fillna=False)

    # Ichimoku Cloud 
    ichimoku = trend.IchimokuIndicator(df_ta['high'], df_ta['low'], fillna=False)
    df_ta['IC_TENKAN'] = ichimoku.ichimoku_conversion_line()
    df_ta['IC_KIJUN'] = ichimoku.ichimoku_base_line()
    df_ta['IC_SPAN_A'] = ichimoku.ichimoku_a()
    df_ta['IC_SPAN_B'] = ichimoku.ichimoku_b()


    # --- 5. Microstructure Features ---
    df_ta['Vol_Ratio'] = df_ta['volume'] / df_ta['volume'].rolling(20).mean()
    df_ta['Close_vs_EMA50'] = (df_ta['close'] - df_ta['EMA_50']) / df_ta['close'].replace(0, 1e-9)

    # Drop NaNs created by TA calculations
    df_ta = dropna(df_ta)
    df_ta = df_ta.reset_index(drop=True) 

    log.info(f"✅ Finished calculating {len(df_ta.columns)} features.")
    return df_ta


# ----------------------------------------------------------------------
# LABELING 
# ----------------------------------------------------------------------

def create_labels(df):
    """Creates the target variable (y) for classification."""
    
    df['Future_Close'] = df['close'].shift(-FUTURE_LOOKBACK)
    df['Price_Change'] = (df['Future_Close'] - df['close']) / df['close']

    def label_trade(change):
        if change >= PROFIT_TARGET_PCT:
            return 1 
        elif change <= -PROFIT_TARGET_PCT:
            return -1 
        return 0 

    df['Target'] = df['Price_Change'].apply(label_trade)
    
    df = df.dropna(subset=['Future_Close', 'Target', 'Price_Change'])
    
    log.info("✅ Finished creating Target labels.")
    dist = df['Target'].value_counts()
    log.info("Target distribution: Longs: %d, Shorts: %d, Chop: %d", 
             dist.get(1, 0), dist.get(-1, 0), dist.get(0, 0))
    
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
    
    if active_returns.std() == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = active_returns.mean() / active_returns.std()
        
    if max_drawdown == 0:
        calmar_ratio = 0
    else:
        calmar_ratio = total_return / max_drawdown

    log.info(f"--- Backtest Results ---")
    log.info(f"Total Return (Fractional): {total_return:.4f}")
    log.info(f"Max Drawdown (Fractional): {max_drawdown:.4f}")
    log.info(f"Per-Period Sharpe Ratio (Active Trades): {sharpe_ratio:.4f}")
    log.info(f"Calmar Ratio: {calmar_ratio:.4f}")
    
    return total_return, max_drawdown, sharpe_ratio, calmar_ratio


# ----------------------------------------------------------------------
# MAIN TRAINING & VALIDATION ORCHESTRATOR
# ----------------------------------------------------------------------

def train_and_save_model(df):
    """Orchestrates tuning, CV, model selection, and saving."""
    
    # --- 1. Feature Selection ---
    feature_cols = [
        'EMA_8', 'EMA_21', 'EMA_50', 'EMA_100', 'EMA_200',
        'EMA_50_SLOPE', 'EMA_50_ACCEL',
        'REGIME_BULL', 'REGIME_BEAR', 'REGIME_CHOP',
        'BB_UPPER', 'BB_LOWER', 'BB_MID', 'BB_WIDTH', 'BB_PCTB', 'BB_SQUEEZE',
        'STOCH_K', 'STOCH_D', 'STOCH_BULL_CROSS', 'STOCH_BEAR_CROSS',
        'RSI', 'RSI_HIDDEN_DIV', 'MACDh', 'ATR', 'OBV', 'ADX',
        'IC_TENKAN', 'IC_KIJUN', 'IC_SPAN_A', 'IC_SPAN_B',
        'Vol_Ratio', 'Close_vs_EMA50'
    ]
    
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    X = df[feature_cols]
    y = df['Target'].replace({-1: 0, 0: 1, 1: 2})
    price_changes_for_backtest = df['Price_Change'].to_numpy()
    
    # --- 2. Hyperparameter Tuning ---
    tuned_lgbm_pipeline = tune_model_lgbm(X, y)
    
    # --- 3. Define Models for Final Evaluation ---
    # Ensure all models are wrapped in a Pipeline for consistent handling later.
    models = {
        "LightGBM_Tuned": tuned_lgbm_pipeline,
        "XGBoost_Base": Pipeline([
            ('scaler', StandardScaler()), 
            ('model', XGBClassifier(
                objective='multi:softprob', num_class=3, n_estimators=300,
                learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1
            ))
        ]),
        "BalancedRF_Base": Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
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
            
            if model_name == "LightGBM_Tuned":
                 cv_pipeline = estimator
                 y_pred = cv_pipeline.predict(X_test)
            else:
                if model_name == "XGBoost_Base":
                    raw_model = estimator.named_steps['model']
                    n_neighbors = min(5, y_train.value_counts().min() - 1)
                    cv_pipeline = make_imb_pipeline(
                        StandardScaler(),
                        SMOTE(random_state=42, k_neighbors=n_neighbors if n_neighbors > 0 else 1),
                        raw_model
                    )
                else:
                    cv_pipeline = estimator
                
                cv_pipeline.fit(X_train, y_train)
                y_pred = cv_pipeline.predict(X_test)
            
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
    log.info(f"Saving the best model pipeline: {best_model_name}...")
    
    final_pipeline = models[best_model_name]
    
    # FIX: Dynamic key extraction for model and scaler steps
    if best_model_name == "LightGBM_Tuned":
        classifier_key = 'lgbmclassifier'
        scaler_key = 'standardscaler'
    else:
        classifier_key = 'model'
        scaler_key = 'scaler'
        
    final_model = final_pipeline.named_steps[classifier_key]
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_pipeline, OUTPUT_MODEL_PATH)
    log.info(f"💾 Final inference pipeline successfully saved to {OUTPUT_MODEL_PATH}")
    
    # --- 7. Feature Importance (SHAP) ---
    if SHAP_INSTALLED and hasattr(final_model, 'feature_importances_'):
        log.info("Calculating SHAP feature importance...")
        try:
            final_scaler = final_pipeline.named_steps[scaler_key]
            X_scaled = final_scaler.transform(X)
            
            X_sample = pd.DataFrame(X_scaled, columns=feature_cols).sample(min(5000, X_scaled.shape[0]), random_state=42)
            
            explainer = TreeExplainer(final_model)
            shap_values = explainer(X_sample)
            
            log.info("\n--- Top 10 Feature Importance (SHAP Global Mean) ---")
            feature_imp = pd.Series(np.abs(shap_values.values).mean(axis=(0, 2)), index=feature_cols).sort_values(ascending=False).head(10)
            log.info(f"\n{feature_imp.to_string()}")

        except Exception as e:
            log.error(f"SHAP analysis failed: {e}")
        
    return final_pipeline

if __name__ == "__main__":
    
    if not os.path.exists(INPUT_FILE):
        log.critical(f"❌ ERROR: Input file not found at {INPUT_FILE}")
        log.critical("Please ensure your historical data file exists at the correct path.")
    else:
        try:
            raw_df = pd.read_csv(INPUT_FILE)
            log.info("Loaded %d rows of historical data.", len(raw_df))
            
            processed_data = []
            for symbol in raw_df['symbol'].unique():
                log.info(f"Processing features for {symbol}...")
                
                symbol_df = raw_df[raw_df['symbol'] == symbol].copy()
                symbol_df['time'] = pd.to_datetime(symbol_df['time'], unit='s')
                symbol_df = symbol_df.set_index('time').sort_index()

                symbol_df = calculate_features(symbol_df)
                symbol_df = create_labels(symbol_df)
                
                processed_data.append(symbol_df)
            
            final_processed_df = pd.concat(processed_data).sort_index().reset_index(drop=True)
            log.info("Total processed data rows: %d", len(final_processed_df))

            if len(final_processed_df) < 500:
                log.critical(f"❌ Not enough data to train (need > 500, got {len(final_processed_df)}).")
            else:
                train_and_save_model(final_processed_df)
            
        except Exception as e:
            log.error(f"💥 An error occurred during training: {e}", exc_info=True)