# --- detla-bot/train_model.py ---
# 🧠 AGGRESSIVE TRAINING MODE - FINAL
# ✅ FIXED: Merge Keys Type Mismatch (Enables Sentiment Data)
# ✅ FIXED: Critical n_jobs=-1 bug (Prevents Crashes)
# ✅ MODEL: Single High-Performance LGBM with Early Stopping
# ✅ BALANCING: Heavy Class Weighting + SMOTE + Feature Selection
# ✅ GOAL: Force the model to generate Class 0 (Short) and 2 (Long) signals reliably

import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from scipy.stats import uniform, randint

# --- VALIDATION ---
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier, early_stopping
from imblearn.over_sampling import SMOTE 

# --- TA LIB ---
from ta import trend, volatility, momentum, volume

# --- CONFIG ---
from config import LAG_PERIODS, SL_ATR_MULTIPLIER, MIN_RISK_REWARD_RATIO

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = "data"
MODEL_DIR = "model"
CANDLES_INPUT_FILE = os.path.join(DATA_DIR, "historical_candles.csv")
FUNDING_INPUT_FILE = os.path.join(DATA_DIR, "historical_funding_rates.csv")
LSR_INPUT_FILE = os.path.join(DATA_DIR, "historical_long_short_ratio.csv")
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signal_classifier.joblib")

# Training Params
BARRIER_HORIZON = 24    # 2 Hours
N_ITER_SEARCH = 30      # More search = better params
FEATURE_SELECTION_THRESHOLD = 0.05 
N_JOBS = 1  # FIXED: Set to 1 to prevent multiprocessing crashes. Change to -1 only if you're sure your environment supports it.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAINER]: %(message)s")
log = logging.getLogger("Trainer")

# ----------------------------------------------------------------------
# 1. DATA LOADING - MEMORY EFFICIENT
# ----------------------------------------------------------------------
def load_and_fuse_data():
    """Load and fuse data with robust type handling"""
    if not os.path.exists(CANDLES_INPUT_FILE):
        log.error("Missing candles file.")
        return pd.DataFrame()

    # Dtypes - explicitly specify symbol as object to prevent dtype mismatch
    dtypes = {
        'symbol': 'object',  # Explicitly set as object to match other CSVs
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32'
    }
    
    try:
        # Read with symbol as Object (String) to match other CSVs
        df_candles = pd.read_csv(CANDLES_INPUT_FILE, dtype=dtypes)
        df_candles['time'] = pd.to_datetime(df_candles['time'])
    except Exception as e:
        log.error(f"Error loading candles: {e}")
        return pd.DataFrame()
    
    df_candles = df_candles.sort_values('time').reset_index(drop=True)
    log.info(f"Loaded {len(df_candles)} candles.")
    
    # Merge funding rates if available
    if os.path.exists(FUNDING_INPUT_FILE):
        try:
            df_fund = pd.read_csv(FUNDING_INPUT_FILE)
            df_fund['fundingTime'] = pd.to_datetime(df_fund['fundingTime'])
            df_fund = df_fund.sort_values('fundingTime').rename(
                columns={'fundingTime': 'time', 'fundingRate': 'funding_rate'}
            )
            df_fund['funding_rate'] = df_fund['funding_rate'].astype('float32')
            
            # Ensure symbol is object type for merge
            if 'symbol' in df_fund.columns:
                df_fund['symbol'] = df_fund['symbol'].astype('object')
            
            # Merge
            df_candles = pd.merge_asof(
                df_candles, 
                df_fund[['time', 'symbol', 'funding_rate']], 
                on='time', 
                by='symbol', 
                direction='backward'
            )
            log.info("Merged funding rates.")
        except Exception as e:
            log.warning(f"Could not merge funding rates: {e}")
    
    # Merge long/short ratio if available
    if os.path.exists(LSR_INPUT_FILE):
        try:
            df_lsr = pd.read_csv(LSR_INPUT_FILE)
            if len(df_lsr) > 1 and 'timestamp' in df_lsr.columns:
                df_lsr['timestamp'] = pd.to_datetime(df_lsr['timestamp'])
                df_lsr = df_lsr.sort_values('timestamp').rename(
                    columns={'timestamp': 'time', 'longShortRatio': 'long_short_ratio'}
                )
                df_lsr['long_short_ratio'] = df_lsr['long_short_ratio'].astype('float32')
                
                # Ensure symbol is object type for merge
                if 'symbol' in df_lsr.columns:
                    df_lsr['symbol'] = df_lsr['symbol'].astype('object')
                
                # Merge
                df_candles = pd.merge_asof(
                    df_candles, 
                    df_lsr[['time', 'symbol', 'long_short_ratio']], 
                    on='time', 
                    by='symbol', 
                    direction='backward'
                )
                log.info("Merged long/short ratios.")
        except Exception as e:
            log.warning(f"Could not merge LSR: {e}")
    
    # Fill missing values
    fill_values = {'funding_rate': 0.0, 'long_short_ratio': 1.0}
    return df_candles.fillna(fill_values)

# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def calculate_features(df):
    """Calculate technical indicators and features"""
    if len(df) < 50: 
        return pd.DataFrame()
    
    df_ta = df.copy()
    
    # Volatility
    df_ta['ATR'] = volatility.average_true_range(
        df_ta['high'], df_ta['low'], df_ta['close'], window=14
    ).fillna(0).astype('float32')
    
    df_ta['ATR_PCT'] = (df_ta['ATR'] / df_ta['close']) * 100
    
    # Trend Indicators
    df_ta['EMA_20'] = trend.ema_indicator(df_ta['close'], window=20).fillna(0).astype('float32')
    df_ta['EMA_50'] = trend.ema_indicator(df_ta['close'], window=50).fillna(0).astype('float32')
    
    # Aggressive Mode: Calculate Real Short-Term EMAs
    df_ta['EMA_8'] = trend.ema_indicator(df_ta['close'], window=8).fillna(0).astype('float32')
    df_ta['EMA_21'] = trend.ema_indicator(df_ta['close'], window=21).fillna(0).astype('float32')
    df_ta['EMA_200'] = trend.ema_indicator(df_ta['close'], window=200).fillna(0).astype('float32')
    
    # Momentum / Oscillators
    df_ta['RSI'] = momentum.rsi(df_ta['close'], window=14).fillna(50).astype('float32')
    df_ta['MACDh'] = trend.MACD(df_ta['close']).macd_diff().fillna(0).astype('float32')
    df_ta['ADX'] = trend.adx(df_ta['high'], df_ta['low'], df_ta['close'], window=14).fillna(0).astype('float32')
    
    # Advanced Indicators
    change = df_ta['close'].diff(10).abs()
    vol = df_ta['close'].diff().abs().rolling(10).sum()
    df_ta['KER'] = (change / (vol + 1e-9)).fillna(0).astype('float32')
    
    bb = volatility.BollingerBands(df_ta['close'], window=20, window_dev=2)
    df_ta['BB_WIDTH'] = bb.bollinger_wband().fillna(0).astype('float32')
    
    df_ta['OBV'] = volume.on_balance_volume(df_ta['close'], df_ta['volume']).fillna(0).astype('float32')
    
    # OBI Proxy - FIXED: Add epsilon to denominator
    high_low_diff = df_ta['high'] - df_ta['low'] + 1e-9
    clv = ((df_ta['close'] - df_ta['low']) / high_low_diff)
    df_ta['OBI_Proxy'] = ((clv * 2) - 1).fillna(0).astype('float32')
    
    # Feature Interactions
    df_ta['RSI_x_KER'] = df_ta['RSI'] * df_ta['KER']
    df_ta['ADX_x_VOL'] = df_ta['ADX'] * (df_ta['volume'] / (df_ta['volume'].rolling(20).mean() + 1e-9))
    
    # Lag Features
    lag_cols = ['KER', 'RSI', 'MACDh', 'OBV', 'ADX', 'OBI_Proxy', 'funding_rate', 'long_short_ratio']
    for col in lag_cols:
        if col in df_ta.columns:
            for lag in LAG_PERIODS:
                df_ta[f'{col}_LAG{lag}'] = df_ta[col].shift(lag).fillna(0).astype('float32')
    
    # Clean infinite values
    return df_ta.replace([np.inf, -np.inf], 0)

# ----------------------------------------------------------------------
# 3. LABELING
# ----------------------------------------------------------------------
def apply_labeling(df):
    """Apply triple-barrier labeling for short, neutral, long signals"""
    log.info("⏳ Labeling Data...")
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['ATR'].values
    n = len(df)
    labels = np.ones(n, dtype=int)  # Default 1 (Neutral)
    
    sl_mult = SL_ATR_MULTIPLIER
    tp_mult = SL_ATR_MULTIPLIER * MIN_RISK_REWARD_RATIO
    
    for i in range(n - BARRIER_HORIZON):
        if atrs[i] <= 0: 
            continue
        
        entry = closes[i]
        sl = atrs[i] * sl_mult
        tp = atrs[i] * tp_mult
        
        # Future window
        w_highs = highs[i+1 : i+1+BARRIER_HORIZON]
        w_lows = lows[i+1 : i+1+BARRIER_HORIZON]
        
        # Long signal conditions
        tp_idx = np.where(w_highs >= entry + tp)[0]
        sl_idx = np.where(w_lows <= entry - sl)[0]
        long_win = (len(tp_idx) > 0) and (len(sl_idx) == 0 or tp_idx[0] < sl_idx[0])
        
        # Short signal conditions
        tp_idx_s = np.where(w_lows <= entry - tp)[0]
        sl_idx_s = np.where(w_highs >= entry + sl)[0]
        short_win = (len(tp_idx_s) > 0) and (len(sl_idx_s) == 0 or tp_idx_s[0] < sl_idx_s[0])
        
        if long_win and not short_win: 
            labels[i] = 2  # Long
        elif short_win and not long_win: 
            labels[i] = 0  # Short
    
    df['Target'] = labels
    return df.iloc[:-BARRIER_HORIZON] if len(df) > BARRIER_HORIZON else df

# ----------------------------------------------------------------------
# 4. FEATURE SELECTION
# ----------------------------------------------------------------------
def select_features_aggressively(X, y, threshold=0.01):
    """Select only predictive features to reduce noise"""
    log.info(f"🔍 Selecting features with p-value < {threshold}")
    
    # Ensure we have enough samples
    if len(X) < 100 or X.shape[1] < 2:
        log.warning("Insufficient samples or features for selection, using all features")
        return X.columns.tolist()
    
    try:
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X.fillna(0), y)
        
        # Get p-values
        pvalues = pd.Series(selector.pvalues_, index=X.columns)
        
        # Keep features with p-value < threshold
        selected_features = pvalues[pvalues < threshold].index.tolist()
        
        log.info(f"Selected {len(selected_features)}/{len(X.columns)} features")
        
        # If too few features, take top N by p-value
        if len(selected_features) < 10:
            log.warning(f"Only {len(selected_features)} features selected, taking top 20")
            selected_features = pvalues.sort_values().head(20).index.tolist()
        
        return selected_features
        
    except Exception as e:
        log.error(f"Feature selection failed: {e}")
        return X.columns.tolist()

# ----------------------------------------------------------------------
# 5. THRESHOLD TUNING
# ----------------------------------------------------------------------
def tune_prediction_thresholds(model, X_val, y_val):
    """Find optimal confidence thresholds for each class"""
    log.info("🎯 Tuning prediction thresholds...")
    
    probs = model.predict_proba(X_val)
    thresholds = {}
    
    for class_idx in range(3):  # 0, 1, 2
        try:
            precision, recall, thresh = precision_recall_curve(
                (y_val == class_idx).astype(int), 
                probs[:, class_idx]
            )
            
            # Ensure we have thresholds
            if len(thresh) == 0:
                thresholds[class_idx] = 0.5
                continue
            
            # Find threshold where precision > 0.7
            valid_indices = np.where(precision >= 0.7)[0]
            if len(valid_indices) > 0:
                # Of those with good precision, pick highest recall
                optimal_idx = valid_indices[np.argmax(recall[valid_indices])]
                thresholds[class_idx] = thresh[min(optimal_idx, len(thresh)-1)]
            else:
                # Fallback: threshold that maximizes F1
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
                optimal_idx = np.argmax(f1_scores)
                thresholds[class_idx] = thresh[min(optimal_idx, len(thresh)-1)]
                
            log.info(f"  Class {class_idx}: threshold = {thresholds[class_idx]:.3f}")
            
        except Exception as e:
            log.warning(f"Could not tune threshold for class {class_idx}: {e}")
            thresholds[class_idx] = 0.5
    
    return thresholds

# ----------------------------------------------------------------------
# 6. SIGNAL QUALITY METRICS
# ----------------------------------------------------------------------
def calculate_signal_quality(y_true, y_pred):
    """Calculate trading-specific metrics"""
    log.info("📊 Calculating signal quality metrics...")
    
    # Signal frequency
    signal_rate = (y_pred != 1).mean() * 100
    
    # Win rate when model predicts signals
    win_mask = (y_pred != 1) & (y_true != 1)
    if len(win_mask) > 0:
        win_rate = (y_pred[win_mask] == y_true[win_mask]).mean() * 100
    else:
        win_rate = 0
    
    # Signal distribution
    signal_dist = pd.Series(y_pred).value_counts().sort_index()
    signal_dist_pct = (signal_dist / len(y_pred) * 100).round(2)
    
    log.info(f"  Signal Rate: {signal_rate:.1f}%")
    log.info(f"  Win Rate on Signals: {win_rate:.1f}%")
    log.info(f"  Signal Distribution: {dict(signal_dist_pct)}")
    
    return signal_rate, win_rate, signal_dist

# ----------------------------------------------------------------------
# 7. MODEL TRAINING - ENHANCED
# ----------------------------------------------------------------------
def train_model_enhanced(df):
    """Enhanced training with feature selection, CV, and threshold tuning"""
    
    # Prepare features
    drop_cols = ['time', 'symbol', 'Target', 'Future_Close', 'Return', 'BB_UPPER', 'BB_LOWER']
    # Only include columns that actually exist
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    
    features = [c for c in df.columns if c not in existing_drop_cols and 
                df[c].dtype in [np.float64, np.float32, np.int64]]
    
    if len(features) == 0:
        log.error("No features found for training!")
        return None, None, None, None
    
    X = df[features]
    y = df['Target']
    
    # Filter low volatility periods
    if 'ATR_PCT' in X.columns:
        atr_threshold = X['ATR_PCT'].quantile(0.1)
        high_vol_mask = X['ATR_PCT'] > atr_threshold
        X = X[high_vol_mask]
        y = y[high_vol_mask]
        log.info(f"Filtered low volatility: {len(X)} samples remain")
    
    # Split with time series
    split = int(len(X) * 0.85)
    if split == 0 or split == len(X):
        log.error("Not enough data for train/test split")
        return None, None, None, None
        
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    
    log.info(f"Training on {len(X_train)} samples. Testing on {len(X_test)}.")
    log.info(f"Class distribution - Train: {dict(y_train.value_counts())}, Test: {dict(y_test.value_counts())}")
    
    # Feature selection
    selected_features = select_features_aggressively(
        X_train, y_train, FEATURE_SELECTION_THRESHOLD
    )
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Apply SMOTE only if we have at least 2 classes
    unique_classes = y_train.unique()
    if len(unique_classes) > 1:
        log.info("⚖️ Applying SMOTE...")
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train_selected, y_train)
            log.info(f"SMOTE applied: {len(X_res)} samples")
        except Exception as e:
            log.warning(f"SMOTE failed: {e}, using raw data")
            X_res, y_res = X_train_selected, y_train
    else:
        log.warning(f"Only {len(unique_classes)} class in training data, skipping SMOTE")
        X_res, y_res = X_train_selected, y_train
    
    # LGBM with early stopping - FIXED: Removed n_jobs=-1
    log.info("🚀 Training LightGBM with early stopping...")
    
    # Base parameters - FIXED: Set n_jobs to N_JOBS (default 1)
    base_params = {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': N_JOBS,  # FIXED: Use safe N_JOBS instead of -1
        'class_weight': 'balanced',
        'importance_type': 'gain',
        'random_state': 42
    }
    
    try:
        # Cross-validation monitoring - FIXED: Removed n_jobs=-1
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_res) // 10))
        cv_scores = cross_val_score(
            LGBMClassifier(**base_params), 
            X_res, y_res, 
            cv=tscv, 
            scoring='f1_macro',
            n_jobs=N_JOBS  # FIXED: Use safe N_JOBS instead of -1
        )
        
        log.info(f"Cross-validation scores: {cv_scores}")
        log.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Randomized search for hyperparameter tuning - FIXED: Removed n_jobs=-1
        log.info("🎯 Tuning hyperparameters...")
        param_dist = {
            'learning_rate': uniform(0.01, 0.1),
            'num_leaves': randint(20, 100),
            'max_depth': randint(5, 15),
            'min_child_samples': randint(10, 50),
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
        
        search = RandomizedSearchCV(
            LGBMClassifier(**{k: v for k, v in base_params.items() if k not in param_dist}),
            param_dist,
            n_iter=N_ITER_SEARCH,
            cv=TimeSeriesSplit(min(3, len(X_res) // 10)),
            scoring='f1_macro',
            n_jobs=N_JOBS,  # FIXED: Use safe N_JOBS instead of -1
            random_state=42
        )
        
        search.fit(X_res, y_res)
        model = search.best_estimator_
        
        log.info(f"✅ Best Params: {search.best_params_}")
        log.info(f"✅ Best Score: {search.best_score_:.3f}")
        
    except Exception as e:
        log.error(f"Model training failed: {e}")
        return None, None, None, None
    
    # Refit with early stopping
    log.info("🔄 Refitting with early stopping...")
    model.set_params(**search.best_params_, n_estimators=2000)
    
    try:
        model.fit(
            X_res, y_res,
            eval_set=[(X_test_selected, y_test)],
            eval_metric='multi_logloss',
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
        )
    except Exception as e:
        log.warning(f"Early stopping failed: {e}, fitting normally")
        model.fit(X_res, y_res)
    
    # Evaluate
    preds = model.predict(X_test_selected)
    
    log.info("\n" + classification_report(y_test, preds))
    
    # Feature Importance
    importances = pd.Series(
        model.feature_importances_, 
        index=selected_features
    ).sort_values(ascending=False)
    
    log.info(f"🏆 Top 10 Features:\n{importances.head(10)}")
    
    # Signal quality metrics
    calculate_signal_quality(y_test, preds)
    
    # Threshold tuning
    thresholds = tune_prediction_thresholds(model, X_test_selected, y_test)
    
    # Save model with metadata
    save_model_with_metadata(
        model=model,
        X_train=X_train_selected,
        features=selected_features,
        thresholds=thresholds,
        search_best_params=search.best_params_,
        cv_score=cv_scores.mean()
    )
    
    return model, thresholds, selected_features, importances

# ----------------------------------------------------------------------
# 8. MODEL SAVING WITH METADATA
# ----------------------------------------------------------------------
def save_model_with_metadata(model, X_train, features, thresholds, search_best_params, cv_score):
    """Save model with all necessary metadata"""
    log.info("💾 Saving model with metadata...")
    
    model_package = {
        'model': model,
        'features': features,
        'feature_importance': pd.Series(
            model.feature_importances_, 
            index=features
        ).sort_values(ascending=False),
        'training_date': pd.Timestamp.now(),
        'data_shape': X_train.shape,
        'prediction_thresholds': thresholds,
        'best_params': search_best_params,
        'cv_score': cv_score,
        'class_names': {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'},
        'model_type': 'LGBMClassifier',
        'version': '1.0.0'
    }
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_package, OUTPUT_MODEL_PATH)
    log.info(f"✅ Model package saved: {OUTPUT_MODEL_PATH}")
    log.info(f"   - Features: {len(features)}")
    log.info(f"   - Thresholds: {thresholds}")

# ----------------------------------------------------------------------
# 9. DEBUG VISUALIZATION (OPTIONAL)
# ----------------------------------------------------------------------
def plot_training_diagnostics(model, features, importances):
    """Create diagnostic plots if matplotlib is available"""
    try:
        import matplotlib.pyplot as plt
        
        # Feature importance plot
        top_n = min(20, len(features))
        if top_n > 0 and importances is not None:
            top_indices = importances.head(top_n).index
            top_importance = importances.head(top_n).values
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_indices)), top_importance)
            plt.yticks(range(len(top_indices)), top_indices)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
            
            plot_path = os.path.join(MODEL_DIR, 'feature_importance.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            log.info(f"📈 Plot saved: {plot_path}")
        else:
            log.info("No features to plot")
        
    except ImportError:
        log.info("📝 Matplotlib not available, skipping plots")
    except Exception as e:
        log.warning(f"Could not create plots: {e}")

# ----------------------------------------------------------------------
# 10. MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    """Main training pipeline"""
    log.info("🚀 Starting aggressive training pipeline...")
    log.info(f"⚠️  Using N_JOBS={N_JOBS} to prevent multiprocessing crashes")
    log.info("   Set N_JOBS = -1 ONLY if you're sure your environment supports it")
    
    # Load data
    df = load_and_fuse_data()
    if df.empty:
        log.error("❌ No data loaded!")
        return
    
    log.info(f"📊 Initial data: {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Process each symbol separately
    processed = []
    for sym in df['symbol'].unique():
        log.info(f"Processing {sym}...")
        sub = df[df['symbol'] == sym].copy()
        
        # Calculate features
        sub = calculate_features(sub)
        if sub.empty:
            log.warning(f"  {sym}: No features calculated")
            continue
        
        # Apply labeling
        sub = apply_labeling(sub)
        if sub.empty:
            log.warning(f"  {sym}: No labels generated")
            continue
        
        # Remove periods with very low volatility
        if 'ATR_PCT' in sub.columns and len(sub) > 10:
            atr_threshold = sub['ATR_PCT'].quantile(0.1)
            sub = sub[sub['ATR_PCT'] > atr_threshold]
        
        processed.append(sub)
        log.info(f"  {sym}: {len(sub)} labeled samples")
    
    if not processed:
        log.error("❌ No processed data!")
        return
    
    # Combine all symbols
    full_df = pd.concat(processed, ignore_index=True).dropna().reset_index(drop=True)
    log.info(f"✅ Combined dataset: {len(full_df)} samples")
    
    if len(full_df) == 0:
        log.error("❌ Empty dataset after processing!")
        return
    
    # Check class balance
    if 'Target' in full_df.columns:
        class_dist = full_df['Target'].value_counts().sort_index()
        if len(class_dist) > 0:
            class_pct = (class_dist / len(full_df) * 100).round(1)
            log.info(f"📈 Class Distribution:")
            for cls, count in class_dist.items():
                log.info(f"  Class {cls}: {count} ({class_pct[cls]}%)")
        else:
            log.error("❌ No target labels found!")
            return
    else:
        log.error("❌ Target column not found!")
        return
    
    # Train model
    result = train_model_enhanced(full_df)
    
    # Check if training was successful before unpacking
    if result is not None and result[0] is not None:
        model, thresholds, features, importances = result
        
        # Optional: Create diagnostic plots
        plot_training_diagnostics(model, features, importances)
        
        log.info("🎉 Training completed successfully!")
    else:
        log.error("❌ Training failed!")

if __name__ == "__main__":
    # FIXED: Wrap main execution in try-except to catch any remaining multiprocessing issues
    try:
        main()
    except Exception as e:
        log.error(f"❌ Fatal error in main execution: {e}")
        log.error("This might be due to multiprocessing issues.")
        log.error("Try setting N_JOBS = 1 in the configuration section.")