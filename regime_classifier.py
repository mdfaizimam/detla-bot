import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import joblib
import logging
import os

log = logging.getLogger("RegimeClassifier")

class RegimeClassifier:
    """
    Classifies market regimes using Gaussian Mixture Models (GMM).
    Identifies states like:
    - Low Volatility / Trending (Bull/calm)
    - High Volatility (Correction)
    - Extreme Volatility (Crash)
    """
    def __init__(self, n_components=3, model_dir="model_institutional"):
        self.n_components = n_components
        self.model_dir = model_dir
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.regime_map = {} # Maps cluster ID to meaning (0: Low Vol, etc)

    def prepare_features(self, df: pd.DataFrame):
        """Extracts and scales features for regime classification"""
        # Ensure we have required features
        df = df.copy()
        
        # Calculate if missing
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            
        if 'vol_20' not in df.columns:
            df['vol_20'] = df['log_ret'].rolling(20).std().fillna(0)
            
        # Features for clustering: Volatility is the primary driver of regime
        features = df[['log_ret', 'vol_20']].replace([np.inf, -np.inf], np.nan).dropna()
        
        return features

    def fit(self, df: pd.DataFrame):
        """Fits the GMM to historical data"""
        log.info("Fitting Regime Classifier (GMM)...")
        features = self.prepare_features(df)
        
        if len(features) < 1000:
            log.warning("Insufficient data to fit Regime Classifier robustly.")
            return

        # Scale features
        X = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.model.fit(X)
        self.is_fitted = True
        
        # Identify Regimes
        # We assume the regime with Highest Average Volatility is "Crash/High Vol"
        means = self.model.means_
        # valid index of 'vol_20' in X (it was the 2nd feature)
        vol_col_idx = 1 
        
        vol_means = means[:, vol_col_idx]
        sorted_indices = np.argsort(vol_means)
        
        # Map: 0 -> Lowest Vol, 1 -> Mid Vol, 2 -> Highest Vol
        self.regime_map = {
            sorted_indices[0]: "Low Vol (Calm/Trend)",
            sorted_indices[1]: "Med Vol (Correction)",
            sorted_indices[2]: "High Vol (Crash/Extreme)"
        }
        
        log.info(f"Regime Map Identified: {self.regime_map}")
        self.save()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicts regime for new data. Returns df with 'regime', 'regime_prob', 'regime_desc'"""
        if not self.is_fitted:
            log.warning("Model not fitted. Returning empty.")
            return df
            
        features = self.prepare_features(df)
        X = self.scaler.transform(features)
        
        # Predict clusters
        regimes = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        # Align with original index (features might be smaller due to dropna)
        result_df = df.copy()
        result_df.loc[features.index, 'regime'] = regimes
        result_df.loc[features.index, 'regime_prob'] = np.max(probs, axis=1)
        
        # Map descriptions
        result_df['regime_desc'] = result_df['regime'].map(self.regime_map)
        
        return result_df

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, "regime_gmm.joblib")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'regime_map': self.regime_map
        }, path)
        log.info(f"Regime classifier saved to {path}")

    def load(self):
        path = os.path.join(self.model_dir, "regime_gmm.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.regime_map = data.get('regime_map', {})
            self.is_fitted = True
            log.info(f"Regime classifier loaded from {path}")
            return True
        return False

if __name__ == "__main__":
    # Test Run
    logging.basicConfig(level=logging.INFO)
    
    # Generate dummy data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'close': np.cumprod(1 + np.random.normal(0, 0.001, 1000)),
        'time': dates
    })
    
    # Inject a "crash"
    df.iloc[800:850, 0] = df.iloc[800:850, 0] * (1 + np.random.normal(0, 0.02, 50))
    
    rc = RegimeClassifier(model_dir=".")
    rc.fit(df)
    res = rc.predict(df)
    print(res[['time', 'close', 'regime', 'regime_desc']].tail(10))
