# --- detla-bot/feature_engine.py ---
import pandas as pd
import numpy as np
import logging

log = logging.getLogger("feature_engine")

class FeatureEngine:
    def __init__(self):
        pass

    def add_features(self, df):
        """
        Applies the exact same feature engineering as generate_training_data.py
        Expects a DataFrame with at least 300 rows for rolling calculations.
        """
        try:
            df = df.copy()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            # 1. Price Action
            df['close_log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            
            # 2. Volume Z-Score (24h window = 288 periods)
            roll_mean_vol = df['volume'].rolling(288).mean()
            roll_std_vol = df['volume'].rolling(288).std().replace(0, 1)
            df['vol_zscore'] = (df['volume'] - roll_mean_vol) / roll_std_vol

            # 3. Basis (If spot data available, else 0)
            if 'spot_close' in df.columns:
                df['basis'] = (df['close'] - df['spot_close']) / df['spot_close']
            else:
                df['basis'] = 0.0

            # 4. Order Book Imbalance Proxy
            df['obi'] = np.sign(df['basis']) * np.log1p(df['volume']) / 10.0

            # 5. Funding ROC
            if 'funding_rate' in df.columns:
                df['funding_roc'] = df['funding_rate'].diff().fillna(0) * 1000
            else:
                df['funding_roc'] = 0.0

            # 6. CVD / Order Flow Features (The Alpha)
            if 'taker_buy_vol' in df.columns and 'spot_volume' in df.columns:
                vol_delta = (2 * df['taker_buy_vol']) - df['spot_volume']
                
                # Velocity
                df['cvd_velocity'] = vol_delta.rolling(12).mean() / df['spot_volume'].rolling(288).mean().replace(0, 1)
                
                # Z-Score
                roll_delta_mean = vol_delta.rolling(288).mean()
                roll_delta_std = vol_delta.rolling(288).std().replace(0, 1)
                df['cvd_zscore'] = (vol_delta - roll_delta_mean) / roll_delta_std
            else:
                df['cvd_velocity'] = 0.0
                df['cvd_zscore'] = 0.0

            # 7. Liquidity Levels (7d window = 2016)
            window_7d = 2016 
            roll_max = df['high'].rolling(window_7d).max()
            roll_min = df['low'].rolling(window_7d).min()
            df['dist_to_long_liq'] = ((df['close'] - roll_min) / df['close'])
            df['dist_to_short_liq'] = ((roll_max - df['close']) / df['close'])

            # 8. Macro Placeholders (If live macro feed not available)
            if 'dxy_close' not in df.columns: df['dxy_roc'] = 0.0
            if 'vix_close' not in df.columns: df['fear_greed_norm'] = 0.5 
            else: df['fear_greed_norm'] = (df['vix_close'] - 10) / 20.0

            # 9. Market Structure
            df['oi_pct_change'] = df['volume'].pct_change(288).fillna(0) # Proxy using vol if OI unavailable
            df['longShortRatio'] = 1.0 
            df['dist_to_poc'] = 0.0 

            # 10. Multi-Timeframe (MTF) - Resampled
            # We calculate this on the fly for the buffer
            # 1H Features
            df_1h = df.resample('1h', on='timestamp').agg({'close': 'last', 'high': 'max', 'low': 'min'}).dropna()
            if len(df_1h) > 50:
                ema_f = df_1h['close'].ewm(span=20).mean()
                ema_s = df_1h['close'].ewm(span=50).mean()
                df_1h['trend_bias_1h'] = (ema_f - ema_s) / ema_s
                
                tr = np.maximum(df_1h['high'] - df_1h['low'], np.abs(df_1h['high'] - df_1h['close'].shift(1)))
                df_1h['volatility_1h'] = tr.rolling(14).mean() / df_1h['close']
                
                # Merge back (Broadcast forward)
                df = pd.merge_asof(df, df_1h[['trend_bias_1h', 'volatility_1h']], on='timestamp', direction='backward')
                df['rsi_1h'] = 50.0 # Placeholder or implement full RSI calculation
            else:
                df['trend_bias_1h'] = 0.0
                df['volatility_1h'] = 0.0
                df['rsi_1h'] = 50.0

            # 4H Placeholders (Simpler for live inference speed if buffer is short)
            df['trend_bias_4h'] = df['trend_bias_1h'] 
            df['volatility_4h'] = df['volatility_1h']
            df['rsi_4h'] = 50.0

            # Cleanup
            df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            
            return df

        except Exception as e:
            log.error(f"Feature engineering failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            return df