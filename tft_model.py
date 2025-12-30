import logging
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRANSFORMER]: %(message)s")
log = logging.getLogger("tft_model")

class CryptoDataset(Dataset):
    def __init__(self, data, seq_len=60, pred_len=7, target="close_log_ret"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data
        self.target = target
        
        # Features to use: Added Multi-Modal Features
        self.feature_cols = [
            "close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", "vix", "obi",
            "dist_to_long_liq", "dist_to_short_liq", "funding_roc"
        ]
        
        # Check if correlation columns exist (dynamic per symbol)
        # We assume the dataframe has been standardized before reaching here
        # But to be safe, we look for 'corr_market_leader' if standardized, or specific columns
        if "corr_BTCUSD" in data.columns:
            self.feature_cols.append("corr_BTCUSD") # If we are trading ALT
        elif "corr_ETHUSD" in data.columns:
            self.feature_cols.append("corr_ETHUSD") # If we are trading BTC
            
        # Ensure all columns exist, fill missing with 0
        for col in self.feature_cols:
            if col not in data.columns:
                data[col] = 0.0
                
        self.features = data[self.feature_cols].values.astype(np.float32)
        self.targets = data[target].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        # Target for next 'pred_len' steps
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)

class TimeSeriesTransformer(nn.Module):
    """
    Custom Transformer for Time Series Forecasting.
    Replaces the heavy TemporalFusionTransformer with a lean, native PyTorch version.
    """
    def __init__(self, input_dim=6, d_model=256, nhead=8, num_layers=4, output_dim=7):
        super(TimeSeriesTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model)) # Simple Positional Encoding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # Predicts 7 steps ahead
        )
        
    def forward(self, src):
        # src shape: [batch, seq_len, input_dim]
        x = self.embedding(src)
        x = x + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(x)
        
        # We take the last hidden state to predict the future sequence
        last_hidden = output[:, -1, :] 
        prediction = self.decoder(last_hidden) 
        return prediction

class TFTPredictor:
    """
    Wrapper class to maintain API compatibility with the previous design.
    """
    def __init__(self, max_encoder_length=60, max_prediction_length=7):
        self.seq_len = max_encoder_length
        self.pred_len = max_prediction_length
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(self, df: pd.DataFrame):
        self.dataset = CryptoDataset(df, seq_len=self.seq_len, pred_len=self.pred_len)
        return self.dataset
        
    def build_model(self, dataset):
        # Determine input dim dynamically
        if dataset:
            input_dim = len(dataset.feature_cols)
        else:
            # Fallback for loading without dataset
            # We assume the standard set if not specified
            input_dim = 9 # Base new features
            
        self.model = TimeSeriesTransformer(input_dim=input_dim, output_dim=self.pred_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() 
        
    def train(self, max_epochs=1, batch_size=32):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        
        for epoch in range(max_epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            log.info(f"Epoch {epoch+1}/{max_epochs} - Loss: {total_loss/len(dataloader):.6f}")

    def predict(self, df: pd.DataFrame):
        # Taking the last sequence from DF
        self.model.eval()
        
        feature_cols = [
            "close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", "vix", "obi",
            "dist_to_long_liq", "dist_to_short_liq", "funding_roc"
        ]
        if "corr_BTCUSD" in df.columns: feature_cols.append("corr_BTCUSD")
        elif "corr_ETHUSD" in df.columns: feature_cols.append("corr_ETHUSD")
        
        # ✅ DYNAMIC DIMENSION CHECK
        # If loaded model expects fewer features (Legacy Model), truncate input.
        if self.model is not None and hasattr(self.model, "embedding"):
             expected_dim = self.model.embedding.in_features
             if len(feature_cols) > expected_dim:
                 feature_cols = feature_cols[:expected_dim]
        
        # Ensure cols exist
        for col in feature_cols:
            if col not in df.columns: df[col] = 0.0
            
        features = df[feature_cols].values
        last_seq = features[-self.seq_len:]
        x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(x)
        return preds.cpu().numpy()
        
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the entire dataframe using rolling windows.
        Returns array of shape (len(df),) containing the mean forecast signal.
        """
        self.model.eval()
        
        feature_cols = [
            "close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", "vix", "obi",
            "dist_to_long_liq", "dist_to_short_liq", "funding_roc"
        ]
        if "corr_BTCUSD" in df.columns: feature_cols.append("corr_BTCUSD")
        elif "corr_ETHUSD" in df.columns: feature_cols.append("corr_ETHUSD")
        
        # ✅ DYNAMIC DIMENSION CHECK
        if self.model is not None and hasattr(self.model, "embedding"):
             expected_dim = self.model.embedding.in_features
             if len(feature_cols) > expected_dim:
                 feature_cols = feature_cols[:expected_dim]

        for col in feature_cols:
             if col not in df.columns: df[col] = 0.0
             
        features = df[feature_cols].values
        
        # Prepare sliding windows efficiently
        if len(df) <= self.seq_len:
            return np.zeros(len(df))
            
        dataset = CryptoDataset(df, seq_len=self.seq_len, pred_len=1) 
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                out = self.model(x) 
                signal = out.mean(dim=1).cpu().numpy()
                all_preds.append(signal)
                
        if not all_preds:
            return np.zeros(len(df))
            
        flat_preds = np.concatenate(all_preds)
        padding = np.zeros(len(df) - len(flat_preds))
        final_preds = np.concatenate([padding, flat_preds])
        
        return final_preds

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Determine input dim from saved Config or Features OR State Dict
            saved_features = checkpoint.get('features', [])
            input_dim = len(saved_features) if saved_features else 6
            
            # Attempt to infer from state_dict if mismatch expected
            state_dict = checkpoint.get('model_state_dict', {})
            if "embedding.weight" in state_dict:
                 # weight shape is [d_model, input_dim]
                 inferred_dim = state_dict["embedding.weight"].shape[1]
                 if inferred_dim != input_dim:
                      log.info(f"🧠 Inferred Input Dim from checkpoint: {inferred_dim} (overriding default {input_dim})")
                      input_dim = inferred_dim

            self.model = TimeSeriesTransformer(input_dim=input_dim, output_dim=self.pred_len).to(self.device)
            self.model.load_state_dict(state_dict)
            log.info(f"✅ Model loaded from {path} (Input Dim: {input_dim})")
            
        except Exception as e:
            log.warning(f"⚠️ Failed to load model from {path}: {e}")
            log.warning("🔄 Re-initializing with fresh model for new architecture...")
            # Initialize with new default dimension (assuming we are upgrading)
            # This allows the bot to start fresh instead of crashing
            self.build_model(None) # Will use default new dim


    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'features': ["close_log_ret", "vol_zscore", "fear_greed_norm", "dxy_roc", "vix", "obi", "dist_to_long_liq", "dist_to_short_liq", "funding_roc", "corr_X"],
            'scaler': None, # We are not using a sklearn scaler in this pipeline yet
            'config': {'d_model': 256, 'seq_len': self.seq_len}
        }, path)
        log.info(f"Model saved to {path}")

if __name__ == "__main__":
    import os
    if os.path.exists("fused_data_sample.csv"):
        df = pd.read_csv("fused_data_sample.csv")
        pred = TFTPredictor()
        pred.prepare_data(df)
        pred.build_model(None)
        pred.train(max_epochs=2)
        
        p = pred.predict(df)
        print("Prediction (Next 7 steps):", p)
