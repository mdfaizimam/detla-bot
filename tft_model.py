# --- detla-bot/tft_model.py ---
# 🧠 TEMPORAL FUSION TRANSFORMER (World Class Edition)
# Features: Gated Residual Networks, Variable Selection, LSTM-Transformer Hybrid.
# Fixes: Target Scaling (*1000) to prevent Model Collapse.

import logging
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRANSFORMER]: %(message)s")
log = logging.getLogger("tft_model")

# --- 1. Gated Residual Network (The Building Block) ---
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        
        # Skip connection projection if sizes differ
        self.res_proj = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x):
        original_x = x
        residual = self.res_proj(x) if self.res_proj else x
        x = self.fc1(x)
        x = self.elu1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        gate = torch.sigmoid(self.gate(original_x))
        return self.norm(residual + gate * x)

# --- 2. Variable Selection Network (The Filter) ---
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Individual GRNs for each feature
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout) 
            for _ in range(input_dim)
        ])
        
        # GRN to calculate weights
        self.flattened_grn = GatedResidualNetwork(input_dim * hidden_size, hidden_size, input_dim, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        var_outputs = []
        for i in range(self.input_dim):
            feat = x[:, :, i:i+1]
            var_outputs.append(self.single_variable_grns[i](feat))
        
        var_outputs = torch.stack(var_outputs, dim=2)
        flat = var_outputs.view(x.size(0), x.size(1), -1)
        weights = self.flattened_grn(flat)
        weights = self.softmax(weights)
        combined = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)
        return combined, weights

# --- 3. The Full Model ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=4, output_dim=7, dropout=0.1):
        super().__init__()
        self.vsn = VariableSelectionNetwork(input_dim, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.grn_out = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x_emb, weights = self.vsn(x) 
        x_lstm, _ = self.lstm(x_emb)
        x_pos = x_lstm + self.pos_encoder[:, :x.size(1), :]
        x_trans = self.transformer_encoder(x_pos)
        last_hidden = self.grn_out(x_trans[:, -1, :])
        prediction = self.decoder(last_hidden)
        return prediction

# --- Wrapper Classes ---
class CryptoDataset(Dataset):
    def __init__(self, data, seq_len=60, pred_len=7, target="close_log_ret"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        exclude = ['timestamp', 'symbol', 'base_asset', 'time_idx', target]
        
        # ✅ ROBUST FEATURES: Force conversion to numeric to avoid "object" dtype issues
        potential_features = [c for c in data.columns if c not in exclude]
        self.feature_cols = []
        
        # Verify valid features
        clean_data = data.copy()
        for c in potential_features:
            try:
                # Force numeric
                clean_data[c] = pd.to_numeric(clean_data[c], errors='coerce').fillna(0.0)
                # Check if it has variance or valid magnitude? (Optional)
                self.feature_cols.append(c)
            except Exception:
                pass
        
        if not self.feature_cols:
            logging.error(f"❌ CryptoDataset found 0 valid features! Input Columns: {data.columns.tolist()}")
            # Fallback to prevent crash (though model will output Garbage)
            self.features = np.zeros((len(data), 1), dtype=np.float32)
        else:
            self.features = clean_data[self.feature_cols].values.astype(np.float32)
        
        # ✅ FIX: SCALING TARGETS BY 1000x TO PREVENT MODEL COLLAPSE
        # Log returns are tiny (e.g. 0.0005). Neural Nets hate tiny numbers.
        # We multiply by 1000 so the model sees 0.5 instead.
        if target in data.columns:
            self.targets = (pd.to_numeric(data[target], errors='coerce').fillna(0.0).values.astype(np.float32) * 1000.0)
        else:
             self.targets = np.zeros(len(data), dtype=np.float32)

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)

class TFTPredictor:
    def __init__(self, max_encoder_length=60, max_prediction_length=7, hidden_size=128, **kwargs):
        self.seq_len = max_encoder_length
        self.pred_len = max_prediction_length
        self.hidden_size = hidden_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(self, df: pd.DataFrame):
        self.dataset = CryptoDataset(df, seq_len=self.seq_len, pred_len=self.pred_len)
        return self.dataset
        
    def build_model(self, dataset):
        input_dim = len(dataset.feature_cols) if dataset else 20
        self.model = TimeSeriesTransformer(input_dim=input_dim, d_model=self.hidden_size, output_dim=self.pred_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() 
        
    def train(self, max_epochs=1, batch_size=64):
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

    def predict(self, df):
        self.model.eval()
        ds = CryptoDataset(df, seq_len=self.seq_len, pred_len=0) 
        features = ds.features 
        last_seq = features[-self.seq_len:]
        x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(x)
            
        # ✅ FIX: RESCALE PREDICTIONS BACK TO NORMAL
        # We trained on 1000x targets, so we must divide by 1000 to get real log returns.
        final_preds = preds.cpu().numpy() / 1000.0
        return final_preds

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'max_encoder_length': self.seq_len,
                'max_prediction_length': self.pred_len,
                'hidden_size': self.hidden_size,
                'input_dim': self.model.vsn.input_dim if self.model else None
            }
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            self.seq_len = config['max_encoder_length']
            self.pred_len = config['max_prediction_length']
            self.hidden_size = config['hidden_size']
            input_dim = config.get('input_dim', 20)
            
            self.model = TimeSeriesTransformer(
                input_dim=input_dim, 
                d_model=self.hidden_size, 
                output_dim=self.pred_len
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            log.info(f"✅ Loaded TFT model from checkpoint (Input dim: {input_dim})")
            
        else:
            if self.model is None:
                log.warning("⚠️  Loading legacy model without config. Assuming input_dim=20 (RISKY).")
                self.model = TimeSeriesTransformer(input_dim=20, d_model=self.hidden_size, output_dim=self.pred_len).to(self.device)
            
            self.model.load_state_dict(checkpoint)
            log.info("✅ Loaded legacy TFT model")
            
        self.model.eval()