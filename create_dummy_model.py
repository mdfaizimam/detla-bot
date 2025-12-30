from tft_model import TFTPredictor
import logging

logging.basicConfig(level=logging.INFO)

print("⚡ Creating 256-Dim Dummy Model for Immediate Launch...")
# Initialize with defaults (d_model=256, seq_len=60, pred_len=7)
pred = TFTPredictor(max_encoder_length=60, max_prediction_length=7)
pred.build_model(None) # Initialize random weights
pred.save("model_institutional/best_sharpe_model.pth")
print("✅ Fixed Model Saved.")
