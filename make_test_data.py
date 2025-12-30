# --- detla-bot/make_test_data.py ---
import pandas as pd
import os
import shutil

# Files (Relative to current folder)
REAL_FILE = "fused_data_real.csv"
BACKUP_FILE = "fused_data_real_FULL.csv"

def make_test_data():
    if not os.path.exists(REAL_FILE):
        print(f"❌ Could not find {REAL_FILE} in {os.getcwd()}")
        print("   Did you run generate_training_data.py first?")
        return

    # 1. Backup the big file (if not already backed up)
    if not os.path.exists(BACKUP_FILE):
        print("📦 Backing up full dataset...")
        shutil.copy(REAL_FILE, BACKUP_FILE)
    else:
        print("ℹ️ Backup already exists, using it to generate test data.")

    # 2. Load and Slice
    print("✂️ Creating tiny test dataset (2000 rows)...")
    try:
        df = pd.read_csv(BACKUP_FILE)
        
        # Sort by time and take the last 2000 rows (most recent data)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        df = df.tail(2000)
        
        # 3. Save as the main file name
        df.to_csv(REAL_FILE, index=False)
        print(f"✅ Ready for testing! {REAL_FILE} is now small (2000 rows).")
        print("👉 Run: python train_hybrid.py --epochs 1 --steps 100")
        print("👉 To undo, run this script again and type 'restore'.")
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")

def restore_full_data():
    if os.path.exists(BACKUP_FILE):
        print("🔄 Restoring full dataset...")
        shutil.copy(BACKUP_FILE, REAL_FILE)
        print("✅ Full data restored.")
    else:
        print("❌ No backup found to restore (fused_data_real_FULL.csv missing).")

if __name__ == "__main__":
    x = input("Type 'test' to create small data, or 'restore' to get big data back: ")
    if x.strip().lower() == 'test':
        make_test_data()
    elif x.strip().lower() == 'restore':
        restore_full_data()
    else:
        print("Invalid input.")