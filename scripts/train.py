import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path to allow direct execution
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sonar_passivo import config
from sonar_passivo.dataset import SonarDataset
from sonar_passivo.model import SonarCNN
from sonar_passivo.engine import training_loop
from sonar_passivo.utils import save_training_logs

def main():
    parser = argparse.ArgumentParser(description="Train SonarCNN")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--mode", type=str, choices=["standard", "adversarial"], default="standard", 
                        help="Train on 'standard' data or mixed 'adversarial' data")
    
    args = parser.parse_args()
    
    config.ensure_dirs()
    
    # Select dataset folder based on mode
    train_dir_name = "train" if args.mode == "standard" else "adversarial_training"
    train_path = config.DATASET_DIR / train_dir_name
    val_path = config.DATASET_DIR / "validate"
    
    print(f"Loading data from {train_path}...")
    
    train_set = SonarDataset(str(train_path))
    val_set = SonarDataset(str(val_path))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = SonarCNN(num_classes=len(train_set.classes)).to(config.DEVICE)
    
    print(f"Model initialized for {len(train_set.classes)} classes.")
    
    # Train
    history_df = training_loop(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs,
        save_path=str(config.NETWORKS_DIR / "robust"),
        device=config.DEVICE
    )
    
    # Save Logs
    save_training_logs(history_df, str(config.LOGS_DIR / "training_log.csv"))

if __name__ == "__main__":
    main()
