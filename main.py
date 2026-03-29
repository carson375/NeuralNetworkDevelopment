import os
import torch
import argparse
from src.model import ConstellationNet
from src.dataset import load_and_preprocess_data, get_dataloaders
from src.train import train_model
from src.predict import calculate_auc, predict_and_save

def run_pipeline(epochs=10, lr=1e-3, batch_size=10, force_train=False):
    """
    Main orchestration function for the Constellation Classification project.
    """
    # 1. Load and Prepare Data
    print("--- Loading and Preprocessing Data ---")
    X_tr, y_tr, X_val, y_val, X_ts, scaler = load_and_preprocess_data()
    train_loader, val_loader = get_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=batch_size)
    print(f"Loaded {len(X_tr)} training samples, {len(X_val)} validation samples, and {len(X_ts)} test samples.\n")

    # 2. Initialize Model
    nin = X_tr.shape[1]
    model = ConstellationNet(nin=nin, nout=1)
    model_path = "models/saved_models/constellation_model.pt"

    # 3. Training
    if not os.path.exists(model_path) or force_train:
        print("--- Training Model ---")
        train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
        print("Training complete.\n")
    else:
        print(f"--- Loading Pre-trained Model from {model_path} ---")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.\n")

    # 4. Evaluation
    print("--- Evaluating Model ---")
    calculate_auc(model, X_val, y_val)
    print("Evaluation complete.\n")

    # 5. Prediction
    print("--- Generating Predictions ---")
    predict_and_save(model, X_ts)
    print("Predictions saved to results/yts_hat_neural.csv.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constellation Classification Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--train", action="store_true", help="Force model training")

    args = parser.parse_args()
    
    run_pipeline(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, force_train=args.train)
