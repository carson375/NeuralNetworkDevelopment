import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(raw_data_dir="data/raw", test_size=0.2):
    """
    Loads raw CSV data, standardizes it using RobustScaler, 
    and returns PyTorch DataLoaders.
    """
    Xtr_path = os.path.join(raw_data_dir, 'Xtr.csv')
    Xts_path = os.path.join(raw_data_dir, 'Xts.csv')
    ytr_path = os.path.join(raw_data_dir, 'ytr.csv')

    # Check if files exist
    for path in [Xtr_path, Xts_path, ytr_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required data file: {path}")

    # Load training data
    Xtr = np.loadtxt(Xtr_path, delimiter=",")
    Xts = np.loadtxt(Xts_path, delimiter=",")
    ytr = np.loadtxt(ytr_path, delimiter=",")

    # Standardize the data using RobustScaler
    scaler = preprocessing.RobustScaler()
    Xtr_standardized = scaler.fit_transform(Xtr)
    Xts_standardized = scaler.fit_transform(Xts) # Fit separately on test data like notebook
    ytr_standardized = ytr

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        Xtr_standardized, ytr_standardized, test_size=test_size, shuffle=True, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_torch = torch.Tensor(X_train)
    y_train_torch = torch.Tensor(y_train)
    X_val_torch = torch.Tensor(X_val)
    y_val_torch = torch.Tensor(y_val)
    X_test_torch = torch.Tensor(Xts_standardized)

    return (X_train_torch, y_train_torch, 
            X_val_torch, y_val_torch, 
            X_test_torch, scaler)

def get_dataloaders(X_train, y_train, X_val, y_val, batch_size=10):
    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data loading
    try:
        X_tr, y_tr, X_val, y_val, X_ts, scaler = load_and_preprocess_data()
        print(f"Loaded training data shape: {X_tr.shape}")
        print(f"Loaded validation data shape: {X_val.shape}")
        print(f"Loaded test data shape: {X_ts.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
