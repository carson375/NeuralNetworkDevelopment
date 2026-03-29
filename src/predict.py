import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from .model import ConstellationNet
from .dataset import load_and_preprocess_data

def predict_and_save(model, X_test_torch, output_path="results/yts_hat_neural.csv"):
    """
    Generates predictions on the test set and saves them to a CSV file.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_torch)
        yts_hat = outputs.data.detach().numpy().ravel()
        
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as CSV with IDs
    df = pd.DataFrame(data={'Id': np.arange(len(yts_hat)),
                            'Label': yts_hat})
    df.to_csv(output_path, index=False)
    print(f"Test label confidences saved in {output_path}")
    return yts_hat

def calculate_auc(model, X_val_torch, y_val_torch):
    """
    Calculates the ROC AUC score on the validation set.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_torch)
        predict = outputs.detach().numpy().ravel()
    
    auc = roc_auc_score(y_val_torch.numpy(), predict)
    print(f"Validation AUC: {auc:.4f}")
    return auc

if __name__ == "__main__":
    # Load model and data
    try:
        X_tr, y_tr, X_val, y_val, X_ts, _ = load_and_preprocess_data()
        
        # Load the latest trained model
        model = ConstellationNet(nin=X_tr.shape[1], nout=1)
        model_path = "models/saved_models/constellation_model.pt"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
            
            # Run evaluation
            calculate_auc(model, X_val, y_val)
            predict_and_save(model, X_ts)
        else:
            print(f"Model file not found at {model_path}. Please train the model first.")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
