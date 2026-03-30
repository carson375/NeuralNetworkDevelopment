import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import ConstellationNet
from src.dataset import load_and_preprocess_data

def visualize_stars(X, y_true=None, y_pred=None, num_samples=5):
    """
    Visualizes star data samples. 
    Assuming features include some form of coordinates (RA/Dec).
    """
    plt.figure(figsize=(15, 5))
    
    # We'll take the first two features as coordinates (RA, Dec proxy)
    # Even if they aren't exactly RA/Dec, they represent the star's "position" in feature space
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        
        # Plotting the "star field" for this sample
        # Since each row is one "instance" (likely a set of features for one star or a small group)
        # We can visualize where this star sits relative to others in the batch
        
        star_x = X[i, 0].item()
        star_y = X[i, 1].item()
        
        # Plot the star
        plt.scatter(star_x, star_y, c='gold', s=100, edgecolors='white', marker='*')
        
        title = f"Sample {i+1}"
        if y_true is not None:
            title += f"\nTrue: {int(y_true[i].item())}"
        if y_pred is not None:
            title += f"\nPred: {y_pred[i]:.2f}"
            
        plt.title(title)
        plt.xlabel("Feat 0")
        plt.ylabel("Feat 1")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/star_visualization.png")
    print("Star visualization saved to results/plots/star_visualization.png")
    plt.show()

def run_visual_prediction():
    # Load model and data
    X_tr, y_tr, X_val, y_val, X_ts, scaler = load_and_preprocess_data()
    
    model = ConstellationNet(nin=X_tr.shape[1], nout=1)
    model_path = "models/saved_models/constellation_model.pt"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():
            # Get predictions for the first few validation samples
            samples_to_show = 5
            X_sample = X_val[:samples_to_show]
            y_true_sample = y_val[:samples_to_show]
            
            outputs = model(X_sample)
            y_pred = outputs.numpy().ravel()
            
            print("\n--- Visual Prediction Report ---")
            for i in range(samples_to_show):
                status = "Correct" if round(y_pred[i]) == y_true_sample[i] else "Incorrect"
                print(f"Sample {i+1}: True Class: {int(y_true_sample[i])} | Confidence: {y_pred[i]:.4f} | Result: {status}")
            
            # Generate the plot
            visualize_stars(X_sample, y_true_sample, y_pred, num_samples=samples_to_show)
    else:
        print("Model not found. Please train the model first.")

if __name__ == "__main__":
    run_visual_prediction()
