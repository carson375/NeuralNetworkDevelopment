import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import ConstellationNet
from src.dataset import load_and_preprocess_data
from sklearn.cluster import KMeans
import argparse

def plot_full_constellation(X_test, y_pred, scaler, is_big_dipper=False):
    """
    Plots the constellation using real coordinates and smart ordering.
    """
    plt.figure(figsize=(14, 10))
    
    # 1. Pure Deep Space Background
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    # 2. Extract Top Predicted Stars
    top_indices = np.argsort(y_pred)[-200:]
    top_stars_std = X_test.numpy()[top_indices]
    
    # Cluster to find the "Main Stars"
    kmeans = KMeans(n_clusters=7, n_init=30).fit(top_stars_std[:, :2])
    centers_std = kmeans.cluster_centers_
    
    # 3. Inverse Transform to get real RA/Dec
    # We create a dummy array to satisfy the 8-feature scaler
    dummy = np.zeros((7, 8))
    dummy[:, :2] = centers_std
    centers_real = scaler.inverse_transform(dummy)[:, :2]
    
    # 4. Smart Ordering (Sort by RA - Feature 0)
    # The Big Dipper handle starts at high RA and goes to low RA
    centers_real = centers_real[centers_real[:, 0].argsort()[::-1]]
    
    # 5. Connect the Dots
    if is_big_dipper:
        # After RA sort:
        # Index 0: Alkaid, 1: Mizar, 2: Alioth, 3: Megrez
        # The remaining 3 (4, 5, 6) are Phecda, Dubhe, Merak (in some order)
        
        # We know index 3 is Megrez. Let's find the bowl stars and sort them to form a box.
        handle = [0, 1, 2, 3]
        
        # The bowl consists of 3 (Megrez) and the others
        bowl_indices = [3, 4, 5, 6]
        bowl_stars = centers_real[bowl_indices]
        
        # To form a non-self-intersecting box (polygon), we can sort by angle from centroid
        centroid = np.mean(bowl_stars, axis=0)
        angles = np.arctan2(bowl_stars[:, 1] - centroid[1], bowl_stars[:, 0] - centroid[0])
        sorted_bowl_indices = np.array(bowl_indices)[np.argsort(angles)]
        
        # Ensure Megrez (index 3) is the start/end of the bowl path
        # Find where 3 is in the sorted list and rotate it
        start_idx = np.where(sorted_bowl_indices == 3)[0][0]
        final_bowl_indices = np.roll(sorted_bowl_indices, -start_idx)
        final_bowl_indices = np.append(final_bowl_indices, 3) # Close the loop
        
        # Draw Handle
        for i in range(len(handle)-1):
            plt.plot(centers_real[handle[i:i+2], 0], centers_real[handle[i:i+2], 1], 
                     c='cyan', alpha=0.6, linewidth=2, zorder=1)
        
        # Draw Bowl
        for i in range(len(final_bowl_indices)-1):
            idx1 = final_bowl_indices[i]
            idx2 = final_bowl_indices[i+1]
            plt.plot(centers_real[[idx1, idx2], 0], centers_real[[idx1, idx2], 1], 
                     c='cyan', alpha=0.6, linewidth=2, zorder=1)
    else:
        # Fallback: Just draw the stars if not Big Dipper
        pass

    # 6. Plot the Stars with "Deep Space Glow"
    plt.scatter(centers_real[:, 0], centers_real[:, 1], 
                c='cyan', s=600, alpha=0.15, zorder=2) # Deep Glow
    plt.scatter(centers_real[:, 0], centers_real[:, 1], 
                c='white', s=150, alpha=0.3, zorder=3) # Inner Glow
    plt.scatter(centers_real[:, 0], centers_real[:, 1], 
                c='gold', s=120, edgecolors='white', marker='*', zorder=4) # Star Core

    plt.title("Neural Network Output: Big Dipper Reconstruction", fontsize=24, color='white', y=1.05)
    plt.axis('off')
    
    # Invert RA axis (Astronomical standard: RA increases to the left)
    plt.gca().invert_xaxis()
    
    save_path = "results/plots/constellation_big_dipper_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Final true-scale map saved to {save_path}")
    plt.show()

def run_full_visualization(data_dir="data/raw"):
    X_tr, y_tr, X_val, y_val, X_ts, scaler = load_and_preprocess_data(raw_data_dir=data_dir)
    model = ConstellationNet(nin=X_ts.shape[1], nout=1)
    
    model_name = "constellation_model.pt"
    is_big_dipper = False
    if "bigdipper" in data_dir:
        model_name = "bigdipper_model.pt"
        is_big_dipper = True
    model_path = os.path.join("models/saved_models", model_name)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            outputs = model(X_ts)
            y_pred = outputs.numpy().ravel()
            plot_full_constellation(X_ts, y_pred, scaler, is_big_dipper=is_big_dipper)
    else:
        print(f"Model file not found at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Constellation Visualization")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory for raw data")
    args = parser.parse_args()
    run_full_visualization(data_dir=args.data_dir)
