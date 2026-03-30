import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from src.model import ConstellationNet
from src.dataset import load_and_preprocess_data
from src.utils import get_real_coordinates, sort_stars_by_angle
from sklearn.cluster import KMeans

def run_visualizer(data_dir="data/raw"):
    # 1. Load Data and Config
    with open("constellations.json", "r") as f:
        config = json.load(f)
    
    # Identify which config to use
    c_key = os.path.basename(data_dir)
    c_config = config.get(c_key, config["default"])
    
    X_tr, y_tr, X_val, y_val, X_ts, scaler = load_and_preprocess_data(raw_data_dir=data_dir)
    
    # 2. Load Model
    model_name = "bigdipper_model.pt" if c_key == "bigdipper" else "constellation_model.pt"
    model_path = os.path.join("models/saved_models", model_name)
    
    model = ConstellationNet(nin=X_ts.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 3. Predict and Cluster
    with torch.no_grad():
        y_pred = model(X_ts).numpy().ravel()
    
    top_indices = np.argsort(y_pred)[-200:]
    top_stars_std = X_ts.numpy()[top_indices, :2]
    
    kmeans = KMeans(n_clusters=c_config["num_stars"], n_init=30).fit(top_stars_std)
    centers_real = get_real_coordinates(kmeans.cluster_centers_, scaler)
    
    # Sort by RA for easier handle/bowl identification
    centers_real = centers_real[centers_real[:, 0].argsort()[::-1]]
    
    # 4. Drawing Logic
    plt.figure(figsize=(14, 10))
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    if c_config["draw_mode"] == "big_dipper_special":
        # Handle
        handle = centers_real[:4]
        plt.plot(handle[:, 0], handle[:, 1], c='cyan', alpha=0.6, linewidth=2)
        
        # Bowl (Smart Angle Sort)
        bowl_stars = centers_real[3:]
        bowl_angles = sort_stars_by_angle(bowl_stars)
        # Ensure Megrez is start
        m_idx = np.where(bowl_angles == 0)[0][0]
        bowl_angles = np.roll(bowl_angles, -m_idx)
        bowl_angles = np.append(bowl_angles, 0) # Close loop
        
        plt.plot(bowl_stars[bowl_angles, 0], bowl_stars[bowl_angles, 1], c='cyan', alpha=0.6, linewidth=2)
    
    # 5. Plot the Stars
    plt.scatter(centers_real[:, 0], centers_real[:, 1], c='cyan', s=500, alpha=0.1)
    plt.scatter(centers_real[:, 0], centers_real[:, 1], c='gold', s=100, edgecolors='white', marker='*')
    
    plt.title(f"Final Output: {c_config['name']}", color='white', fontsize=22)
    plt.axis('off')
    plt.gca().invert_xaxis() # Sky standard
    
    save_path = f"results/plots/{c_key}_final_config.png"
    plt.savefig(save_path, dpi=300, facecolor='black')
    print(f"Successfully generated {save_path} using configuration.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/bigdipper")
    args = parser.parse_args()
    run_visualizer(data_dir=args.data_dir)
