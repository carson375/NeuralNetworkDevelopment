import numpy as np
import os

def generate_big_dipper_data(num_samples=5000, output_dir="data/raw/bigdipper"):
    """
    Generates synthetic star data representing the Big Dipper constellation.
    8 features: [RA, Dec, Parallax, PM_RA, PM_Dec, RV, Magnitude, Color]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Define Big Dipper coordinates (approximate relative RA/Dec)
    # The ladle of the Big Dipper: Alkaid, Mizar, Alioth, Megrez, Phecda, Merak, Dubhe
    coords = np.array([
        [13.79, 49.31], # Alkaid
        [13.40, 54.92], # Mizar
        [12.90, 55.96], # Alioth
        [12.25, 57.03], # Megrez
        [11.89, 53.69], # Phecda
        [11.03, 56.38], # Merak
        [11.06, 61.75]  # Dubhe
    ])
    
    # 2. Generate Background Data (Random noise)
    X_bg = np.random.randn(num_samples, 8)
    # Scale coordinates to match celestial ranges
    X_bg[:, 0] = np.random.uniform(10, 15, num_samples) # RA
    X_bg[:, 1] = np.random.uniform(45, 65, num_samples) # Dec
    y_bg = np.zeros(num_samples)
    
    # 3. Generate Big Dipper stars (The target constellation)
    num_dipper_stars = 200
    X_dipper = np.zeros((num_dipper_stars, 8))
    y_dipper = np.ones(num_dipper_stars)
    
    for i in range(num_dipper_stars):
        # Pick one of the 7 core points
        core_point = coords[np.random.randint(0, 7)]
        # Add slight noise to simulate star clusters
        X_dipper[i, 0:2] = core_point + np.random.normal(0, 0.1, 2)
        
        # Give them a distinct physical signature (Parallax, PM, Magnitude)
        # Big Dipper stars are mostly 70-120 light years away
        X_dipper[i, 2] = np.random.normal(12.0, 1.0) # Parallax
        X_dipper[i, 3:5] = np.random.normal(-15.0, 2.0, 2) # Proper Motion
        X_dipper[i, 6] = np.random.normal(2.0, 0.5) # Brightness (Magnitude)
        X_dipper[i, 7] = np.random.normal(0.0, 0.1) # Color
        
    # Combine
    X = np.vstack([X_bg, X_dipper])
    y = np.hstack([y_bg, y_dipper])
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Split into train and test
    split = int(0.8 * len(X))
    Xtr, Xts = X[:split], X[split:]
    ytr, yts = y[:split], y[split:]
    
    # Save
    np.savetxt(os.path.join(output_dir, "Xtr.csv"), Xtr, delimiter=",")
    np.savetxt(os.path.join(output_dir, "Xts.csv"), Xts, delimiter=",")
    np.savetxt(os.path.join(output_dir, "ytr.csv"), ytr, delimiter=",")
    
    print(f"Synthetic Big Dipper data generated in {output_dir}")
    print(f"Total stars: {len(X)} | Constellation stars: {num_dipper_stars}")

if __name__ == "__main__":
    generate_big_dipper_data()
