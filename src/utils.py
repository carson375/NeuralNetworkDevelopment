import numpy as np

def get_real_coordinates(centers_std, scaler):
    """
    Inverse transforms standardized centers to real RA/Dec.
    """
    dummy = np.zeros((len(centers_std), 8))
    dummy[:, :2] = centers_std
    return scaler.inverse_transform(dummy)[:, :2]

def sort_stars_by_angle(stars):
    """
    Sorts a set of points by their angle from their center to form a polygon.
    """
    centroid = np.mean(stars, axis=0)
    angles = np.arctan2(stars[:, 1] - centroid[1], stars[:, 0] - centroid[0])
    return np.argsort(angles)

def order_dipper_bowl(bowl_stars, megrez_idx=0):
    """
    Specifically orders the 4 bowl stars to ensure they form a rectangle.
    """
    angles = sort_stars_by_angle(bowl_stars)
    # Ensure Megrez is the start of the bowl loop
    start_pos = np.where(angles == megrez_idx)[0][0]
    final_indices = np.roll(angles, -start_pos)
    return np.append(final_indices, megrez_idx)
