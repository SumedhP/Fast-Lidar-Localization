import numpy as np
from numba import njit

@njit
def compute_likelihoods(particles: np.ndarray, scan: np.ndarray, lookup_table: np.ndarray, map_resolution: float, sensor_params: dict) -> np.ndarray:
    """
    Compute likelihoods for all particles given a LiDAR scan.
    Args:
        particles: (N, 3) array of [x, y, theta] particles.
        scan: (M, 2) array of [distance, angle] LiDAR rays.
        lookup_table: Precomputed distance table (H, W, A).
        map_resolution: Size of each map cell (meters).
        sensor_params: Dict of sensor model parameters (noise, max range, etc).
    Returns:
        (N,) array of likelihoods.
    """
    N = particles.shape[0]
    M = scan.shape[0]
    H, W, A = lookup_table.shape
    sigma = sensor_params.get('sigma', 0.2)
    max_range = sensor_params.get('max_range', 10.0)
    num_angles = A
    likelihoods = np.ones(N, dtype=np.float32)
    for i in range(N):
        x = particles[i, 0] / map_resolution
        y = particles[i, 1] / map_resolution
        theta = particles[i, 2]
        if x < 0 or y < 0 or x >= H or y >= W:
            likelihoods[i] = 1e-9
            continue
        total_log_prob = 0.0
        for j in range(M):
            obs_dist = scan[j, 0]
            obs_angle = scan[j, 1]
            map_angle = (theta + obs_angle) % (2 * np.pi)
            angle_idx = int(map_angle / (2 * np.pi) * num_angles) % num_angles
            xi = int(x)
            yi = int(y)
            expected_dist = lookup_table[xi, yi, angle_idx] * map_resolution
            if obs_dist > max_range:
                continue
            error = obs_dist - expected_dist
            log_prob = -0.5 * (error / sigma) ** 2
            total_log_prob += log_prob
        likelihoods[i] = np.exp(total_log_prob)
    return likelihoods 
