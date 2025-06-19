import numpy as np
from numba import njit

@njit
def apply_motion_model(particles: np.ndarray, delta: np.ndarray, noise_std: np.ndarray) -> np.ndarray:
    """
    Propagate all particles using odometry delta and Gaussian noise.
    Args:
        particles: (N, 3) array of [x, y, theta] particles.
        delta: (3,) array of [dx, dy, dtheta] odometry.
        noise_std: (3,) array of stddev for [x, y, theta] noise.
    Returns:
        Updated (N, 3) array of particles.
    """
    N = particles.shape[0]
    updated = np.empty_like(particles)
    for i in range(N):
        dx = delta[0] + np.random.normal(0, noise_std[0])
        dy = delta[1] + np.random.normal(0, noise_std[1])
        dtheta = delta[2] + np.random.normal(0, noise_std[2])
        # Update state
        updated[i, 0] = particles[i, 0] + dx * np.cos(particles[i, 2]) - dy * np.sin(particles[i, 2])
        updated[i, 1] = particles[i, 1] + dx * np.sin(particles[i, 2]) + dy * np.cos(particles[i, 2])
        updated[i, 2] = particles[i, 2] + dtheta
    return updated 
