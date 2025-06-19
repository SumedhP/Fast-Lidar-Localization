import numpy as np
from numba import njit
from lidarpf.core.types import ParticleState


@njit(parallel=True)
def apply_motion_model(
    particles: np.ndarray, delta: np.ndarray, noise_std: np.ndarray
) -> np.ndarray:
    """
    Propagate all particles using odometry delta and Gaussian noise.

    Args:
        particles: (N, 3) array of [x, y, theta] particles.
        delta: (3,) array of [dx, dy, dtheta] odometry.
        noise_std: (3,) array of stddev for [x, y, theta] noise.

    Returns:
        Updated (N, 3) array of particles.
    """
    if len(particles.shape) != 2 or particles.shape[1] != 3:
        raise ValueError(f"Particles must be a (N, 3) array. Received shape: {particles.shape}")
    if len(delta) != 3:
        raise ValueError(f"Delta must be a (3,) array. Received shape: {delta.shape}")
    if len(noise_std) != 3:
        raise ValueError(f"Noise std must be a (3,) array. Received shape: {noise_std.shape}")

    N = particles.shape[0]

    dx = delta[ParticleState.X]
    dy = delta[ParticleState.Y]
    dtheta = delta[ParticleState.THETA]

    x_std = noise_std[ParticleState.X]
    y_std = noise_std[ParticleState.Y]
    theta_std = noise_std[ParticleState.THETA]

    particles[:, ParticleState.X] += dx + np.random.normal(0, x_std, N)
    particles[:, ParticleState.Y] += dy + np.random.normal(0, y_std, N)
    particles[:, ParticleState.THETA] += dtheta + np.random.normal(0, theta_std, N)

    # Ensure theta is within [0, 2*pi]
    particles[:, ParticleState.THETA] = np.mod(particles[:, ParticleState.THETA], 2 * np.pi)

    return particles
