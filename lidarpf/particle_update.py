import numpy as np
from numba import njit

from .types import (
    ParticleArray,
    WeightArray,
    OccupancyGridArray,
    LidarScanArray,
    OdomUpdateArray,
    OdomNoiseArray,
    X,
    Y,
    THETA,
)


@njit
def chassis_odom_update(
    particles: ParticleArray,
    odometry: OdomUpdateArray,
    noise: OdomNoiseArray,
    max_height: float,
    max_width: float,
) -> None:
    N = particles.shape[0]

    x_delta = odometry[X]
    y_delta = odometry[Y]
    theta_delta = odometry[THETA]

    x_std = noise[X]
    y_std = noise[Y]
    theta_std = noise[THETA]

    particles[:, X] += x_delta + np.random.normal(0, x_std, N)
    particles[:, Y] += y_delta + np.random.normal(0, y_std, N)
    particles[:, THETA] += theta_delta + np.random.normal(0, theta_std, N)

    np.clip(particles[:, X], 0, max_height, out=particles[:, X])
    np.clip(particles[:, Y], 0, max_width, out=particles[:, Y])
    particles[:, THETA] = np.fabs(np.fmod(particles[:, THETA], 2 * np.pi))


@njit(parallel=True)
def scan_update(
    particles: ParticleArray,
    scan: LidarScanArray,
    occupancy_grid: OccupancyGridArray,
    occupancy_grid_index_scalar: float,
    lidar_std_dev: float,
) -> WeightArray:
    """
    """

    N = particles.shape[0]
    num_beams = scan.shape[0]

    likelihoods = np.ones(N, dtype=np.float32)

    # Iterate over each particle and compute the likelihood based on the scan
    for i in range(N):
        particle = particles[i]
        x, y, theta = particle[X], particle[Y], particle[THETA]

        x_idx = int(x * occupancy_grid_index_scalar)
        y_idx = int(y * occupancy_grid_index_scalar)
        x_idx = np.clip(x_idx, 0, occupancy_grid.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, occupancy_grid.shape[1] - 1)

        # Compute probability of each measurement in the scan
        for j in range(num_beams):
            distance, angle = scan[j]

            angle += theta
            angle_idx = int(angle / (2 * np.pi) * occupancy_grid.shape[2])
            angle_idx = np.clip(angle_idx, 0, occupancy_grid.shape[2] - 1)

            expected_distance = occupancy_grid[x_idx, y_idx, angle_idx]
            diff = distance - expected_distance
            likelihood = np.exp(-0.5 * ((diff) ** 2) / (lidar_std_dev**2))

            likelihoods[i] *= likelihood

    # Normalize likelihoods
    likelihoods = likelihoods / np.sum(likelihoods)

    return likelihoods
