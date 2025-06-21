import numpy as np
from numba import njit, prange

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


def chassis_odom_update(
    particles: ParticleArray,
    odometry: OdomUpdateArray,
    noise: OdomNoiseArray,
    max_height: float,
    max_width: float,
) -> None:
    """
    Update particle positions using odometry measurements with noise.

    This function applies odometry updates to all particles, adding Gaussian noise
    to simulate sensor uncertainty. Particles are constrained to stay within the
    specified map boundaries, and angles are normalized to [0, 2π).

    Args:
        particles: Array of shape (N, 3) containing particle states [x, y, theta].
                  Modified in-place.
        odometry: Array of shape (3,) containing odometry deltas [dx, dy, dtheta].
        noise: Array of shape (3,) containing standard deviations for Gaussian noise
               [std_x, std_y, std_theta].
        max_height: Maximum x-coordinate (height) of the map boundary.
        max_width: Maximum y-coordinate (width) of the map boundary.

    Note:
        - All angles are in radians
        - Particles outside boundaries are clipped to the boundary
        - Angle normalization uses absolute value and modulo 2π
    """
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


@njit(parallel=True, cache=True)
def chassis_odom_update_numba(
    particles: ParticleArray,
    odometry: OdomUpdateArray,
    noise: OdomNoiseArray,
    max_height: float,
    max_width: float,
) -> None:
    """
    Update particle positions using odometry measurements with noise.

    This function applies odometry updates to all particles, adding Gaussian noise
    to simulate sensor uncertainty. Particles are constrained to stay within the
    specified map boundaries, and angles are normalized to [0, 2π).

    Args:
        particles: Array of shape (N, 3) containing particle states [x, y, theta].
                  Modified in-place.
        odometry: Array of shape (3,) containing odometry deltas [dx, dy, dtheta].
        noise: Array of shape (3,) containing standard deviations for Gaussian noise
               [std_x, std_y, std_theta].
        max_height: Maximum x-coordinate (height) of the map boundary.
        max_width: Maximum y-coordinate (width) of the map boundary.

    Note:
        - All angles are in radians
        - Particles outside boundaries are clipped to the boundary
        - Angle normalization uses absolute value and modulo 2π
    """
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


def scan_update(
    particles: ParticleArray,
    scan: LidarScanArray,
    occupancy_grid: OccupancyGridArray,
    occupancy_grid_index_scalar: float,
    lidar_std_dev: float,
) -> WeightArray:
    """
    Update particle weights based on LiDAR scan measurements.

    This function computes the likelihood of each particle given the current LiDAR scan.
    For each particle, it projects the LiDAR beams into the occupancy grid and compares
    the expected distances with the measured distances.

    Args:
        particles: Array of shape (N, 3) containing particle states [x, y, theta].
        scan: Array of shape (num_beams, 2) containing LiDAR measurements [distance, angle].
              Distances should be in the same units as the occupancy grid.
        occupancy_grid: 3D array of shape (height, width, angle_bins) containing
                       expected distances for each grid cell and angle bin.
        occupancy_grid_index_scalar: Scaling factor to convert particle coordinates
                                    to grid indices (typically 1/resolution).
        lidar_std_dev: Standard deviation of LiDAR distance measurements for the
                      Gaussian likelihood model.

    Returns:
        Array of shape (N,) containing normalized particle weights (likelihoods).
        Weights are normalized to sum to 1.0.

    Note:
        - LiDAR scan distances must use the same distance units as the occupancy grid
        - Angles in the scan are relative to the robot's heading
        - Grid indices are clipped to valid bounds to handle edge cases
    """
    N = particles.shape[0]
    num_beams = scan.shape[0]

    likelihoods = np.ones(N, dtype=np.float32)

    # Iterate over each particle and compute the likelihood based on the scan
    for i in prange(N):  # type: ignore[not-iterable]
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


@njit(parallel=True, cache=True)
def scan_update_numba(
    particles: ParticleArray,
    scan: LidarScanArray,
    occupancy_grid: OccupancyGridArray,
    occupancy_grid_index_scalar: float,
    lidar_std_dev: float,
) -> WeightArray:
    """
    Update particle weights based on LiDAR scan measurements.

    This function computes the likelihood of each particle given the current LiDAR scan.
    For each particle, it projects the LiDAR beams into the occupancy grid and compares
    the expected distances with the measured distances.

    Args:
        particles: Array of shape (N, 3) containing particle states [x, y, theta].
        scan: Array of shape (num_beams, 2) containing LiDAR measurements [distance, angle].
              Distances should be in the same units as the occupancy grid.
        occupancy_grid: 3D array of shape (height, width, angle_bins) containing
                       expected distances for each grid cell and angle bin.
        occupancy_grid_index_scalar: Scaling factor to convert particle coordinates
                                    to grid indices (typically 1/resolution).
        lidar_std_dev: Standard deviation of LiDAR distance measurements for the
                      Gaussian likelihood model.

    Returns:
        Array of shape (N,) containing normalized particle weights (likelihoods).
        Weights are normalized to sum to 1.0.

    Note:
        - LiDAR scan distances must use the same distance units as the occupancy grid
        - Angles in the scan are relative to the robot's heading
        - Grid indices are clipped to valid bounds to handle edge cases
    """
    N = particles.shape[0]
    num_beams = scan.shape[0]

    likelihoods = np.ones(N, dtype=np.float32)

    # Iterate over each particle and compute the likelihood based on the scan
    for i in prange(N):  # type: ignore[not-iterable]
        particle = particles[i]
        x, y, theta = particle[X], particle[Y], particle[THETA]

        x_idx = int(x * occupancy_grid_index_scalar)
        y_idx = int(y * occupancy_grid_index_scalar)
        x_idx = max(0, min(x_idx, occupancy_grid.shape[0] - 1))
        y_idx = max(0, min(y_idx, occupancy_grid.shape[1] - 1))

        # Compute probability of each measurement in the scan
        for j in range(num_beams):
            distance, angle = scan[j]

            angle += theta
            angle_idx = int(angle / (2 * np.pi) * occupancy_grid.shape[2])
            angle_idx = max(0, min(angle_idx, occupancy_grid.shape[2] - 1))

            expected_distance = occupancy_grid[x_idx, y_idx, angle_idx]
            diff = distance - expected_distance
            likelihood = np.exp(-0.5 * ((diff) ** 2) / (lidar_std_dev**2))

            likelihoods[i] *= likelihood

    # Normalize likelihoods
    likelihoods = likelihoods / np.sum(likelihoods)

    return likelihoods
