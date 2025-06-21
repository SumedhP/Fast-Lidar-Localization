import numpy as np
import numpy.typing as npt
from numba import njit, prange

from lidarpf.validation import (
    validate_lidar_scan,
    validate_occupancy_grid,
    validate_odometry_noise,
    validate_odometry_update,
    validate_particle_array,
    validate_weight_array,
)

from .types import (
    ParticleArray,
    WeightArray,
    OccupancyGridArray,
    LidarScanArray,
    OdomUpdateArray,
    OdomNoiseArray,
    X,
    Y,
)


@njit(parallel=True, cache=True)
def chassis_odom_update_compiled(
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
        particles: Array of shape (N, 2) containing particle states [x, y].
                  Modified in-place.
        odometry: Array of shape (2,) containing odometry deltas [dx, dy].
        noise: Array of shape (2,) containing standard deviations for Gaussian noise
               [std_x, std_y].
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

    x_std = noise[X]
    y_std = noise[Y]

    particles[:, X] += x_delta + np.random.normal(0, x_std, N)
    particles[:, Y] += y_delta + np.random.normal(0, y_std, N)

    np.clip(particles[:, X], 0, max_height, out=particles[:, X])
    np.clip(particles[:, Y], 0, max_width, out=particles[:, Y])


@njit(parallel=True, cache=True)
def scan_update_compiled(
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
        particles: Array of shape (N, 2) containing particle states [x, y].
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
    # Note for a developer: Vectorizing this just made it slower with Numba.
    N = particles.shape[0]
    num_beams = scan.shape[0]

    likelihoods = np.ones(N, dtype=np.float32)

    # Iterate over each particle and compute the likelihood based on the scan
    for i in prange(N):  # type: ignore[not-iterable]
        particle = particles[i]
        x, y = particle[X], particle[Y]

        x_idx = int(x * occupancy_grid_index_scalar)
        y_idx = int(y * occupancy_grid_index_scalar)
        x_idx = max(0, min(x_idx, occupancy_grid.shape[0] - 1))
        y_idx = max(0, min(y_idx, occupancy_grid.shape[1] - 1))

        # Compute probability of each measurement in the scan
        for j in range(num_beams):
            distance, angle = scan[j]

            angle_idx = int(angle / (2 * np.pi) * occupancy_grid.shape[2])
            angle_idx = max(0, min(angle_idx, occupancy_grid.shape[2] - 1))

            expected_distance = occupancy_grid[x_idx, y_idx, angle_idx]
            diff = distance - expected_distance
            likelihood = np.exp(-0.5 * (diff**2) / (lidar_std_dev**2))

            likelihoods[i] *= likelihood

    # Normalize likelihoods
    likelihoods = likelihoods / (np.sum(likelihoods) + 1e-10)

    return likelihoods


@njit(parallel=True, cache=True)
def resample_compiled(weights: WeightArray) -> npt.NDArray[np.int32]:
    """
    Systematic resampling algorithm based on FilterPy implementation.

    This function implements the systematic resampling algorithm originally
    from the FilterPy library by Roger Labbe, but wrapped with Numba for performance.

    Original FilterPy source: https://github.com/rlabbe/filterpy

    Args:
        weights: Array of particle weights for resampling

    Returns:
        Array of resampled particle indices
    """
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N
    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def chassis_odom_update(
    particles: ParticleArray,
    odometry: OdomUpdateArray,
    noise: OdomNoiseArray,
    max_height: float,
    max_width: float,
) -> None:
    """
    Update particle positions using odometry measurements with noise.
    Read the docstring of `chassis_odom_update_compiled` for details.
    """
    validate_particle_array(particles)
    validate_odometry_update(odometry)
    validate_odometry_noise(noise)
    chassis_odom_update_compiled(particles, odometry, noise, max_height, max_width)


def scan_update(
    particles: ParticleArray,
    scan: LidarScanArray,
    occupancy_grid: OccupancyGridArray,
    occupancy_grid_index_scalar: float,
    lidar_std_dev: float,
) -> WeightArray:
    """
    Update particle weights based on LiDAR scan measurements.
    Read the docstring of `scan_update_compiled` for details.
    """
    validate_particle_array(particles)
    validate_lidar_scan(scan)
    validate_occupancy_grid(occupancy_grid)
    return scan_update_compiled(particles, scan, occupancy_grid, occupancy_grid_index_scalar, lidar_std_dev)


def resample(weights: WeightArray) -> npt.NDArray[np.int32]:
    """
    Resample particles based on their weights using systematic resampling.
    Read the docstring of `resample_compiled` for details.
    """
    validate_weight_array(weights)
    return resample_compiled(weights)
