from .types import (
    ParticleArray,
    WeightArray,
    OccupancyGridArray,
    LidarScanArray,
    OdomUpdateArray,
    OdomNoiseArray,
)
import numpy as np


def validate_particle_array(particles: ParticleArray) -> None:
    """
    Validate the particle array for correct shape and data type.

    Args:
        particles: Particle array to validate.

    Raises:
        ValueError: If the particle array is not a 2D array or does not have float32 type.
    """
    if not isinstance(particles, np.ndarray):
        raise ValueError("Particles must be a numpy array of type float32.")
    if particles.dtype != np.float32:
        raise ValueError("Particles must have dtype float32.")
    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError(f"Particles must be a 2D array with shape (N, 2). Input shape: {particles.shape}")


def validate_weight_array(weights: WeightArray) -> None:
    """
    Validate the weight array for correct shape and data type.

    Args:
        weights: Weight array to validate.

    Raises:
        ValueError: If the weight array is not a 1D array or does not have float32 type.
    """
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a numpy array of type float32.")
    if weights.dtype != np.float32:
        raise ValueError("Weights must have dtype float32.")
    if weights.ndim != 1:
        raise ValueError(f"Weights must be a 1D array. Input shape: {weights.shape}")


def validate_occupancy_grid(grid: OccupancyGridArray) -> None:
    """
    Validate the occupancy grid for correct shape and data type.

    Args:
        grid: Occupancy grid to validate.

    Raises:
        ValueError: If the occupancy grid is not a 3D array or does not have float32 type.
    """
    if not isinstance(grid, np.ndarray):
        raise ValueError("Occupancy grid must be a numpy array of type float32.")
    if grid.dtype != np.float32:
        raise ValueError("Occupancy grid must have dtype float32.")
    if grid.ndim != 3:
        raise ValueError(f"Occupancy grid must be a 3D array with shape (H, W, A). Input shape: {grid.shape}")


def validate_lidar_scan(scan: LidarScanArray) -> None:
    """
    Validate the LiDAR scan for correct shape and data type.

    Args:
        scan: LiDAR scan to validate.

    Raises:
        ValueError: If the LiDAR scan is not a 2D array or does not have float32 type.
    """
    if not isinstance(scan, np.ndarray):
        raise ValueError("LiDAR scan must be a numpy array of type float32.")
    if scan.dtype != np.float32:
        raise ValueError("LiDAR scan must have dtype float32.")
    if scan.ndim != 2 or scan.shape[1] != 2:
        raise ValueError(f"LiDAR scan must be a 2D array with shape (N, 2). Input shape: {scan.shape}")


def validate_odometry_update(update: OdomUpdateArray) -> None:
    """
    Validate the odometry update for correct shape and data type.

    Args:
        update: Odometry update to validate.

    Raises:
        ValueError: If the odometry update is not a 1D array or does not have float32 type.
    """
    if not isinstance(update, np.ndarray):
        raise ValueError("Odometry update must be a numpy array of type float32.")
    if update.dtype != np.float32:
        raise ValueError("Odometry update must have dtype float32.")
    if update.ndim != 1 or update.shape[0] != 2:
        raise ValueError(f"Odometry update must be a 1D array with shape (2,). Input shape: {update.shape}")


def validate_odometry_noise(noise: OdomNoiseArray) -> None:
    """
    Validate the odometry noise for correct shape and data type.

    Args:
        noise: Odometry noise to validate.

    Raises:
        ValueError: If the odometry noise is not a 1D array or does not have float32 type.
    """
    if not isinstance(noise, np.ndarray):
        raise ValueError("Odometry noise must be a numpy array of type float32.")
    if noise.dtype != np.float32:
        raise ValueError("Odometry noise must have dtype float32.")
    if noise.ndim != 1 or noise.shape[0] != 2:
        raise ValueError(f"Odometry noise must be a 1D array with shape (2,). Input shape: {noise.shape}")
