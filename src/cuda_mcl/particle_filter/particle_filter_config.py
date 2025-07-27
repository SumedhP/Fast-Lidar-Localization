from typing import Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ParticleFilterConfig:
    num_particles: int  # Number of particles in the filter
    chassis_noise_std: Tuple[float, float]  # Standard deviation of noise in chassis updates (x, y)
    lidar_std_dev: float  # Standard deviation of the LiDAR measurements
    occupancy_grid: np.ndarray  # Occupancy grid for the environment
    occupancy_grid_index_scalar: float  # Scalar for converting coordinates to grid indices

    # Do validation of the configuration parameters after creation
    def __post_init__(self):
        if self.num_particles <= 0:
            raise ValueError("Number of particles must be greater than zero.")
        if self.chassis_noise_std[0] < 0 or self.chassis_noise_std[1] < 0:
            raise ValueError("Chassis noise standard deviations must be non-negative.")
        if self.lidar_std_dev <= 0:
            raise ValueError("LiDAR standard deviation must be greater than zero.")
        if not isinstance(self.occupancy_grid, np.ndarray):
            raise TypeError("Occupancy grid must be a numpy array.")
        if self.occupancy_grid.ndim != 3:
            raise ValueError("Occupancy grid must be a 3D numpy array consisting of height, width, and angle bins.")
        if self.occupancy_grid_index_scalar <= 0:
            raise ValueError("Occupancy grid index scalar must be greater than zero.")
