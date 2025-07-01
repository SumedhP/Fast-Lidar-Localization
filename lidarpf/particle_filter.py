from .types import OccupancyGridArray, OdomNoiseArray, OdomUpdateArray, ParticleArray, WeightArray, X, Y
from .numba_kernel import resample, chassis_odom_update, scan_update
import numpy as np
import time


class ParticleFilter:
    def __init__(self, num_particles: int, occupancy_grid: OccupancyGridArray, resample_period) -> None:
        self._num_particles = num_particles
        self._occupancy_grid = occupancy_grid
        self._resample_period = resample_period

        self._particles = np.zeros((num_particles, 2), dtype=np.float32)
        self._weights = np.ones(num_particles, dtype=np.float32) / num_particles
        self._previous_resample_time = time.time()
    
    def initialize_particles(self, x: float, y: float, std_dev: float) -> None:
        self._particles[:, X] = x + np.random.normal(0, std_dev, self._num_particles)
        self._particles[:, Y] = y + np.random.normal(0, std_dev, self._num_particles)

    def chassis_odometry_update(self, odometry: OdomUpdateArray, noise: OdomNoiseArray) -> None:
        chassis_odom_update(
            self._particles,
            odometry,
            noise,
            self._occupancy_grid.shape[X],
            self._occupancy_grid.shape[Y],
        )
