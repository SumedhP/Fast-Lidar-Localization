# from typing import Optional, Tuple

# import time
# import numpy as np
# import numpy.typing as npt
# from numba import njit

# from .types import LidarErrorArray, OccupancyGridArray, ParticleArray, WeightArray
# from .numba_kernel import resample, chassis_odom_update, scan_update


# class ParticleFilter:
#     def __init__(self, num_particles: int, occupancy_grid: OccupancyGridArray, resample_period) -> None:
#         self._particles = np.zeros((num_particles, 3), dtype=np.float32)
#         self._previous_resample_time = time.time()

