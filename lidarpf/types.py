import numpy as np
import numpy.typing as npt

ParticleArray = npt.NDArray[np.float32]  # (N, 2) array for N particles [x, y]
WeightArray = npt.NDArray[np.float32]  # (N,) array for N particle weights
OccupancyGridArray = npt.NDArray[np.float32]  # (H, W, A) occupancy grid
LidarScanArray = npt.NDArray[np.float32]  # (N, 2) array for N LiDAR scans [distance, angle]
OdomUpdateArray = npt.NDArray[np.float32]  # (N, 2) array for odometry updates [dx, dy]
OdomNoiseArray = npt.NDArray[np.float32]  # (2,) array for odometry noise [std_x, std_y]

X, Y = 0, 1  # Indices for particle state components
