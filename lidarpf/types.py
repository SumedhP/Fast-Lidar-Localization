import numpy as np
import numpy.typing as npt

# Type aliases for performance-critical arrays
ParticleArray = npt.NDArray[np.float32]  # (N, 3) array for N particles [x, y, theta]
WeightArray = npt.NDArray[np.float32]  # (N,) array for N particle weights
OccupancyGridArray = npt.NDArray[np.float32]  # (H, W, A) occupancy grid
LidarScanArray = npt.NDArray[np.float32]  # (N, 2) array for N LiDAR scans [distance, angle]
LidarErrorArray = npt.NDArray[np.float32]  # 1D array for LiDAR error probabilities
OdomUpdateArray = npt.NDArray[np.float32]  # (N, 3) array for odometry updates [dx, dy, dtheta]
OdomNoiseArray = npt.NDArray[np.float32]  # (3,) array for odometry noise [std_x, std_y, std_theta]

X, Y, THETA = 0, 1, 2  # Indices for particle state components
