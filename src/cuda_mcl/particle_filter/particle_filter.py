from typing import Tuple
import cupy as cp
import numpy as np
from .particle_filter_config import ParticleFilterConfig


class ParticleFilter:
    def __init__(self, config: ParticleFilterConfig):
        self._config = config
        self._particles = cp.zeros((config.num_particles, 2), dtype=cp.float32)
        self._weights = cp.ones(config.num_particles, dtype=cp.float32) / config.num_particles
        self._occupancy_grid = cp.asarray(config.occupancy_grid, dtype=cp.float32)

        # For resampling
        self._cumulative_weights = cp.zeros(config.num_particles, dtype=cp.float32)
        self._positions = cp.zeros(config.num_particles, dtype=cp.float32)

        self._init_kernels()

    def initialize_particles(self, initial_x: float, initial_y: float, position_std_dev: float) -> None:
        """
        Initialize particles around a given position with noise.

        Args:
            initial_x (float): Initial x position of the robot.
            initial_y (float): Initial y position of the robot.
            position_std_dev (float): Standard deviation for the initial particle positions.
        """
        noise_x = cp.random.normal(0, position_std_dev, size=self._particles.shape[0])
        noise_y = cp.random.normal(0, position_std_dev, size=self._particles.shape[0])

        self._particles[:, 0] = initial_x + noise_x
        self._particles[:, 1] = initial_y + noise_y

    def update_chassis_measurement(self, odom_x: float, odom_y: float) -> None:
        """
        Update particle positions based on chassis measurements.

        Args:
            odom_x (float): Chassis x displacement.
            odom_y (float): Chassis y displacement.
        """
        noise_x = cp.random.normal(0, self._config.chassis_noise_std[0], size=self._particles.shape[0])
        noise_y = cp.random.normal(0, self._config.chassis_noise_std[1], size=self._particles.shape[0])

        self._particles[:, 0], self._particles[:, 1] = self._chassis_update_kernel(
            self._particles[:, 0],
            self._particles[:, 1],
            noise_x,
            noise_y,
            odom_x,
            odom_y,
            self._occupancy_grid.shape[0] / self._config.occupancy_grid_index_scalar,
            self._occupancy_grid.shape[1] / self._config.occupancy_grid_index_scalar,
        )

    def update_lidar_measurement(self, scan: np.ndarray) -> None:
        """
        Update particle weights based on LiDAR measurements.

        Args:
            scan (np.ndarray): LiDAR scan data, expected to be a 2D array with shape (num_beams, 2),
                               where each row contains [distance, angle].
        """
        H, W, A = self._occupancy_grid.shape
        num_beams = scan.shape[0]
        d_scan = cp.asarray(scan, dtype=cp.float32)

        # Precompute some constants
        angle_bin_scalar = np.float32(A / (2 * np.pi))
        inverse_lidar_variance = np.float32(1.0 / (self._config.lidar_std_dev**2))

        threads_per_block = 256
        blocks = (self._config.num_particles + threads_per_block - 1) // threads_per_block

        self._scan_update_kernel(
            (blocks,),
            (threads_per_block,),
            self._particles,
            self._weights,
            d_scan,
            self._occupancy_grid,
            self._config.occupancy_grid_index_scalar,
            inverse_lidar_variance,
            angle_bin_scalar,
            self._config.num_particles,
            num_beams,
            H,
            W,
            A,
        )

    def get_pose(self) -> Tuple[float, float]:
        """
        Get the estimated pose of the robot.

        Returns:
            Tuple[float, float]: Estimated (x, y) position of the robot.
        """
        X = cp.average(self._particles[:, 0], weights=self._weights)
        Y = cp.average(self._particles[:, 1], weights=self._weights)
        return (X, Y)

    def resample_particles(self) -> None:
        """
        Resample particles based on their weights.
        """
        cp.cumsum(self._weights, out=self._cumulative_weights)
        self._cumulative_weights /= self._cumulative_weights[-1]

        u = cp.random.rand() / self._config.num_particles
        self._positions[:] = u + cp.arange(self._config.num_particles) / self._config.num_particles

        indices = cp.searchsorted(self._cumulative_weights, self._positions)

        self._particles[:] = self._particles[indices]
        self._weights.fill(1.0 / self._config.num_particles)

    def _init_kernels(self):
        self._chassis_update_kernel = cp.ElementwiseKernel(
            # Input arguments
            "float32 x_old, float32 y_old, "
            "float32 n_x, float32 n_y, "
            "float32 odom_x, float32 odom_y, "
            "float32 max_h, float32 max_w",
            # Output parameters
            "float32 x_new, float32 y_new",
            # Kernel code
            """
                float xu = x_old + odom_x + n_x;
                float yu = y_old + odom_y + n_y;
                
                // clip
                x_new = min(max(xu, 0.0f), max_h);
                y_new = min(max(yu, 0.0f), max_w);
            """,
            name="chassis_update_kernel",
        )

        self._scan_update_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void scan_update_kernel(
            const float* particles,
            float* weights,
            const float* scan,
            const float* occupancy_grid,
            const float occupancy_grid_index_scalar,
            const float inverse_lidar_variance,
            const float angle_bin_scalar,
            const int N,
            const int num_beams,
            const int H,
            const int W,
            const int A
        ) {
            // Get thread index
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (i >= N) return;
                                        
            // Compute grid strides for indexing
            const int grid_stride_y = W * A;  // stride for moving in x direction
            const int grid_stride_z = A;      // stride for moving in y direction
            
            const float x = particles[i * 2];
            const float y = particles[i * 2 + 1];
            
            // Convert to grid indices with bounds checking
            int x_idx = int(x * occupancy_grid_index_scalar);
            int y_idx = int(y * occupancy_grid_index_scalar);
                                                        
            if (x_idx < 0) x_idx = 0;
            if (x_idx >= H) x_idx = H - 1;
            if (y_idx < 0) y_idx = 0;
            if (y_idx >= W) y_idx = W - 1;
            
            // Precompute base grid offset for this particle
            const int base_grid_offset = x_idx * grid_stride_y + y_idx * grid_stride_z;
            
            // Accumulate log likelihood
            float log_likelihood = 0.0f;
            
            // Process beams
            #pragma unroll
            for (int j = 0; j < num_beams; j++) {
                const float distance = scan[j * 2];
                const float angle = scan[j * 2 + 1];
                
                // Convert angle to grid index with fast rounding
                int angle_idx = int(angle * angle_bin_scalar);
                if (angle_idx < 0) angle_idx = 0;
                if (angle_idx >= A) angle_idx = A - 1;
                
                // Get expected distance with optimized indexing
                const float expected_distance = occupancy_grid[base_grid_offset + angle_idx];
                
                // Compute squared difference and accumulate log likelihood
                const float diff = distance - expected_distance;
                log_likelihood += -0.5f * inverse_lidar_variance * diff * diff;
            }
            
            // Convert back to linear space and store
            weights[i] = expf(log_likelihood);
        }
        """,
            name="scan_update_kernel",
        )
