"""
Type definitions for LiDAR Particle Filter.

This module contains type aliases and data structures used throughout
the particle filter implementation.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Type aliases for performance-critical arrays
ParticleArray = npt.NDArray[np.float32]  # (N, 3) array for N particles [x, y, theta]
WeightArray = npt.NDArray[np.float32]  # (N,) array for N particle weights
OccupancyGridArray = npt.NDArray[np.float32]  # (H, W, A) occupancy grid
LidarScanArray = npt.NDArray[np.float32]  # (N, 2) array for N LiDAR scans [distance, angle]
LidarErrorArray = npt.NDArray[np.float32]  # 1D array for LiDAR error probabilities
IndexArray = npt.NDArray[np.int32]  # Array of indices for resampling


@dataclass
class ParticleState:
    """Represents the state of particles in the filter."""

    particles: ParticleArray  # (N, 3) array [x, y, theta]
    weights: WeightArray  # (N,) array of weights

    def __post_init__(self) -> None:
        """Validate particle state after initialization."""
        if self.particles.shape[1] != 3:
            raise ValueError("Particles must have shape (N, 3)")
        if len(self.particles) != len(self.weights):
            raise ValueError("Number of particles must match number of weights")
        if len(self.particles) == 0:
            raise ValueError("Must have at least one particle")


@dataclass
class LidarScan:
    """Represents a LiDAR scan with distances and angles."""

    distances: npt.NDArray[np.float32]  # (N,) array of distances in meters
    angles: npt.NDArray[np.float32]  # (N,) array of angles in radians

    def __post_init__(self) -> None:
        """Validate LiDAR scan after initialization."""
        if len(self.distances) != len(self.angles):
            raise ValueError("Number of distances must match number of angles")
        if len(self.distances) == 0:
            raise ValueError("LiDAR scan cannot be empty")


@dataclass
class OccupancyGrid:
    """Represents the occupancy grid lookup table."""

    grid: OccupancyGridArray  # (H, W, A) array
    height_meters: float  # Height of grid in meters
    width_meters: float  # Width of grid in meters
    angle_bins: int  # Number of angle bins

    def __post_init__(self) -> None:
        """Validate occupancy grid after initialization."""
        if len(self.grid.shape) != 3:
            raise ValueError("Occupancy grid must be 3D (H, W, A)")
        if self.grid.shape[2] != self.angle_bins:
            raise ValueError("Third dimension must match angle_bins")
        if self.height_meters <= 0 or self.width_meters <= 0:
            raise ValueError("Grid dimensions must be positive")


@dataclass
class LidarErrorTable:
    """Represents the LiDAR error probability table."""

    error_probs: LidarErrorArray  # 1D array of error probabilities
    max_range: float  # Maximum LiDAR range in meters
    precision: float  # Precision increment in meters

    def __post_init__(self) -> None:
        """Validate error table after initialization."""
        if len(self.error_probs) == 0:
            raise ValueError("Error table cannot be empty")
        if self.max_range <= 0 or self.precision <= 0:
            raise ValueError("Max range and precision must be positive")
        expected_size = int(2 * self.max_range / self.precision)
        if len(self.error_probs) != expected_size:
            raise ValueError(
                f"Error table size should be {expected_size}, got {len(self.error_probs)}"
            )
