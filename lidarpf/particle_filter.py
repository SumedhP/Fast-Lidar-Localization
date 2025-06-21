"""
Main Particle Filter implementation for LiDAR localization.

This module contains the ParticleFilter class that orchestrates the motion model,
sensor model, and resampling to provide a complete particle filter implementation
for 2D LiDAR-based robot localization.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from .types import LidarErrorArray, OccupancyGridArray, ParticleArray, WeightArray


@njit
def _update_particles_motion(
    particles: ParticleArray,
    delta_x: float,
    delta_y: float,
    delta_theta: float,
    x_std: float,
    y_std: float,
    theta_std: float,
    width: float,
    height: float,
) -> ParticleArray:
    N = len(particles)
    updated_particles = particles.copy()
    x_noise = np.random.normal(0.0, x_std, N).astype(np.float32)
    y_noise = np.random.normal(0.0, y_std, N).astype(np.float32)
    theta_noise = np.random.normal(0.0, theta_std, N).astype(np.float32)
    for i in range(N):
        new_x = particles[i, 0] + delta_x + x_noise[i]
        new_y = particles[i, 1] + delta_y + y_noise[i]
        new_theta = particles[i, 2] + delta_theta + theta_noise[i]
        new_x = max(0.0, min(new_x, width - 1e-6))
        new_y = max(0.0, min(new_y, height - 1e-6))
        new_theta = new_theta % (2.0 * np.pi)
        updated_particles[i, 0] = new_x
        updated_particles[i, 1] = new_y
        updated_particles[i, 2] = new_theta
    return updated_particles


@njit
def _compute_particle_likelihoods(
    particles: ParticleArray,
    distances: npt.NDArray[np.float32],
    angles: npt.NDArray[np.float32],
    occupancy_grid: OccupancyGridArray,
    error_table: LidarErrorArray,
    lut_scalar: float,
    error_scalar: float,
    max_range: float,
) -> WeightArray:
    N = len(particles)
    num_scans = len(distances)
    likelihoods = np.ones(N, dtype=np.float32)
    grid_height, grid_width, num_angle_bins = occupancy_grid.shape
    angle_bin_size = 2.0 * np.pi / num_angle_bins
    for i in range(N):
        particle_likelihood = 1.0
        for j in range(num_scans):
            px, py, ptheta = particles[i]
            expected_angle = angles[j] + ptheta
            expected_angle = expected_angle % (2.0 * np.pi)
            grid_x = int(px * lut_scalar)
            grid_y = int(py * lut_scalar)
            grid_x = max(0, min(grid_x, grid_height - 1))
            grid_y = max(0, min(grid_y, grid_width - 1))
            angle_bin = int(expected_angle / angle_bin_size)
            angle_bin = max(0, min(angle_bin, num_angle_bins - 1))
            expected_distance = occupancy_grid[grid_x, grid_y, angle_bin]
            expected_distance = min(expected_distance, max_range)
            error = abs(expected_distance - distances[j])
            error_index = int(error * error_scalar)
            error_index = max(0, min(error_index, len(error_table) - 1))
            scan_likelihood = error_table[error_index]
            particle_likelihood *= scan_likelihood
        likelihoods[i] = particle_likelihood
    return likelihoods


class ParticleFilter:
    """
    High-performance LiDAR particle filter for robot localization.

    This class implements a complete particle filter for 2D LiDAR-based
    robot localization using Numba-optimized components for performance.

    Attributes:
        num_particles: Number of particles in the filter
        particles: Current particle states (N, 3) array [x, y, theta]
        weights: Current particle weights (N,) array
        occupancy_grid: Occupancy grid lookup table (H, W, A)
        error_table: LiDAR error probability table
        lut_scalar: Scalar to convert particle positions to grid coordinates
        error_scalar: Scalar to convert distance errors to error table indices
        max_range: Maximum LiDAR range in meters
        width: Map width in meters
        height: Map height in meters
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGridArray,
        error_table: LidarErrorArray,
        lut_scalar: float,
        error_scalar: float,
        max_range: float,
        num_particles: int,
        width: float,
        height: float,
    ) -> None:
        """
        Initialize the particle filter.

        Args:
            occupancy_grid: (H, W, A) occupancy grid lookup table
            error_table: Array of LiDAR error probabilities
            lut_scalar: Scalar to convert particle positions to grid coordinates
            error_scalar: Scalar to convert distance errors to error table indices
            max_range: Maximum LiDAR range in meters
            num_particles: Number of particles to use
            width: Map width in meters
            height: Map height in meters
        """
        self.num_particles = num_particles
        self.occupancy_grid = occupancy_grid
        self.error_table = error_table
        self.lut_scalar = lut_scalar
        self.error_scalar = error_scalar
        self.max_range = max_range
        self.width = width
        self.height = height
        self.particles: Optional[ParticleArray] = None
        self.weights: Optional[WeightArray] = None

    def initialize(
        self,
        start_x: float,
        start_y: float,
        start_theta: float,
        position_std: float,
        angle_std: float,
    ) -> None:
        """
        Initialize particles with uniform distribution around starting pose.

        Args:
            start_x: Starting x position in meters
            start_y: Starting y position in meters
            start_theta: Starting theta angle in radians
            position_std: Standard deviation for position initialization (meters)
            angle_std: Standard deviation for angle initialization (radians)
        """
        start_theta = start_theta % (2.0 * np.pi)
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float32)
        self.particles[:, 0] = np.random.normal(start_x, position_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(start_y, position_std, self.num_particles)
        self.particles[:, 2] = np.random.normal(start_theta, angle_std, self.num_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.width - 1e-6)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.height - 1e-6)
        self.particles[:, 2] = self.particles[:, 2] % (2.0 * np.pi)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float32)

    def odometry_update(
        self,
        delta_x: float,
        delta_y: float,
        delta_theta: float,
        x_std: float,
        y_std: float,
        theta_std: float,
    ) -> None:
        """
        Update particles based on odometry with noise.

        Args:
            delta_x: Change in x position in world frame (meters)
            delta_y: Change in y position in world frame (meters)
            delta_theta: Change in theta angle in world frame (radians)
            x_std: Standard deviation for x position noise (meters)
            y_std: Standard deviation for y position noise (meters)
            theta_std: Standard deviation for theta angle noise (radians)
        """
        if self.particles is None or self.weights is None:
            raise ValueError("Particles must be initialized before odometry update")
        self.particles = _update_particles_motion(
            self.particles,
            delta_x,
            delta_y,
            delta_theta,
            x_std,
            y_std,
            theta_std,
            self.width,
            self.height,
        )

    def lidar_update(self, distances: npt.NDArray[np.float32], angles: npt.NDArray[np.float32]) -> None:
        """
        Update particle weights based on LiDAR measurements.

        Args:
            distances: Array of measured distances in meters
            angles: Array of measured angles in radians
        """
        if self.particles is None or self.weights is None:
            raise ValueError("Particles must be initialized before LiDAR update")
        likelihoods = _compute_particle_likelihoods(
            self.particles,
            distances,
            angles,
            self.occupancy_grid,
            self.error_table,
            self.lut_scalar,
            self.error_scalar,
            self.max_range,
        )
        self.weights = self.weights * likelihoods
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
        else:
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float32)

    def resample_particles(self) -> None:
        """
        Resample particles based on their weights using systematic resampling.
        """
        if self.particles is None or self.weights is None:
            raise ValueError("Particles must be initialized before resampling")
        indices = resample(self.weights)
        self.particles = self.particles[indices].copy()
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float32)

    def get_expected_pose(self) -> Tuple[float, float, float]:
        """
        Get the expected robot pose as weighted mean of particles.

        Returns:
            Tuple of (x, y, theta) representing expected pose
        """
        if self.particles is None or self.weights is None:
            raise ValueError("Particles must be initialized before getting expected pose")
        expected_x = np.sum(self.particles[:, 0] * self.weights)
        expected_y = np.sum(self.particles[:, 1] * self.weights)
        cos_theta = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        sin_theta = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        expected_theta = np.arctan2(sin_theta, cos_theta)
        if expected_theta < 0:
            expected_theta += 2.0 * np.pi
        return expected_x, expected_y, expected_theta

    def get_particle_state(self) -> Tuple[ParticleArray, WeightArray]:
        """
        Get current particle states and weights.

        Returns:
            Tuple of (particles, weights)
        """
        if self.particles is None or self.weights is None:
            raise ValueError("Particles are not initialized")
        return self.particles.copy(), self.weights.copy()
