import numpy as np
from lidarpf.core.types import ParticleState
from lidarpf.filter.motion_model import apply_motion_model
from lidarpf.filter.sensor_model import compute_likelihoods
from lidarpf.filter.resample import numba_resample


class ParticleFilter:
    """
    Particle Filter for 2D LiDAR-based localization.
    Maintains a set of particles, updates them using motion and sensor models,
    and resamples based on weights.
    """

    def __init__(self, occupancy_grid, lookup_table, num_particles: int):
        """
        Initialize the particle filter.
        Args:
            occupancy_grid: 2D numpy array (bool), map of the environment.
            lookup_table: Precomputed distance lookup table.
            num_particles: Number of particles to maintain.
        """
        self.occupancy_grid = occupancy_grid
        self.lookup_table = lookup_table
        self.num_particles = num_particles
        self.particles = None  # (N, 3) array of [x, y, theta]
        self.weights = None  # (N,) array
        self._initialize_particles()

    def _initialize_particles(self):
        """Randomly initialize particles in free space."""
        free = np.argwhere(self.occupancy_grid == 0)
        if free.shape[0] < self.num_particles:
            raise ValueError("Not enough free space to initialize all particles.")
        idx = np.random.choice(free.shape[0], self.num_particles, replace=False)
        poses = free[idx]
        thetas = np.random.uniform(0, 2 * np.pi, self.num_particles)
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float32)
        self.particles[:, ParticleState.X] = poses[:, 0]
        self.particles[:, ParticleState.Y] = poses[:, 1]
        self.particles[:, ParticleState.THETA] = thetas
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def predict(self, odometry_delta, noise_std):
        """Propagate particles using the motion model."""
        self.particles = apply_motion_model(self.particles, odometry_delta, noise_std)

    def update(self, scan, map_resolution, sensor_params):
        """Update particle weights using the sensor model."""
        likelihoods = compute_likelihoods(
            self.particles, scan, self.lookup_table, map_resolution, sensor_params
        )
        self.weights = likelihoods + 1e-12  # avoid zeros
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights."""
        indices = numba_resample(self.weights.astype(np.float32))
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

    def estimate(self):
        """Return the estimated state (weighted mean)."""
        mean = np.average(self.particles, axis=0, weights=self.weights)
        return mean
