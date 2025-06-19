import numpy as np
import numpy.typing as npt
from numba import njit
from lidarpf.core.types import ParticleState


@njit
def compute_likelihoods(
    particles: npt.NDArray[np.float32],
    scan: npt.NDArray[np.float32],
    distance_lookup_table: npt.NDArray[np.float32],
    error_lookup_table: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Compute the likelihoods of particles given a laser scan and lookup tables.

    Args:
        particles: (N, 3) array of [x, y, theta] particles.
        scan: (M, 2) array of [distance, angle] for each beam.
        distance_lookup_table: (H, W, A) lookup table for distances to obstacles.
        distance_scalar: Scalar to convert lookup table values to expected distances.
        error_lookup_table: (E, ) array for 1D error distribution. Error is the distance from the expected distance to the measured distance.
        error_scalar: Scalar to convert lookup table values to error distances.
    Returns:
        likelihoods: (N,) array of likelihoods for each particle.
    """

    """
    On a single particle, compute the expected distance to a wall by looking up it's position in the lookup table.
    Need something that's like particle position -> look up table scalar
    Need something that's like error in measurement to expected -> LUT scalar (bin size)
    Likelihood is 1 * probablity for each measurement
    """

    if len(particles.shape) != 2 or particles.shape[1] != 3:
        raise ValueError(f"Particles must be a (N, 3) array. Received shape: {particles.shape}")
    if len(scan.shape) != 2 or scan.shape[1] != 2:
        raise ValueError(f"Scan must be a (M, 2) array. Received shape: {scan.shape}")
    if len(distance_lookup_table.shape) != 3:
        raise ValueError(
            f"Distance lookup table must be a (H, W, A) array. Received shape: {distance_lookup_table.shape}"
        )
    if len(error_lookup_table.shape) != 1:
        raise ValueError(
            f"Error lookup table must be a (E, ) array. Received shape: {error_lookup_table.shape}"
        )

    NUM_PARTICLES = particles.shape[0]

    # Get lookup table dimensions
    H, W, A = distance_lookup_table.shape
    E = error_lookup_table.shape[0]

    # Initialize likelihoods array
    likelihoods = np.ones(NUM_PARTICLES, dtype=np.float32)

    measured_distances = scan[:, 0]
    beam_angles = scan[:, 1]

    for i in range(NUM_PARTICLES):
        x, y, theta = particles[i]

        angles = np.fmod(np.fabs(theta + beam_angles), 2 * np.pi)  # Normalize angles to [0, 2π)
        angle_idxs = (angles * A / (2 * np.pi)).astype(
            np.int32
        )  # Convert angles to lookup table indices
        x_idx = int(x)
        y_idx = int(y)
        expected_distances = distance_lookup_table[x_idx, y_idx, angle_idxs]
        errors = np.fabs(measured_distances - expected_distances)

    return likelihoods
