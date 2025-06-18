from numba import njit
import numpy as np
import numpy.typing as npt


@njit
def numba_resample(weights: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
    """
    Systematic resampling algorithm based on FilterPy implementation.

    This function implements the systematic resampling algorithm originally
    from the FilterPy library by Roger Labbe, but wrapped with Numba for performance.

    Original FilterPy source: https://github.com/rlabbe/filterpy

    Args:
        weights: Array of particle weights for resampling

    Returns:
        Array of resampled particle indices
    """

    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
