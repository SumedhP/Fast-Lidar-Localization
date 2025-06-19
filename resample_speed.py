import numpy as np
from timeit import timeit
from lidarpf.filter.resample import numba_resample


def test_speeds(func) -> None:
    num_particles = [100, 1_000, 5_000, 10_000, 20_000, 1_000_000]
    for n in num_particles:
        weights = np.random.rand(n)
        weights /= np.sum(weights)  # Normalize weights

        ITERATIONS = 10
        time_taken = timeit(lambda: func(weights), number=ITERATIONS) / ITERATIONS
        print(f"{func.__name__} with {n} particles took {time_taken:.6f} seconds")


if __name__ == "__main__":
    print("\nTesting Resampling speed...")
    # Warmup call
    numba_resample(np.random.rand(1_000).astype(np.float32))
    test_speeds(numba_resample)
