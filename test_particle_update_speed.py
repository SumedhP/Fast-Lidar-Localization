from lidarpf.numba_kernel import chassis_odom_update, scan_update, resample

import numpy as np
from timeit import timeit

PARTICLE_SIZES = [100, 100, 1_000, 10_000, 20_000, 100_000]


def test_chassis_odom_update(func, print_results=True):
    for size in PARTICLE_SIZES:
        particles = np.random.rand(size, 3).astype(np.float32) * 100
        odometry = np.array([1.0, 0.5, 0.1], dtype=np.float32)
        noise = np.array([0.1, 0.1, 0.01], dtype=np.float32)
        max_height = 100.0
        max_width = 100.0
        ITERATIONS = 1000
        time_taken = (
            timeit(lambda: func(particles, odometry, noise, max_height, max_width), number=ITERATIONS) / ITERATIONS
        ) * 1000
        if print_results:
            print(f"{func.__name__} with {size} particles took {time_taken:.6f} milliseconds")


def test_scan_update(func, print_results=True):
    for size in PARTICLE_SIZES:
        particles = np.random.rand(size, 3).astype(np.float32) * 100
        scan = np.random.rand(12, 2).astype(np.float32) * 10
        occupancy_grid = np.random.rand(1200, 800, 120).astype(np.float32)
        ITERATIONS = 1000
        time_taken = timeit(lambda: func(particles, scan, occupancy_grid, 100, 5), number=ITERATIONS) / ITERATIONS * 1000
        if print_results:
            print(f"{func.__name__} with {size} particles took {time_taken:.6f} milliseconds")


def test_resample(func, print_results=True):
    for size in PARTICLE_SIZES:
        weights = np.random.rand(size).astype(np.float32)
        ITERATIONS = 1000
        time_taken = timeit(lambda: func(weights), number=ITERATIONS) / ITERATIONS * 1000
        if print_results:
            print(f"{func.__name__} with {size} particles took {time_taken:.6f} milliseconds")


if __name__ == "__main__":
    print("Testing chassis odom update...")
    test_chassis_odom_update(chassis_odom_update, print_results=False)
    test_chassis_odom_update(chassis_odom_update)

    print("\nTesting LiDAR scan update...")
    test_scan_update(scan_update, print_results=False)
    test_scan_update(scan_update)

    print("\nTesting resampling...")
    test_resample(resample, print_results=False)
    test_resample(resample)
