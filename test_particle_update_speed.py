from lidarpf.particle_update import chassis_odom_update, chassis_odom_update_numba, scan_update, scan_update_numba, scan_update_numba_vectorized

import numpy as np
from timeit import timeit


def test_chassis_odom_update(func, print_results=True):
    PARTICLE_SIZES = [1_000, 10_000, 20_000, 100_000]

    for size in PARTICLE_SIZES:
        particles = np.random.rand(size, 3).astype(np.float32) * 100
        odometry = np.array([1.0, 0.5, 0.1], dtype=np.float32)
        noise = np.array([0.1, 0.1, 0.01], dtype=np.float32)
        max_height = 100.0
        max_width = 100.0
        ITERATIONS = 100
        time_taken = timeit(lambda: func(particles, odometry, noise, max_height, max_width), number=ITERATIONS) / ITERATIONS
        if print_results:
            print(f"{func.__name__} with {size} particles took {time_taken:.6f} seconds")


def test_scan_update(func, print_results=True):
    PARTICLE_SIZES = [1_000, 10_000, 20_000, 100_000]

    for size in PARTICLE_SIZES:
        particles = np.random.rand(size, 3).astype(np.float32) * 100
        scan = np.random.rand(100, 2).astype(np.float32) * 10
        occupancy_grid = np.random.rand(1200, 800, 120).astype(np.float32)
        ITERATIONS = 50
        time_taken = timeit(lambda: func(particles, scan, occupancy_grid, 100, 5), number=ITERATIONS) / ITERATIONS
        if print_results:
            print(f"{func.__name__} with {size} particles took {time_taken:.6f} seconds")

if __name__ == "__main__":
    # print("Testing chassis odom update (regular)...")
    # test_chassis_odom_update(chassis_odom_update)

    # print("\nTesting chassis odom update (Numba)...")
    # test_chassis_odom_update(chassis_odom_update_numba, print_results=False)
    # test_chassis_odom_update(chassis_odom_update_numba)

    # print("\nTesting scan update (regular)...")
    # # test_scan_update(scan_update)

    # print("\nTesting scan update (Numba)...")
    # test_scan_update(scan_update_numba, print_results=False)
    # test_scan_update(scan_update_numba)

    print("\nTesting scan update (Numba vectorized)...")
    test_scan_update(scan_update_numba_vectorized, print_results=False)
    test_scan_update(scan_update_numba_vectorized)
