import numpy as np
from timeit import timeit
from lidarpf.filter.resample import numba_resample
from lidarpf.filter.motion_model import apply_motion_model
from lidarpf.filter.sensor_model import compute_likelihoods


def benchmark_resample():
    print("Benchmarking numba_resample...")
    for n in [100, 1000, 5000, 10000, 50000]:
        weights = np.random.rand(n).astype(np.float32)
        weights /= np.sum(weights)
        t = timeit(lambda: numba_resample(weights), number=10) / 10
        print(f"  {n} particles: {t:.6f} s")


def benchmark_motion_model():
    print("Benchmarking apply_motion_model...")
    for n in [100, 1000, 5000, 10000, 50000]:
        particles = np.zeros((n, 3), dtype=np.float32)
        delta = np.array([1.0, 0.0, 0.1], dtype=np.float32)
        noise = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        t = timeit(lambda: apply_motion_model(particles, delta, noise), number=10) / 10
        print(f"  {n} particles: {t:.6f} s")


def benchmark_sensor_model():
    print("Benchmarking compute_likelihoods...")
    for n in [100, 1000, 5000]:
        particles = np.zeros((n, 3), dtype=np.float32)
        scan = np.ones((10, 2), dtype=np.float32)
        lookup_table = np.ones((20, 20, 16), dtype=np.float64)
        map_resolution = 1.0
        sensor_params = {"sigma": 0.2, "max_range": 10.0}
        t = (
            timeit(
                lambda: compute_likelihoods(
                    particles, scan, lookup_table, map_resolution, sensor_params
                ),
                number=5,
            )
            / 5
        )
        print(f"  {n} particles: {t:.6f} s")


def main():
    benchmark_resample()
    benchmark_motion_model()
    benchmark_sensor_model()


if __name__ == "__main__":
    main()
