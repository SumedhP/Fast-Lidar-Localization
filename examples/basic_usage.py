import numpy as np
from lidarpf.filter.particle_filter import ParticleFilter


def main():
    occupancy_grid = np.zeros((20, 20), dtype=bool)
    lookup_table = np.ones((20, 20, 16), dtype=np.float64)
    pf = ParticleFilter(occupancy_grid, lookup_table, num_particles=100)
    print("Initialized ParticleFilter with 100 particles.")
    # Simulate odometry
    odom = np.array([1.0, 0.0, 0.05], dtype=np.float32)
    noise = np.array([0.02, 0.02, 0.01], dtype=np.float32)
    pf.predict(odom, noise)
    print("Predicted particle states.")
    # Simulate LiDAR scan
    scan = np.ones((10, 2), dtype=np.float32)
    map_resolution = 1.0
    sensor_params = {"sigma": 0.2, "max_range": 10.0}
    pf.update(scan, map_resolution, sensor_params)
    print("Updated particle weights.")
    pf.resample()
    print("Resampled particles.")
    est = pf.estimate()
    print(f"Estimated state: x={est[0]:.2f}, y={est[1]:.2f}, theta={est[2]:.2f}")


if __name__ == "__main__":
    main()
