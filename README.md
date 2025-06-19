# lidarpf: High-Performance LiDAR Particle Filter Localization

A fast, modular, and extensible particle filter localization system for robotics, optimized for NVIDIA Jetson and edge devices. Includes Numba-accelerated motion and sensor models, precomputed lookup tables, and robust testing.

## Features
- Efficient particle filter for 2D LiDAR-based localization
- Numba-optimized motion, sensor, and resampling steps
- Precomputed distance lookup tables (CDDT-inspired)
- Supports LD19 LiDAR and chassis odometry integration
- Designed for Jetson/edge deployment (numpy.float32 throughout)
- Uses a type-safe `ParticleState` enum for state indexing (no magic numbers)
- Comprehensive tests and benchmarks

## Installation
```bash
pip install lidarpf
# or from source
pip install -e .
```

## Basic Usage
```python
import numpy as np
from lidarpf.filter.particle_filter import ParticleFilter
from lidarpf.core.types import ParticleState

occupancy_grid = np.zeros((20, 20), dtype=bool)
lookup_table = np.ones((20, 20, 16), dtype=np.float64)
pf = ParticleFilter(occupancy_grid, lookup_table, num_particles=100)

# Predict step
odom = np.array([1.0, 0.0, 0.05], dtype=np.float32)
noise = np.array([0.02, 0.02, 0.01], dtype=np.float32)
pf.predict(odom, noise)

# Update step
scan = np.ones((10, 2), dtype=np.float32)
map_resolution = 1.0
sensor_params = {'sigma': 0.2, 'max_range': 10.0}
pf.update(scan, map_resolution, sensor_params)

# Resample and estimate
pf.resample()
est = pf.estimate()
print(f"Estimated state: x={est[ParticleState.X]}, y={est[ParticleState.Y]}, theta={est[ParticleState.THETA]}")
```

## API Overview
- `ParticleFilter`: Main class for localization
- `apply_motion_model`: Numba-optimized motion update
- `compute_likelihoods`: Numba-optimized sensor model
- `numba_resample`: Numba-optimized systematic resampling
- `compute_lookup_table`: Precompute distance tables for fast likelihoods
- `ParticleState`: Enum for state indexing (X, Y, THETA)

## Benchmarks
Run the provided benchmark script:
```bash
python examples/benchmark.py
```

## Testing
Run all tests with pytest:
```bash
pytest tests/
```

## License
MIT
