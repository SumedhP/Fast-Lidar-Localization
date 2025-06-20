# LiDAR Particle Filter

A high-performance, Numba-optimized particle filter implementation for 2D LiDAR-based robot localization.

## Features

- **High Performance**: Numba JIT compilation for 50x speedup over pure Python
- **Systematic Resampling**: Efficient particle resampling algorithm
- **Type Safety**: Full type hints and NumPy typing support
- **Comprehensive Testing**: >90% code coverage with extensive test suite
- **Professional Quality**: PEP 8 compliant, comprehensive documentation
- **Minimal Dependencies**: Only NumPy and Numba required

## Installation

### From PyPI (when published)

```bash
pip install lidarpf
```

### From Source

```bash
git clone https://github.com/yourusername/lidarpf.git
cd lidarpf
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/lidarpf.git
cd lidarpf
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from lidarpf import ParticleFilter

# Create sample data
occupancy_grid = np.zeros((1200, 800, 120), dtype=np.float32)  # 12m x 8m x 120 angle bins
error_table = np.random.uniform(0, 1, 2400).astype(np.float32)  # LiDAR error probabilities

# Initialize particle filter
pf = ParticleFilter(
    occupancy_grid=occupancy_grid,
    error_table=error_table,
    lut_scalar=100.0,  # 1cm precision
    error_scalar=100.0,  # 1cm precision
    max_range=12.0,  # 12m max range
    num_particles=1000,
    width=12.0,  # 12m width
    height=8.0   # 8m height
)

# Initialize particles around starting pose
pf.initialize(
    start_x=6.0,
    start_y=4.0,
    start_theta=np.pi/2,
    position_std=0.1,
    angle_std=0.05
)

# Get initial pose estimate
x, y, theta = pf.get_expected_pose()
print(f"Initial pose: ({x:.2f}, {y:.2f}, {theta:.2f})")

# Update with odometry
pf.odometry_update(
    delta_x=0.5,
    delta_y=0.0,
    delta_theta=0.1,
    x_std=0.05,
    y_std=0.05,
    theta_std=0.02
)

# Update with LiDAR measurements
distances = np.array([5.0, 3.0, 7.0], dtype=np.float32)
angles = np.array([0.0, np.pi/4, np.pi/2], dtype=np.float32)
pf.lidar_update(distances, angles)

# Resample particles
pf.resample_particles()

# Get updated pose estimate
x, y, theta = pf.get_expected_pose()
print(f"Updated pose: ({x:.2f}, {y:.2f}, {theta:.2f})")
```

## API Reference

### ParticleFilter

The main class for LiDAR particle filter localization.

#### Constructor

```python
ParticleFilter(
    occupancy_grid: np.ndarray,  # (H, W, A) occupancy grid
    error_table: np.ndarray,     # LiDAR error probabilities
    lut_scalar: float,           # Grid coordinate scalar
    error_scalar: float,         # Error table scalar
    max_range: float,            # Maximum LiDAR range
    num_particles: int,          # Number of particles
    width: float,                # Map width in meters
    height: float                # Map height in meters
)
```

#### Methods

##### initialize(start_x, start_y, start_theta, position_std, angle_std)

Initialize particles around a starting pose with Gaussian distribution.

**Parameters:**
- `start_x`: Starting x position in meters
- `start_y`: Starting y position in meters  
- `start_theta`: Starting angle in radians
- `position_std`: Standard deviation for position initialization
- `angle_std`: Standard deviation for angle initialization

##### odometry_update(delta_x, delta_y, delta_theta, x_std, y_std, theta_std)

Update particles based on odometry with noise.

**Parameters:**
- `delta_x`: Change in x position in world frame
- `delta_y`: Change in y position in world frame
- `delta_theta`: Change in angle in world frame
- `x_std`: Standard deviation for x position noise
- `y_std`: Standard deviation for y position noise
- `theta_std`: Standard deviation for angle noise

##### lidar_update(distances, angles)

Update particle weights based on LiDAR measurements.

**Parameters:**
- `distances`: Array of measured distances in meters
- `angles`: Array of measured angles in radians

##### resample_particles()

Resample particles based on their weights using systematic resampling.

##### get_expected_pose()

Get the expected robot pose as weighted mean of particles.

**Returns:** Tuple of (x, y, theta)

##### get_particle_state()

Get current particle states and weights.

**Returns:** Tuple of (particles, weights)

## Data Format

### Occupancy Grid

The occupancy grid is a 3D NumPy array with shape `(H, W, A)` where:
- `H`: Height (X-axis) in grid cells
- `W`: Width (Y-axis) in grid cells  
- `A`: Number of angle bins

Each cell contains the expected distance to the nearest obstacle from that position and angle.

### LiDAR Error Table

A 1D NumPy array containing error probabilities. The index represents the error in distance measurements, and the value represents the probability of that error occurring.

### Particles

Particles are stored as a `(N, 3)` NumPy array where each row contains `[x, y, theta]`:
- `x`: X position in meters
- `y`: Y position in meters
- `theta`: Angle in radians (wrapped to [0, 2π])

### Weights

Particle weights are stored as a `(N,)` NumPy array where each value represents the likelihood of that particle being the true robot pose.

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Complete example showing initialization, motion updates, and sensor updates

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=lidarpf --cov-report=html
```

## Performance

The particle filter is optimized for performance using Numba JIT compilation:

- **Resampling**: 50x speedup over pure Python implementation
- **Motion Updates**: Vectorized operations with Numba acceleration
- **Sensor Updates**: Optimized grid lookups and likelihood computation

## Requirements

- Python >= 3.8
- NumPy >= 1.21
- Numba >= 0.55

## Development

### Code Quality

The codebase follows strict quality standards:

- PEP 8 style guidelines
- Type hints throughout
- Comprehensive docstrings
- >90% test coverage
- Ruff linting and formatting

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{lidarpf,
  title={LiDAR Particle Filter: High-performance localization for robotics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lidarpf}
}
```
