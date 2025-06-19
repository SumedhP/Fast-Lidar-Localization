import numpy as np
import pytest
from lidarpf.filter.particle_filter import ParticleFilter
from lidarpf.core.types import ParticleState

class DummyLookupTable:
    def __getitem__(self, idx):
        return 1.0

def test_particle_filter_initialization():
    occupancy_grid = np.zeros((10, 10), dtype=bool)
    lookup_table = np.ones((10, 10, 8), dtype=np.float64)
    pf = ParticleFilter(occupancy_grid, lookup_table, num_particles=100)
    assert pf.occupancy_grid.shape == (10, 10)
    assert pf.lookup_table.shape == (10, 10, 8)
    assert pf.num_particles == 100
    assert pf.particles.shape == (100, 3)
    assert np.isclose(np.sum(pf.weights), 1.0)
    # Check enum usage
    assert np.all(pf.particles[:, ParticleState.X] == pf.particles[:, 0])
    assert np.all(pf.particles[:, ParticleState.Y] == pf.particles[:, 1])
    assert np.all(pf.particles[:, ParticleState.THETA] == pf.particles[:, 2])

def test_particle_filter_workflow():
    occupancy_grid = np.zeros((10, 10), dtype=bool)
    lookup_table = np.ones((10, 10, 8), dtype=np.float64)
    pf = ParticleFilter(occupancy_grid, lookup_table, num_particles=50)
    # Predict step
    odom = np.array([1.0, 0.0, 0.1], dtype=np.float32)
    noise = np.array([0.01, 0.01, 0.01], dtype=np.float32)
    pf.predict(odom, noise)
    # Update step
    scan = np.ones((5, 2), dtype=np.float32)
    map_resolution = 1.0
    sensor_params = {'sigma': 0.2, 'max_range': 10.0}
    pf.update(scan, map_resolution, sensor_params)
    assert np.isclose(np.sum(pf.weights), 1.0)
    # Resample step
    pf.resample()
    assert pf.particles.shape == (50, 3)
    assert np.isclose(np.sum(pf.weights), 1.0)
    # Estimate
    est = pf.estimate()
    assert est.shape == (3,) 
