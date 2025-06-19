import numpy as np
from lidarpf.filter.sensor_model import compute_likelihoods

def test_compute_likelihoods_shape():
    particles = np.zeros((50, 3), dtype=np.float32)
    scan = np.zeros((10, 2), dtype=np.float32)
    lookup_table = np.ones((10, 10, 8), dtype=np.float64)
    map_resolution = 0.1
    sensor_params = {}
    likelihoods = compute_likelihoods(particles, scan, lookup_table, map_resolution, sensor_params)
    assert likelihoods.shape == (50,) 
