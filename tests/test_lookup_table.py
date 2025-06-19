import numpy as np
from lidarpf.utils.lookup_table import compute_lookup_table

def test_compute_lookup_table_shape():
    occupancy_grid = np.zeros((5, 5), dtype=bool)
    num_angles = 8
    table = compute_lookup_table(occupancy_grid, num_angles)
    assert table.shape == (5, 5, 8) 
