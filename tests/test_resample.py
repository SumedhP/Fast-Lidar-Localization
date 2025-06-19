import numpy as np
from lidarpf.filter.resample import numba_resample


def test_numba_resample_shape():
    weights = np.ones(100, dtype=np.float32)
    weights /= np.sum(weights)
    indices = numba_resample(weights)
    assert indices.shape == (100,)
    assert indices.dtype == np.int32
