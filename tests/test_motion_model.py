import numpy as np
from lidarpf.filter.motion_model import apply_motion_model

def test_apply_motion_model_shape():
    particles = np.zeros((100, 3), dtype=np.float32)
    delta = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    noise_std = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    updated = apply_motion_model(particles, delta, noise_std)
    assert updated.shape == (100, 3) 
