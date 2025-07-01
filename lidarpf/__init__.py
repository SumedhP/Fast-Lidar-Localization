"""
LiDAR Particle Filter - High-performance localization for robotics.

A Numba-optimized particle filter implementation for 2D LiDAR-based
robot localization with systematic resampling.
"""

from .numba_kernel import (
    chassis_odom_update,
    scan_update,
    resample,
)

__all__ = [
    "chassis_odom_update",
    "scan_update",
    "resample",
]
