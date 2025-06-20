"""
LiDAR Particle Filter - High-performance localization for robotics.

A Numba-optimized particle filter implementation for 2D LiDAR-based
robot localization with systematic resampling.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .particle_filter import ParticleFilter
from .types import ParticleState, LidarScan, OccupancyGrid, LidarErrorTable

__all__ = [
    "ParticleFilter",
    "ParticleState", 
    "LidarScan",
    "OccupancyGrid",
    "LidarErrorTable",
] 
