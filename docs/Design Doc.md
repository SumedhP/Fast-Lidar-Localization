# LiDAR Particle Filter Design Doc

## Preface

### Goal:
Develop a fast, modular, and extensible localization system that:
- Accepts a occupancy grid map as an input
- Integrates real time LiDAR data, with a target device of the LD19 LiDAR
- Integrates chassis odometry data
- Performs localization using a particle filter
- Runs efficient on edge devices, with a target device of a NVIDIA Jetson Orin NX/AGX

### Challenges:
- Handling updates on large amounts of partcles quickly
- Memory management for particle storage
- Performing particle likelihood calculations efficiently

### Assumptions:
- The occupancy grid map is static (no dynamic obstacles)
- LiDAR data is on a 2D plane
- Chassis odometry data is present, with translation in both X and Y axes present
- Motion model for odometry and LiDAR data is Gaussian

---

## Protocol

### State
Particle:
X, Y, Theta representing the position and orientation of the particle in the map.
X is defined as positive upwards, Y is defined as positive to the right, and Theta is defined as positive counter-clockwise.
(0, 0, 0) is the origin of the map, located in the bottom right corner.

Each particle also has a weight, which is used to determine the likelihood of the particle being the true state of the robot.

Lidar Ray:
(Distance, Angle) representing the distance and angle of the LiDAR ray from the robot's position.

Lidar Scan:
A list of Lidar Rays representing a single scan from the LiDAR sensor.

Occupancy Grid Map:
A 2D grid representing the environment, where each cell can be occupied or free.

### Classes

