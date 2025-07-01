Lidar PF

1) Meant to work with a 2D lidar
2) Uses a particle filter to track (x, y, theta) of the robot
3) Expected pose is defined as mean of the particles, weighted by their likelihood
4) Particles get resampled based on their likelihood
5) Everything needs to be writtten with Numpy and Numba for speed, this will be run on a Jetson Orin NX
6) The user will provide a look up table, at no point should we care about the actual map or occupancy grid

Conventions:
- Particles will have position in meters and radians
- Particles cannot have a position less than 0, and cannot exceed map size
- Particles will have a theta that gets wrapped between 0 and 2pi
- (0,0) is represnted as the bottom right of the occupancy grid and we follow the FLU convention for XYZ

States:

Particles:
(N, 3) array for N particles, each with a position and angle. This will be in fp32
Expected Unit: Meters and radians

Weights:
(N,) array for N particles, each is a fp32 value of it's likelihood

Occupancy grid:
(H, W, A) where H is height (X-axis), W is width (y-axis), and A is unique angles (allowing for angle discretization).
The expected size of this occupancy grid is (1200, 800, 120), where we have cm precision across a 12m x 8m field, and bin every 
3 degrees into 1 bucket. The CDDT paper showed no loss in accuracy through this metric
Expected Unit: Centimeters and radians

LiDAR error probabilty:
(2 * max_range * precision, ) array for 1D lidar error. This is defiend as the probability that a given amount of error between
expected measurement and actual measurement occurs. Since the LD19 has a maximum range of 12m, this would be a 24 * precision array.
Since our precision increment we'd want is in cm, this would be a 2400 fp32 array that gets generated
Expected Unit: Centimeters for index, probability for value

LiDAR scan:
(N, 2) array for N LiDAR scans, each with a distance and angle. This is in fp32
Expected Unit: Meters and radians

How we're doing this:

Resampling particles:
- Particles are resampled based on Systemic resampling algorithm. Particles that are more likely to be the true pose of the robot
will have a higher chance of being selected in the next iteration. This is using FilterPy's implementation, but wrapped in numba
for a 50x speed up.

Particle odometry update:
- Particles get updated with an delta X, Y, and theta all in world frame. Each particle also gets added some noise in measurement
across x, y, and theta

LiDAR update:
- Generate the set of expected distance readings given the particles theta and the new angle for each scan
- Get these expected distances by converting the particles position into the LUT position (by LUT scalar)
- Get angle that falls into bin
- After getting expected distance, subtract expected - measured to get error, and put that into the error table to get likelihood
- Particle probabilty is those liklihoods multiplied across all scan readings

Classes:

Particle filter class:

On creation:
- Take in LUT for expected distances, LiDAR std dev, LiDAR max range, number of particles
- Scalar between LUT and particle position
- Scalar for error table
- weight is None

Initialize (start X, start y, start theta, position_std_dev, angle_std_dev):
- Creates particles with a uniform distribution across the map, with a std dev in position and angle
- Creates weights with a uniform distribution across all particles (1/N)

odometry_update(delta_x, delta_y, delta_theta, x_std_dev, y_std_dev, theta_std_dev):
- Updates each particle with the delta x, y, and theta in world frame
- Adds noise to each particle based on the std devs provided

lidar_update(lidar_scan):
- For each particle, gets the expected distance reading based on the LUT and the particle's theta
- For each expected distance, gets the error between expected and measured
- Gets the likelihood of that error from the error table
- Multiplies the likelihoods across all scans to get the particle's probability

resample_particles():
- Uses the systemic resampling algorithm to resample particles based on their likelihoods

