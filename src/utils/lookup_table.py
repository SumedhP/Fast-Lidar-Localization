import numpy as np
import numpy.typing as npt
from numba import njit, prange
from src.utils.constants import X, Y, THETA

@njit
def _is_occupied(occupancy_grid: npt.NDArray[np.bool], x: int, y: int) -> bool:
    """
    Check if a given cell in the occupancy grid is occupied.

    Out of bounds coordinates are considered occupied.

    Args:
        occupancy_grid (npt.NDArray[np.bool]): A 2D occupancy grid where True indicates an obstacle.
        x (int): The x-coordinate of the cell to check.
        y (int): The y-coordinate of the cell to check.

    Returns:
        bool: True if the cell is occupied, False otherwise.
    """
    assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array."
    if x < 0 or y < 0:
        return True
    if x >= occupancy_grid.shape[X] or y >= occupancy_grid.shape[Y]:
        return True

    return occupancy_grid[x, y]


@njit
def _bresenhams_ray_cast(
    start_x: int, start_y: int, theta: float, occupancy_grid: npt.NDArray[np.bool]
) -> float:
    """
    Perform Bresenham's ray casting algorithm to find the distance to the nearest obstacle
    from a given start point in the occupancy grid.

    Args:
        start_x (int): The x-coordinate of the starting point.
        start_y (int): The y-coordinate of the starting point.
        theta (float): The angle in radians from which to cast the ray.
        occupancy_grid (npt.NDArray[np.bool]): A 2D occupancy grid where True indicates an obstacle.

    Returns:
        float: The distance to the nearest obstacle along the ray.
    """

    assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array."

    if (
        start_x < 0
        or start_y < 0
        or start_x >= occupancy_grid.shape[X]
        or start_y >= occupancy_grid.shape[Y]
    ):
        return 0

    dx = np.cos(theta)
    dy = np.sin(theta)

    x, y = start_x, start_y
    while not _is_occupied(occupancy_grid, x, y):
        x += dx
        y += dy
    return np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)


@njit(parallel=True)
def compute_lookup_table(
    occupancy_grid: npt.NDArray[np.bool], num_angles: int
) -> npt.NDArray[np.float64]:
    """
    Compute a lookup table for the distance to the nearest obstacle for each angle.

    Args:
        occupancy_grid (npt.NDArray[np.bool]): A 2D occupancy grid where True indicates an obstacle.
        num_angles (int): The number of angles to compute the lookup table for.

    Returns:
        npt.NDArray[np.float64]: A 2D array where each row corresponds to an angle and contains the distances
                                  to the nearest obstacle from each cell in the occupancy grid.
    """
    assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array."
    height, width = occupancy_grid.shape
    lookup_table = np.zeros((height, width, num_angles), dtype=np.float64)

    for x in prange(height):  # type: ignore[not-iterable]
        for y in range(width):
            for angle in range(num_angles):
                theta = angle * (2 * np.pi / num_angles)
                lookup_table[x, y, angle] = _bresenhams_ray_cast(
                    x, y, theta, occupancy_grid
                )

    return lookup_table

def visualize_occupancy_grid(occupancy_grid: npt.NDArray[np.bool]) -> None:
    """
    Visualize the occupancy grid using datashader.

    Args:
        occupancy_grid (npt.NDArray[np.bool]): A 2D occupancy grid where True indicates an obstacle.
    """
    import datashader as ds
    import datashader.transfer_functions as tf

    canvas = ds.Canvas(plot_width=1920, plot_height=1080)
    agg = canvas.points(np.argwhere(occupancy_grid), agg=ds.count())
    img = tf.shade(agg, cmap=["white", "black"])
    img.to_pil().show()

if __name__ == "__main__":
    # Example usage
    # Generate a 12000x800 occupancy grid with random obstacles
    occupancy_grid = np.random.choice([True, False], size=(12000, 800), p=[0.1, 0.9])
    num_angles = 120
    lookup_table = compute_lookup_table(occupancy_grid, num_angles)
    print("Lookup table shape:", lookup_table.shape)
    print("Sample distance at (0, 0) for angle 0:", lookup_table[0, 0, 0])
