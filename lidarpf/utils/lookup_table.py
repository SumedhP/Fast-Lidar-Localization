import numpy as np
import numpy.typing as npt
from numba import njit, prange
from lidarpf.core.types import ParticleState


@njit
def _is_occupied(occupancy_grid: npt.NDArray[np.bool_], x: int, y: int) -> bool:
    """
    Check if a given cell in the occupancy grid is occupied.
    Out of bounds coordinates are considered occupied.
    """
    assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array."
    if x < 0 or y < 0:
        return True
    if x >= occupancy_grid.shape[ParticleState.X] or y >= occupancy_grid.shape[ParticleState.Y]:
        return True
    return occupancy_grid[x, y]


@njit
def _bresenhams_ray_cast(
    start_x: int, start_y: int, theta: float, occupancy_grid: npt.NDArray[np.bool_]
) -> float:
    """
    Perform Bresenham's ray casting algorithm to find the distance to the nearest obstacle from a given start point.
    """
    assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array."
    if (
        start_x < 0
        or start_y < 0
        or start_x >= occupancy_grid.shape[ParticleState.X]
        or start_y >= occupancy_grid.shape[ParticleState.Y]
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
    occupancy_grid: npt.NDArray[np.bool_], num_angles: int
) -> npt.NDArray[np.float64]:
    """
    Compute a lookup table for the distance to the nearest obstacle for each angle.
    """
    assert occupancy_grid.ndim == 2, "Occupancy grid must be a 2D array."
    height, width = occupancy_grid.shape
    lookup_table = np.zeros((height, width, num_angles), dtype=np.float64)
    for x in prange(height):  # type: ignore[not-iterable]
        for y in range(width):
            for angle in range(num_angles):
                theta = angle * (2 * np.pi / num_angles)
                lookup_table[x, y, angle] = _bresenhams_ray_cast(x, y, theta, occupancy_grid)
    return lookup_table


def visualize_occupancy_grid(occupancy_grid: npt.NDArray[np.bool_]) -> None:
    """
    Visualize the occupancy grid using datashader.
    """
    import datashader as ds
    import datashader.transfer_functions as tf

    canvas = ds.Canvas(plot_width=1920, plot_height=1080)
    agg = canvas.points(np.argwhere(occupancy_grid), agg=ds.count())
    img = tf.shade(agg, cmap=["white", "black"])
    img.to_pil().show()


if __name__ == "__main__":
    # Example usage
    occupancy_grid = np.random.choice([True, False], size=(12000, 800), p=[0.1, 0.9])
    num_angles = 120
    lookup_table = compute_lookup_table(occupancy_grid, num_angles)
    print("Lookup table shape:", lookup_table.shape)
    print("Sample distance at (0, 0) for angle 0:", lookup_table[0, 0, 0])
