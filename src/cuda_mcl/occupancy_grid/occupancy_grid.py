import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


class OccupancyGrid:
    def __init__(self, height: int, width: int):
        self.grid = np.zeros((height, width), dtype=np.uint8)
    
    def fill_rectangle(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Fill a rectangle in the occupancy grid with a specified value.

        Args:
            x1 (int): Top-left x coordinate.
            y1 (int): Top-left y coordinate.
            x2 (int): Bottom-right x coordinate.
            y2 (int): Bottom-right y coordinate.
            value (int): Value to fill the rectangle with.
        """
        self.grid[y1:y2, x1:x2] = 1

    def reset(self) -> None:
        """
        Reset the occupancy grid to all zeros.
        """
        self.grid.fill(0)

    def visualize(self) -> None:
        """
        Visualize the occupancy grid using matplotlib.
        """
        # Flip the image horizontally and then vertically to match the expected orientation, where 0,0 is the bottom-right corner
        plt.imshow(np.fliplr(self.grid), cmap='gray', origin='lower')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Occupancy Grid')
        plt.show()

def get_rmna_2024_grid() -> OccupancyGrid:
    """
    Create and return the RMNA 2024 field as an occupancy grid.

    Returns:
        OccupancyGrid: An occupancy grid representing the RMNA 2024 field.
    """
    grid = OccupancyGrid(1200, 800)

    grid.fill_rectangle(400, 200, 800, 300)

    return grid

if __name__ == "__main__":
    grid = get_rmna_2024_grid()
    grid.visualize()
