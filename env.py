import numpy as np
from config import GRID_DIMS, GRID_RESOLUTION, CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT, \
    PARTIAL_SUPPORT_THRESHOLD


class BinPackingEnv:
    """
    Simulation environment for the online 3D bin-packing task.

    The container is represented as a 3D grid with each cell having one of three states:
      0: Occupied
      1: Empty but unsupported
      2: Empty and supported (i.e., stable for placement)

    When the environment is reset, the bottom layer is initialized as supported.
    """

    def __init__(self):
        # Get grid dimensions from config (already computed based on container dimensions and resolution)
        self.grid_dims = GRID_DIMS  # (grid_L, grid_W, grid_H)
        self.grid = None
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.
        The entire grid is set to 1 (empty, unsupported) except the bottom layer,
        which is set to 2 (supported, representing the container floor).
        Returns the initial state (the grid).
        """
        grid_L, grid_W, grid_H = self.grid_dims
        # Initialize all cells as empty but unsupported (state = 1)
        self.grid = np.ones((grid_L, grid_W, grid_H), dtype=np.int8)
        # Set the bottom layer (z=0) as supported (state = 2)
        self.grid[:, :, 0] = 2
        return self.get_state()

    def get_state(self):
        """
        Returns the current state representation.
        For simplicity, we return the grid array.
        (Later, additional features can be concatenated if needed.)
        """
        return self.grid.copy()

    def _is_within_bounds(self, x0, y0, z0, l, w, h):
        """
        Check if the proposed placement (starting at (x0, y0, z0) with size (l, w, h))
        is completely within the container bounds.
        """
        grid_L, grid_W, grid_H = self.grid_dims
        if x0 < 0 or y0 < 0 or z0 < 0:
            return False
        if (x0 + l) > grid_L or (y0 + w) > grid_W or (z0 + h) > grid_H:
            return False
        return True

    def check_overlap(self, x0, y0, z0, l, w, h):
        """
        Check if the proposed placement overlaps with any already occupied cell (state 0).
        Returns True if there is an overlap, False otherwise.
        """
        region = self.grid[x0:x0 + l, y0:y0 + w, z0:z0 + h]
        # Overlap if any cell in region is already occupied (0)
        return np.any(region == 0)

    def check_partial_support(self, x0, y0, z0, l, w, h):
        """
        Check whether the proposed placement satisfies the partial support constraints.
        For the base of the box (assumed to be at z=z0):
          - At least PARTIAL_SUPPORT_THRESHOLD (e.g., 0.5) fraction of the cells in the footprint
            should be supported (state 2).
          - The center cell of the footprint must be supported.

        Returns True if the placement passes the check; False otherwise.
        """
        # Ensure the base lies within the grid boundaries
        if not self._is_within_bounds(x0, y0, z0, l, w, 1):
            return False

        base_region = self.grid[x0:x0 + l, y0:y0 + w, z0]
        total_cells = l * w

        # Count number of supported cells (state 2) in the base region
        supported_cells = np.sum(base_region == 2)

        # Check fraction of supported cells
        if supported_cells < PARTIAL_SUPPORT_THRESHOLD * total_cells:
            return False

        # Check center cell: determine indices (using floor division)
        center_x = x0 + l // 2
        center_y = y0 + w // 2
        if self.grid[center_x, center_y, z0] != 2:
            return False

        return True

    def place_item(self, x0, y0, z0, l, w, h):
        """
        Place an item (box) in the container with the given starting coordinates and dimensions.
        This function first checks for boundaries, overlaps, and partial support.

        If placement is valid:
            - Update the grid cells in the region [x0:x0+l, y0:y0+w, z0:z0+h] to 0 (occupied).
            - Update cells above the placement as supported (set to 2) if appropriate.
        Returns True if the placement is successful, False otherwise.
        """
        # Check boundaries
        if not self._is_within_bounds(x0, y0, z0, l, w, h):
            return False

        # Check if the region is free (no overlap)
        if self.check_overlap(x0, y0, z0, l, w, h):
            return False

        # Check partial support at the base layer (z=z0)
        if not self.check_partial_support(x0, y0, z0, l, w, 1):
            return False

        # If all checks pass, mark the region as occupied (state 0)
        self.grid[x0:x0 + l, y0:y0 + w, z0:z0 + h] = 0

        # Update support for the layer immediately above the placed item (if within bounds)
        grid_L, grid_W, grid_H = self.grid_dims
        new_z = z0 + h
        if new_z < grid_H:
            # For cells directly above the item, set to supported (state 2)
            self.grid[x0:x0 + l, y0:y0 + w, new_z] = 2

        return True

    def get_available_actions(self, item_dims, extreme_points):
        """
        Generate a list of feasible placement actions for a given item.
        item_dims is a tuple (l, w, h) for the current item.
        extreme_points is a list of candidate (x, y, z) positions (from heuristics).

        For each candidate extreme point, check:
         - Whether the placement is within bounds.
         - Whether there is no overlap.
         - Whether the partial support constraints are met.

        Returns a list of feasible actions in the form:
         [(x, y, z, orientation), ...]
        Here, orientation can be one of the allowed ones defined in config (e.g., (0, 0, 0) or (0, 0, 90)).
        (Rotation logic can be integrated here based on the item dims.)
        """
        feasible_actions = []
        l, w, h = item_dims

        # Import allowed orientations from config
        from config import ORIENTATIONS

        for ep in extreme_points:
            # ep is a tuple (x, y, z); here we assume z is determined by the current container state
            x0, y0, z0 = ep
            for ori in ORIENTATIONS:
                # For simplicity, assume orientation only affects (l, w) by swapping if necessary.
                if ori == (0, 0, 90):
                    l_use, w_use = w, l
                else:
                    l_use, w_use = l, w

                # Check if placement at (x0, y0, z0) with size (l_use, w_use, h) is valid.
                if self._is_within_bounds(x0, y0, z0, l_use, w_use, h) and \
                        (not self.check_overlap(x0, y0, z0, l_use, w_use, h)) and \
                        self.check_partial_support(x0, y0, z0, l_use, w_use, 1):
                    feasible_actions.append((x0, y0, z0, ori))
        return feasible_actions


# If run as a script, you can perform a quick test.
if __name__ == "__main__":
    env = BinPackingEnv()
    state = env.reset()
    print("Initial grid shape:", state.shape)
    # Example item dimensions (in grid units) e.g., 30x40x20 cm -> 30, 40, 20 if resolution=1cm.
    example_item = (30, 40, 20)
    # For testing, define an extreme point candidate manually.
    # In practice, extreme points come from heuristic methods.
    extreme_points = [(50, 50, 0)]

    actions = env.get_available_actions(example_item, extreme_points)
    print("Feasible actions for the example item:", actions)

    # Attempt to place the item at the first feasible action if available.
    if actions:
        x, y, z, ori = actions[0]
        # Adjust dimensions if rotated
        if ori == (0, 0, 90):
            dims = (example_item[1], example_item[0], example_item[2])
        else:
            dims = example_item
        success = env.place_item(x, y, z, *dims)
        print("Placement success:", success)
    else:
        print("No feasible actions found.")