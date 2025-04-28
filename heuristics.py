import numpy as np
from config import GRID_RESOLUTION, PARTIAL_SUPPORT_THRESHOLD, ORIENTATIONS, RHO, MU, TAU, OMEGA, NEIGHBOR_BONUS


#########################
# Extreme Point Functions
#########################

def generate_extreme_points(placed_items, container_dims):
    """
    Generate candidate extreme points from already placed items.
    Each placed item is assumed to be a dictionary with keys:
      'x', 'y', 'z', 'l', 'w', 'h'
    where (x,y,z) is the lower (front-left-bottom) corner and (l,w,h) are dimensions.

    For each item, generate three extreme points:
      - Upper extreme: (x, y, z + h)
      - Left-front extreme: (x, y + w, z)
      - Right-rear extreme: (x + l, y, z)

    Also, include the container's origin (0,0,0) if not already present.

    container_dims: tuple (L, W, H) in grid units.

    Returns a list of extreme points as tuples (x, y, z).
    """
    extreme_points = set()
    extreme_points.add((0, 0, 0))

    for item in placed_items:
        x, y, z = item['x'], item['y'], item['z']
        l, w, h = item['l'], item['w'], item['h']
        ep_upper = (x, y, z + h)
        ep_left_front = (x, y + w, z)
        ep_right_rear = (x + l, y, z)
        extreme_points.update([ep_upper, ep_left_front, ep_right_rear])

    L, W, H = container_dims
    # extreme_points = [ep for ep in extreme_points if ep[0] < L and ep[1] < W and ep[2] < H]
    extreme_points = [ep for ep in extreme_points if ep[0] <= L - 1 and ep[1] <= W - 1 and ep[2] <= H - 1]
    return extreme_points


def calculate_remaining_space(item_dims, ep, env):
    """
    For a given item (item_dims: (l, w, h)) and candidate extreme point ep = (x, y, z),
    simulate placing the item at that point and estimate the residual space along the x and y axes.

    Uses GRID_RESOLUTION from config to scale if needed.

    Returns a scalar value representing the estimated residual space.
    """
    l, w, h = item_dims
    x0, y0, z0 = ep
    grid = env.grid
    grid_L, grid_W, _ = grid.shape

    # Calculate contiguous free cells along x-axis starting at x0+l.
    available_x = 0
    for x in range(x0 + l, grid_L):
        region = grid[x, y0:y0 + w, z0]
        if np.any(region == 0):
            break
        available_x += 1

    # Calculate contiguous free cells along y-axis starting at y0+w.
    available_y = 0
    for y in range(y0 + w, grid_W):
        region = grid[x0:x0 + l, y, z0]
        if np.any(region == 0):
            break
        available_y += 1

    # Multiply by GRID_RESOLUTION if you need a real-world measure; here grid units are used directly.
    rem_space = available_x * available_y
    return rem_space


def dec_waste_space(rem_space, placed_items):
    """
    Decide if the residual space (rem_space) is wasted.
    If rem_space is 0, treat it as a perfect fit (return False).
    Otherwise, compare rem_space with the minimum base area among placed items.

    Returns True if rem_space is too small (i.e., wasted), False otherwise.
    """
    if rem_space == 0:
        return False  # Perfect fit.
    if not placed_items:
        return False  # No items placed yet.
    min_area = min(item['l'] * item['w'] for item in placed_items)
    return rem_space < min_area


def priority_sort_extreme_points(item_dims, extreme_points, env, placed_items):
    """
    For the current item (dimensions item_dims) and candidate extreme points,
    compute the residual space for each candidate and sort the points based on waste.

    For each extreme point, compute:
      rem_space = calculate_remaining_space(item_dims, ep, env)
      If dec_waste_space returns True, assign score = -rem_space; else score = 0.

    Return the sorted list of extreme points (best candidate first).
    """
    scores = []
    for ep in extreme_points:
        rem_space = calculate_remaining_space(item_dims, ep, env)
        score = -rem_space if dec_waste_space(rem_space, placed_items) else 0
        scores.append((ep, score))

    # Higher score is better (0 is highest; negative values indicate waste).
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_extreme_points = [ep for ep, _ in scores_sorted]
    return sorted_extreme_points


def sort_extreme_points_flatness(extreme_points, env):
    """
    Sort extreme points preferring low z, flatness, and centrality.
    """
    sorted_eps = []
    grid = env.grid
    grid_L, grid_W, grid_H = grid.shape
    center_x, center_y = grid_L // 2, grid_W // 2

    for ep in extreme_points:
        x, y, z = ep
        flatness = 0
        # Count how many adjacent x-y cells are empty (state 1 or 2)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_L and 0 <= ny < grid_W:
                if grid[nx, ny, z] in (1, 2):  # Empty or supported
                    flatness += 1

        center_bonus = - (abs(x - center_x) + abs(y - center_y)) / (grid_L + grid_W)  # normalized distance
        score = -z + 0.5 * flatness + 0.1 * center_bonus
        sorted_eps.append((ep, score))

    # Sort descending: best score first
    sorted_eps.sort(key=lambda x: x[1], reverse=True)
    return [ep for ep, _ in sorted_eps]


#########################
# DBLF Heuristic Functions
#########################

def initialize_score_grid(container_dims):
    """
    Initialize a 3D score grid for DBLF heuristic.
    Score is defined to decrease from the bottom-left corner to the top-right corner.
    For a cell at (x,y,z), one option is:
         score = (L - x) + (W - y) + (H - z)
    container_dims: tuple (L, W, H) in grid units.

    Returns a NumPy array with initial scores.
    """
    L, W, H = container_dims
    score_grid = np.zeros((L, W, H), dtype=np.float32)
    for x in range(L):
        for y in range(W):
            for z in range(H):
                score_grid[x, y, z] = (L - x) + (W - y) + (H - z)
    return score_grid


def update_score_grid(score_grid, env):
    """
    Update the score grid based on the current grid state.
    - For cells that are occupied (state 0), set score to 0.
    - For empty cells adjacent (in x-y plane) to occupied cells, add NEIGHBOR_BONUS.
    The updates are done in place.
    """
    L, W, H = score_grid.shape
    grid = env.grid
    # Define 4-connected neighbor offsets in the x-y plane.
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
    for x in range(L):
        for y in range(W):
            for z in range(H):
                if grid[x, y, z] == 0:
                    score_grid[x, y, z] = 0
                else:
                    bonus = 0
                    for dx, dy, dz in offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < L and 0 <= ny < W:
                            if grid[nx, ny, z] == 0:
                                bonus += NEIGHBOR_BONUS
                    score_grid[x, y, z] += bonus
    return score_grid


def compute_dblf_score(placement, item_dims, score_grid):
    """
    Compute the DBLF score for a candidate placement.
    placement: tuple (x, y, z, orientation)
    item_dims: tuple (l, w, h); adjust if rotated.
    score_grid: 3D score grid.

    The DBLF score is the sum of scores over the region the item would occupy.
    """
    x0, y0, z0, ori = placement
    l, w, h = item_dims
    # Adjust dimensions for a 90° rotation if needed.
    if ori == (0, 0, 90):
        l, w = w, l
    region = score_grid[x0:x0 + l, y0:y0 + w, z0]
    dblf_score = np.sum(region)
    return dblf_score