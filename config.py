# config.py
RANDOM_SEED = 45

# Container dimensions (in centimeters)
CONTAINER_LENGTH = 400   # L
CONTAINER_WIDTH  = 300   # W
CONTAINER_HEIGHT = 200   # H

# Grid resolution (each cell represents 1 cm)
GRID_RESOLUTION = 10

# Derived grid dimensions based on container dimensions and resolution.
# For a container of 400x300x200 and resolution 1 cm, the grid is:
GRID_DIMS = (
    int(CONTAINER_LENGTH / GRID_RESOLUTION),  # e.g., 400/5=80 if using 5 cm, here 400 if resolution=1cm
    int(CONTAINER_WIDTH / GRID_RESOLUTION),
    int(CONTAINER_HEIGHT / GRID_RESOLUTION)
)

# To reduce the state vector size, we will use 3D pooling.
POOL_KERNEL_SIZE = 4  # For average pooling

# Partial support threshold: fraction of the item’s base that must be supported (state=2)
PARTIAL_SUPPORT_THRESHOLD = 0.5

# Allowed orientations (only two are allowed)
ORIENTATIONS = [(0, 0, 0), (0, 0, 90)]

MAX_STACK_LEVEL = 10
MIN_EXTREME_POINTS = 5
FLOOR_SCAN_STEP = 5
TOP_TOLERANCE = 0.2        # Allowed tolerance (in grid units) for candidate's z vs. current top.
FLOOR_SCAN_STRIDE = 5    # Step size when scanning the floor for candidate placements.
# config.py
# ... other constants ...
NEIGHBORHOOD_WINDOW = 5      # Window size (in grid cells) for computing local neighborhood maximum.
HIGH_ITEM_HEIGHT_THRESHOLD = 40   # Example: items with height >= 40 (grid units) are considered tall.

FLOOR_SAMPLE_STRIDE = 10     # When sampling the floor, check every 10 grid cells.
MIN_CANDIDATE_DISTANCE = 5   # Minimum Euclidean distance between candidate placements (in grid units).

# Reward parameters (example values)
SIGMA = 80         # Standard for space utilization
BETA = 0.2         # β constant
GAMMA = 0.6        # γ constant (2β + γ should equal 1)
RHO   = 0.7        # Step reward constant
MU    = 0.3        # Penalty constant
TAU   = 0.2        # Step reward constant
OMEGA = 0.8        # Step reward constant

# Additional constants for heuristics
NEIGHBOR_BONUS = 3

PREDEFINED_ITEM_SET1 = [
    (30, 40, 20),  # Orange
    (30, 50, 20),  # Blue
    (40, 50, 20),  # Purple
    (30, 50, 40),  # Green
    (40, 50, 30),  # Light blue
]