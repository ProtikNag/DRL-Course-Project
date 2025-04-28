# utils.py
import numpy as np
import torch
import torch.nn.functional as F
from config import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT


def state_to_vector(state, item_dims, container_dims=None):
    """
    Convert the environment state (3D grid) and current item dimensions into a flat feature vector.

    Instead of flattening the full grid (which is huge), we apply 3D average pooling to reduce the
    dimensionality. For example, if our grid has shape (80, 60, 40) and we use a pooling kernel of 4,
    the pooled state will have shape (20, 15, 10). Then we flatten and concatenate with the normalized
    item dimensions.

    Parameters:
      state (numpy.ndarray): The 3D grid representing the container state (shape: (D, H, W)).
      item_dims (tuple): Dimensions (l, w, h) of the current item (in grid units).
      container_dims (tuple, optional): Container dimensions (L, W, H). Defaults to config values.

    Returns:
      torch.Tensor: A 1D tensor representing the state.
    """
    if container_dims is None:
        container_dims = (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert state to a tensor and add batch and channel dimensions.
    # Our state shape is assumed to be (D, H, W), where D=grid_L, H=grid_W, W=grid_H.
    # In our case, GRID_DIMS = (80, 60, 40) => D=80, H=60, W=40.
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(
        0)  # shape: (1, 1, 80, 60, 40)

    # Apply 3D average pooling with kernel_size 4.
    pooled = F.avg_pool3d(state_tensor, kernel_size=4)  # shape: (1, 1, 20, 15, 10)
    pooled = pooled.squeeze()  # shape: (20, 15, 10)
    pooled = pooled.flatten()  # shape: (20*15*10 = 3000,)
    pooled = pooled / 2.0  # normalize: grid cells are 0,1,2 so now in [0,1]

    # Normalize item dimensions by container dimensions.
    L, W, H = container_dims
    l_norm = item_dims[0] / L
    w_norm = item_dims[1] / W
    h_norm = item_dims[2] / H
    item_tensor = torch.tensor([l_norm, w_norm, h_norm], dtype=torch.float32, device=device)

    return torch.cat([pooled, item_tensor])


def adjust_item_dims_by_resolution(item_dims, grid_resolution):
    return dict([(key, int(val * grid_resolution)) for key, val in item_dims.items()])


def candidate_action_to_vector(action, container_dims=None):
    """
    Convert a candidate action into a normalized feature vector.

    Parameters:
      action (tuple): Candidate action as (x, y, z, orientation).
      container_dims (tuple, optional): Container dimensions (L, W, H). Defaults to config values.

    Returns:
      torch.Tensor: A tensor of shape (4,) with normalized x, y, z and orientation index.
    """
    from config import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT, ORIENTATIONS
    if container_dims is None:
        container_dims = (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)

    x, y, z, orientation = action
    L, W, H = container_dims
    x_norm = x / L
    y_norm = y / W
    z_norm = z / H
    ori_idx = 0 if orientation == ORIENTATIONS[0] else 1

    return torch.tensor([x_norm, y_norm, z_norm, ori_idx], dtype=torch.float32)


def compute_utilization(placed_items, container_dims=None):
    """
    Compute the utilization ratio.

    Parameters:
      placed_items (list): A list of dictionaries for each placed item.
         Each dictionary must have the keys 'l', 'w', 'h' representing the dimensions
         (in grid units) of the placed item.
      container_dims (tuple, optional): A tuple (L, W, H) representing container dimensions.
         If not provided, defaults to (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
         from the config.

    Returns:
      float: The total utilization ratio (occupied volume / container volume).
    """
    if container_dims is None:
        container_dims = (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)

    L, W, H = container_dims
    container_volume = L * W * H
    total_volume = sum(item['l'] * item['w'] * item['h'] for item in placed_items)
    return total_volume / container_volume


def get_state_tensor(state, item, grid_dims):
    grid_flat = state.flatten() / 2.0
    item_norm = np.array([
        item[0] / grid_dims[0],
        item[1] / grid_dims[1],
        item[2] / grid_dims[2]
    ])
    return torch.tensor(np.concatenate([grid_flat, item_norm]), dtype=torch.float32)