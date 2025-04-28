# run_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import torch
import random
import argparse
import logging
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import RANDOM_SEED, CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT, GRID_DIMS, POOL_KERNEL_SIZE, \
    ORIENTATIONS, GRID_RESOLUTION, PREDEFINED_ITEM_SET1
from env import BinPackingEnv
# from heuristics import generate_extreme_points, priority_sort_extreme_points
from dqn_agent import DQN
from utils import adjust_item_dims_by_resolution, compute_utilization

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')


def create_box_vertices(x, y, z, l, w, h):
    """
    Create vertices for a 3D box.

    Parameters:
        x, y, z: coordinates of the bottom-left-front corner
        l, w, h: length, width, height of the box

    Returns:
        List of vertices and faces for the box
    """
    vertices = [
        [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],
        [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]  # right
    ]

    return faces


def plot_container_and_items(ax, placed_items, container_dims, view='3d'):
    """
    Plot the container and placed items.

    Parameters:
        ax: matplotlib axis
        placed_items: list of placed items
        container_dims: tuple (L, W, H) representing container dimensions
        view: '3d', 'xy', 'yz', or 'xz' for different projections
    """
    L, W, H = container_dims
    placed_items = [adjust_item_dims_by_resolution(item, GRID_RESOLUTION) for item in placed_items]

    if view == '3d':
        # Plot container wireframe
        container_edges = [
            [(0, 0, 0), (L, 0, 0)], [(0, 0, 0), (0, W, 0)], [(0, 0, 0), (0, 0, H)],
            [(L, 0, 0), (L, W, 0)], [(L, 0, 0), (L, 0, H)], [(0, W, 0), (L, W, 0)],
            [(0, W, 0), (0, W, H)], [(0, 0, H), (L, 0, H)], [(0, 0, H), (0, W, H)],
            [(L, W, 0), (L, W, H)], [(L, 0, H), (L, W, H)], [(0, W, H), (L, W, H)]
        ]

        for edge in container_edges:
            ax.plot3D(*zip(*edge), color='black', linestyle='--', alpha=0.3)

        # Plot placed items
        for i, item in enumerate(placed_items):
            x, y, z = item['x'], item['y'], item['z']
            l, w, h = item['l'], item['w'], item['h']
            color = plt.cm.tab20(i % 20)  # Use tab20 colormap for more distinct colors

            faces = create_box_vertices(x, y, z, l, w, h)
            ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', alpha=0.7))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
        ax.set_title('3D View')

    elif view == 'xy':
        # Plot container boundary
        ax.add_patch(patches.Rectangle((0, 0), L, W, fill=False, edgecolor='black', linestyle='--'))

        # Plot placed items (top view)
        for i, item in enumerate(placed_items):
            x, y = item['x'], item['y']
            l, w = item['l'], item['w']
            color = plt.cm.tab20(i % 20)

            ax.add_patch(patches.Rectangle((x, y), l, w, fill=True, edgecolor='black', facecolor=color, alpha=0.7))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_title('XY Projection (Top View)')
        ax.set_aspect('equal')

    elif view == 'yz':
        # Plot container boundary
        ax.add_patch(patches.Rectangle((0, 0), W, H, fill=False, edgecolor='black', linestyle='--'))

        # Plot placed items (side view)
        for i, item in enumerate(placed_items):
            y, z = item['y'], item['z']
            w, h = item['w'], item['h']
            color = plt.cm.tab20(i % 20)

            ax.add_patch(patches.Rectangle((y, z), w, h, fill=True, edgecolor='black', facecolor=color, alpha=0.7))

        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_title('YZ Projection (Side View)')
        ax.set_aspect('equal')

    elif view == 'xz':
        # Plot container boundary
        ax.add_patch(patches.Rectangle((0, 0), L, H, fill=False, edgecolor='black', linestyle='--'))

        # Plot placed items (front view)
        for i, item in enumerate(placed_items):
            x, z = item['x'], item['z']
            l, h = item['l'], item['h']
            color = plt.cm.tab20(i % 20)

            ax.add_patch(patches.Rectangle((x, z), l, h, fill=True, edgecolor='black', facecolor=color, alpha=0.7))

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_title('XZ Projection (Front View)')
        ax.set_aspect('equal')


def run_simulation_3d(checkpoint_path, item_set, delay=0.5, num_runs=100):
    """
    Run one simulation episode using the trained PPO agent and update an interactive multi-view figure.
    After each placement, update four views: 3D, Top (XY), YZ, and XZ.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    container_dims = (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
    predefined_items = [tuple([int(i / GRID_RESOLUTION) for i in item]) for item in item_set]

    # Calculate pooled state dimension.
    pooled_dims = tuple([index // POOL_KERNEL_SIZE for index in GRID_DIMS])
    pooled_size = np.prod(pooled_dims)
    state_dim = int(pooled_size + 3)
    action_feat_dim = 4

    # Initialize the environment
    env = BinPackingEnv()

    # Initialize PPO agent
    agent = PPOAgent(state_dim, action_feat_dim, hidden_dim=256)

    # Load checkpoint if provided
    agent.load_checkpoint(checkpoint_path)
    logging.info("Checkpoint loaded. Running 3D simulation test episode...")

    utilization_ratio_per_run = []
    num_of_placement_per_run = []
    for i in range(num_runs):
        # Set up interactive multi-view figure.
        # plt.ion()

        state = env.reset()

        count = 0
        is_container_full = False
        item_volume = 0
        while not is_container_full:
            item = random.choice(predefined_items)
            agent.current_item_dims = item
            logging.info("Test Episode: item dimenstion at step %d; with dims %s.", count, item)

            # Get feasible actions from extreme points
            feasible_actions = env.get_actions(item)
            if not feasible_actions:
                logging.info("Test Episode: No feasible actions at step %d; ending episode.", count)
                break

            chosen_action, _, _ = agent.select_action(state=env.get_current_state(),
                                                      candidate_actions=feasible_actions,
                                                      container_dims=env.grid_dims)

            # try to place the item according to chosen_action
            placement_success = env.place_item(item, chosen_action[1:])
            if not placement_success:
                logging.warning("Test Episode: Placement failed at step %d.", count)
                is_container_full = True
            else:
                count += 1
                item_volume += np.prod(item)
                utilization_ratio = (item_volume / np.prod(env.grid_dims)) * 100
                logging.info("Test Episode: Placed item %d at %s with dims %s.", count, chosen_action, item)

        utilization_ratio = compute_utilization(env.placed_items, env.grid_dims) * 100
        utilization_ratio_per_run.append(utilization_ratio)
        num_of_placement_per_run.append(len(env.placed_items))

        logging.info("Episode %d: Number of Item Placed=%.0f, Total Utilization Ratio=%.2f%%",
                     i, len(env.placed_items), utilization_ratio)

        if utilization_ratio >= 60:
            fig = plt.figure(figsize=(15, 10))
            ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
            ax_xy = fig.add_subplot(2, 2, 2)
            ax_yz = fig.add_subplot(2, 2, 3)
            ax_xz = fig.add_subplot(2, 2, 4)

            # # Update plots
            plot_container_and_items(ax_3d, env.placed_items, container_dims, view='3d')
            plot_container_and_items(ax_xy, env.placed_items, container_dims, view='xy')
            plot_container_and_items(ax_yz, env.placed_items, container_dims, view='yz')
            plot_container_and_items(ax_xz, env.placed_items, container_dims, view='xz')

            # Add statistics to the figure
            plt.figtext(0.5, 0.01, f"Items Placed: {count} | Space Utilization: {utilization_ratio:.2f}%",
                        ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

            # Update the display
            plt.tight_layout()
            plt.draw()

            # plt.tight_layout()
            save_path = checkpoint_path.replace('.pth', f"_views{i:.0f}.png")
            plt.savefig(save_path)
            plt.close()

    return utilization_ratio_per_run, num_of_placement_per_run


if __name__ == "__main__":
    checkpoint_path = "/model_weights/dqn_binpacking.pth"  # Adjust as needed.
    utilization_ratios, num_of_placements = run_simulation_3d(checkpoint_path, PREDEFINED_ITEM_SET1, num_runs=25)
    print(
        f"Total Utilization Ratio: average = {np.mean(utilization_ratios):.2f}%, median = {np.median(utilization_ratios):.2f}%, max = {np.max(utilization_ratios):.2f}%, min = {np.min(utilization_ratios):.2f}%")
    print(
        f"Number of Item Placed: average = {np.mean(num_of_placements):.0f}, median = {np.median(num_of_placements):.0f}, max = {np.max(num_of_placements):.0f}, min = {np.min(num_of_placements):.0f}")
    print(utilization_ratios)
    print(num_of_placements)
