# run_simulation.py
import numpy as np
import torch
import os
import random
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT, GRID_DIMS, POOL_KERNEL_SIZE, ORIENTATIONS, GRID_RESOLUTION
from env import BinPackingEnv
from dqn_agent import DQN
from utils import compute_utilization, state_to_vector, get_state_tensor
from heuristics import generate_extreme_points, priority_sort_extreme_points

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def get_random_color():
    """
    Return a random RGB color with full opacity (alpha = 1.0).
    """
    color = np.random.rand(3)  # random RGB in [0,1]
    return np.append(color, 1.0)  # append alpha=1.0


def get_cuboid_faces(x, y, z, l, w, h):
    """
    Given the lower-front-left corner (x, y, z) and dimensions (l, w, h) of a cuboid,
    return the six faces as lists of vertices.
    """
    v0 = [x, y, z]
    v1 = [x + l, y, z]
    v2 = [x + l, y + w, z]
    v3 = [x, y + w, z]
    v4 = [x, y, z + h]
    v5 = [x + l, y, z + h]
    v6 = [x + l, y + w, z + h]
    v7 = [x, y + w, z + h]
    faces = [
        [v0, v1, v2, v3],  # bottom
        [v4, v5, v6, v7],  # top
        [v0, v1, v5, v4],  # front
        [v2, v3, v7, v6],  # back
        [v1, v2, v6, v5],  # right
        [v3, v0, v4, v7]  # left
    ]
    return faces


def update_multi_views(ax1, ax2, ax3, ax4, placed_items, container_dims):
    L, W, H = container_dims
    L /= GRID_RESOLUTION
    W /= GRID_RESOLUTION
    H /= GRID_RESOLUTION

    # 3D View
    ax1.clear()
    container_corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for e in edges:
        ax1.plot3D(*zip(container_corners[e[0]], container_corners[e[1]]), color="black", linewidth=1)
    for item in placed_items:
        faces = get_cuboid_faces(item['x'], item['y'], item['z'], item['l'], item['w'], item['h'])
        cuboid = Poly3DCollection(faces, facecolors=item.get("color", "cyan"),
                                  edgecolors="black", linewidths=1, alpha=1.0)
        ax1.add_collection3d(cuboid)
    ax1.set_xlim([0, L])
    ax1.set_ylim([0, W])
    ax1.set_zlim([0, H])
    ax1.set_title("3D View")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.set_zlabel("Z (cm)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))
    ax1.zaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos: f"{int(z * GRID_RESOLUTION)}"))

    # Top (XY) View
    ax2.clear()
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, W])
    ax2.set_aspect('equal')
    rect_container = plt.Rectangle((0, 0), L, W, fill=False, color="black", linewidth=1)
    ax2.add_patch(rect_container)
    for item in placed_items:
        rect = plt.Rectangle((item['x'], item['y']), item['l'], item['w'],
                             fill=True, facecolor=item.get("color", "cyan"),
                             edgecolor="black", linewidth=1, alpha=1.0)
        ax2.add_patch(rect)
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    ax2.set_title("Top (XY) View")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))

    # YZ View
    ax3.clear()
    ax3.set_xlim([0, W])
    ax3.set_ylim([0, H])
    ax3.set_aspect('equal')
    rect_container_yz = plt.Rectangle((0, 0), W, H, fill=False, color="black", linewidth=1)
    ax3.add_patch(rect_container_yz)
    for item in placed_items:
        rect = plt.Rectangle((item['y'], item['z']), item['w'], item['h'],
                             fill=True, facecolor=item.get("color", "cyan"),
                             edgecolor="black", linewidth=1, alpha=1.0)
        ax3.add_patch(rect)
    ax3.set_xlabel("Y (cm)")
    ax3.set_ylabel("Z (cm)")
    ax3.set_title("YZ View")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))

    # XZ View
    ax4.clear()
    ax4.set_xlim([0, L])
    ax4.set_ylim([0, H])
    ax4.set_aspect('equal')
    rect_container_xz = plt.Rectangle((0, 0), L, H, fill=False, color="black", linewidth=1)
    ax4.add_patch(rect_container_xz)
    for item in placed_items:
        rect = plt.Rectangle((item['x'], item['z']), item['l'], item['h'],
                             fill=True, facecolor=item.get("color", "cyan"),
                             edgecolor="black", linewidth=1, alpha=1.0)
        ax4.add_patch(rect)
    ax4.set_xlabel("X (cm)")
    ax4.set_ylabel("Z (cm)")
    ax4.set_title("XZ View")
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))

    plt.tight_layout()
    plt.draw()
    plt.pause(0.5)


def visualize_all_views(placed_items, container_dims, save_path=None):
    """
    Create a figure with four subplots showing:
      - 3D view (upper left)
      - Top (XY) view (upper right)
      - YZ view (lower left)
      - XZ view (lower right)
    Each item is drawn with its random color (opaque).
    """
    L, W, H = container_dims
    L /= GRID_RESOLUTION
    W /= GRID_RESOLUTION
    H /= GRID_RESOLUTION
    fig = plt.figure(figsize=(16, 12))

    # --- 3D View ---
    ax1 = fig.add_subplot(221, projection='3d')
    container_corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for e in edges:
        ax1.plot3D(*zip(container_corners[e[0]], container_corners[e[1]]), color="black", linewidth=1)
    for item in placed_items:
        faces = get_cuboid_faces(item['x'], item['y'], item['z'], item['l'], item['w'], item['h'])
        cuboid = Poly3DCollection(faces, facecolors=item.get("color", "cyan"),
                                  edgecolors="black", linewidths=1, alpha=1.0)
        ax1.add_collection3d(cuboid)
    ax1.set_xlim([0, L])
    ax1.set_ylim([0, W])
    ax1.set_zlim([0, H])
    ax1.set_title("3D View")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.set_zlabel("Z (cm)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))
    ax1.zaxis.set_major_formatter(ticker.FuncFormatter(lambda z, pos: f"{int(z * GRID_RESOLUTION)}"))

    # --- Top (XY) View ---
    ax2 = fig.add_subplot(222)
    ax2.set_xlim([0, L])
    ax2.set_ylim([0, W])
    ax2.set_aspect('equal')
    rect_container = plt.Rectangle((0, 0), L, W, fill=False, color="black", linewidth=1)
    ax2.add_patch(rect_container)
    for item in placed_items:
        rect = plt.Rectangle((item['x'], item['y']), item['l'], item['w'],
                             fill=True, facecolor=item.get("color", "cyan"),
                             edgecolor="black", linewidth=1, alpha=1.0)
        ax2.add_patch(rect)
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    ax2.set_title("Top (XY) View")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))

    # --- YZ View ---
    ax3 = fig.add_subplot(223)
    ax3.set_xlim([0, W])
    ax3.set_ylim([0, H])
    ax3.set_aspect('equal')
    rect_container_yz = plt.Rectangle((0, 0), W, H, fill=False, color="black", linewidth=1)
    ax3.add_patch(rect_container_yz)
    for item in placed_items:
        rect = plt.Rectangle((item['y'], item['z']), item['w'], item['h'],
                             fill=True, facecolor=item.get("color", "cyan"),
                             edgecolor="black", linewidth=1, alpha=1.0)
        ax3.add_patch(rect)
    ax3.set_xlabel("Y (cm)")
    ax3.set_ylabel("Z (cm)")
    ax3.set_title("YZ View")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))

    # --- XZ View ---
    ax4 = fig.add_subplot(224)
    ax4.set_xlim([0, L])
    ax4.set_ylim([0, H])
    ax4.set_aspect('equal')
    rect_container_xz = plt.Rectangle((0, 0), L, H, fill=False, color="black", linewidth=1)
    ax4.add_patch(rect_container_xz)
    for item in placed_items:
        rect = plt.Rectangle((item['x'], item['z']), item['l'], item['h'],
                             fill=True, facecolor=item.get("color", "cyan"),
                             edgecolor="black", linewidth=1, alpha=1.0)
        ax4.add_patch(rect)
    ax4.set_xlabel("X (cm)")
    ax4.set_ylabel("Z (cm)")
    ax4.set_title("XZ View")
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x * GRID_RESOLUTION)}"))
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y * GRID_RESOLUTION)}"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=True)


def run_simulation_3d(checkpoint_path, num_test_items=20, visualize=True):
    """
    Run one simulation episode using the trained agent.
    If visualize=True, it shows multi-views.
    Returns: (number of items placed, utilization ratio)
    """
    container_dims = (CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)
    env = BinPackingEnv()

    input_dim = np.prod(GRID_DIMS) + 3
    max_actions = 20

    agent = DQN(input_dim, max_actions)
    if not os.path.exists(checkpoint_path):
        logging.error("Checkpoint not found at %s", checkpoint_path)
        exit(1)
    agent.load_state_dict(torch.load(checkpoint_path))
    agent.eval()
    logging.info("Checkpoint loaded. Running 3D simulation...")

    PREDEFINED_ITEM_SET2 = [
        (30, 40, 20),  # Orange
        (30, 50, 20),  # Blue
        (40, 50, 20),  # Purple
        (30, 50, 40),  # Green
        (40, 50, 30),  # Light blue
    ]
    predefined_items = [tuple([int(i / GRID_RESOLUTION) for i in item]) for item in PREDEFINED_ITEM_SET2]

    test_items = [random.choice(predefined_items) for _ in range(num_test_items)]

    placed_items = []
    extreme_points = [(0, 0, 0)]
    state = env.reset()

    if visualize:
        plt.ion()
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        update_multi_views(ax1, ax2, ax3, ax4, placed_items, container_dims)

    for idx, item in enumerate(test_items):
        agent.current_item_dims = item
        feasible_actions = env.get_available_actions(item, extreme_points)
        dims = (item[1], item[0], item[2])

        if not feasible_actions:
            logging.info("No feasible actions at step %d", idx)
            break

        state_tensor = get_state_tensor(state, item, GRID_DIMS)
        chosen_action = agent.select_action(state_tensor, feasible_actions)

        if chosen_action[3] == ORIENTATIONS[1]:
            dims = (item[1], item[0], item[2])
        else:
            dims = item

        placement_success = env.place_item(chosen_action[0], chosen_action[1], chosen_action[2],
                                           dims[0], dims[1], dims[2])
        if not placement_success:
            logging.warning("Placement failed at step %d", idx)
            continue

        new_item = {
            'x': chosen_action[0],
            'y': chosen_action[1],
            'z': chosen_action[2],
            'l': dims[0],
            'w': dims[1],
            'h': dims[2],
            'color': get_random_color()
        }
        placed_items.append(new_item)
        extreme_points = generate_extreme_points(placed_items, GRID_DIMS)
        extreme_points = priority_sort_extreme_points(item, extreme_points, env, placed_items)
        state = env.get_state()

        if visualize:
            update_multi_views(ax1, ax2, ax3, ax4, placed_items, container_dims)

    if visualize:
        plt.ioff()
        visualize_all_views(placed_items, container_dims, save_path="final_all_views.png")

    utilization_ratio = compute_utilization(placed_items, container_dims)
    num_items_placed = len(placed_items)

    return num_items_placed, utilization_ratio


if __name__ == "__main__":
    checkpoint_path = "model_weights/dqn_binpacking.pth"

    num_runs = 25
    total_items = 0
    total_utilization = 0.0

    visualize = False
    utilizations = []  # <--- store all utilization ratios here
    num_of_placements = []  # <--- store all item counts here

    for run in range(num_runs):
        if run > 0:
            visualize = False

        logging.info(f"Starting run {run+1}/{num_runs}...")
        num_items, utilization = run_simulation_3d(checkpoint_path, num_test_items=500, visualize=visualize)

        logging.info(f"Run {run+1}: Items Placed = {num_items}, Utilization Ratio = {utilization:.4f}")
        total_items += num_items
        total_utilization += utilization

        utilizations.append(utilization * 100)
        num_of_placements.append(num_items)

    avg_items = total_items / num_runs
    avg_utilization = total_utilization / num_runs

    print("=" * 60)
    print(f"utilization = {utilizations}")
    print(f"num_of_placements = {num_of_placements}")
    print("-" * 60)
    print(f"Average over {num_runs} runs:")
    print(f"- Average Number of Items Placed: {avg_items:.2f}")
    print(f"- Average Utilization Ratio: {avg_utilization * 100:.4f}%")
    print("=" * 60)
