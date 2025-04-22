import numpy as np
import matplotlib.pyplot as plt
from env import BinPackingEnv
from heuristics import generate_extreme_points, priority_sort_extreme_points
from config import GRID_DIMS

def compute_utilization(placed_items):
    used = sum([item['l'] * item['w'] * item['h'] for item in placed_items])
    total = np.prod(GRID_DIMS)
    return used / total

def visualize_container(grid, title="Container View"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.where(grid == 0)
    ax.scatter(x, y, z, c='red', alpha=0.6, label='Occupied')

    x_s, y_s, z_s = np.where(grid == 2)
    ax.scatter(x_s, y_s, z_s, c='green', alpha=0.3, label='Supported (Available)')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    env = BinPackingEnv()
    initial_state = env.reset()

    placed_items = []
    for i in range(200):
        # Random item (baseline config)
        item = tuple(np.random.randint([5, 5, 3], [10, 10, 6]))

        # Generate and prioritize actions
        eps = generate_extreme_points(placed_items, env.grid_dims)
        sorted_eps = priority_sort_extreme_points(item, eps, env, placed_items)
        actions = env.get_available_actions(item, sorted_eps)
        if not actions:
            print(f"No feasible action for item {i+1}. Skipping...")
            continue

        x, y, z, ori = actions[0]
        dims = (item[1], item[0], item[2]) if ori == (0, 0, 90) else item
        success = env.place_item(x, y, z, *dims)

        if success:
            placed_items.append({'x': x, 'y': y, 'z': z, 'l': dims[0], 'w': dims[1], 'h': dims[2]})
            print(f"Item {i+1} placed at ({x},{y},{z}) with dims {dims}.")
        else:
            print(f"Item {i+1} could not be placed.")

    visualize_container(env.grid, "Final Container After Simulated Placements")
    utilization = compute_utilization(placed_items)
    print(f"Final space utilization: {utilization * 100:.2f}%")
