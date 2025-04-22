import numpy as np
import torch
import torch.optim as optim
from env import BinPackingEnv
from heuristics import generate_extreme_points, priority_sort_extreme_points
from dqn_agent import DQN
from replay_buffer import ReplayBuffer
from config import GRID_DIMS, ORIENTATIONS
import matplotlib.pyplot as plt
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_state_tensor(state, item):
    grid_flat = state.flatten() / 2.0
    item_norm = np.array([item[0] / GRID_DIMS[0], item[1] / GRID_DIMS[1], item[2] / GRID_DIMS[2]])
    return torch.tensor(np.concatenate([grid_flat, item_norm]), dtype=torch.float32)


def dqn_train(num_episodes=100, batch_size=32, lr=1e-3, gamma=0.99,
              epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.05):
    env = BinPackingEnv()
    input_dim = np.prod(GRID_DIMS) + 3
    max_actions = 10

    dqn = DQN(input_dim, max_actions).to(device)
    target_dqn = DQN(input_dim, max_actions).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    buffer = ReplayBuffer(500)

    epsilon = epsilon_start

    episode_rewards = []
    volume_utilizations = []

    log_path = "logs/training_log.csv"
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Reward", "VolumeUtilization", "Epsilon"])

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        placed_items = []
        used_volume = 0

        for step in range(100):
            item = tuple(np.random.randint([5, 5, 3], [10, 10, 6]))
            item_volume = np.prod(item)

            eps = generate_extreme_points(placed_items, GRID_DIMS)
            sorted_eps = priority_sort_extreme_points(item, eps, env, placed_items)
            actions = env.get_available_actions(item, sorted_eps)
            if not actions:
                break

            actions = actions[:max_actions]
            state_tensor = get_state_tensor(state, item).to(device)

            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(actions))
            else:
                with torch.no_grad():
                    q_vals = dqn(state_tensor.unsqueeze(0)).squeeze()
                    action_idx = torch.argmax(q_vals[:len(actions)]).item()

            x, y, z, ori = actions[action_idx]
            dims = (item[1], item[0], item[2]) if ori == (0, 0, 90) else item
            success = env.place_item(x, y, z, *dims)

            reward = 1.0 if success else -1.0
            next_state = env.get_state()
            done = not success

            buffer.push(state_tensor.numpy(), action_idx, reward, get_state_tensor(next_state, item).numpy(), done)

            if success:
                used_volume += np.prod(dims)
                placed_items.append({'x': x, 'y': y, 'z': z, 'l': dims[0], 'w': dims[1], 'h': dims[2]})
                state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch = buffer.sample(batch_size)
                s_batch, a_batch, r_batch, s2_batch, d_batch = s_batch.to(device), a_batch.to(device), r_batch.to(
                    device), s2_batch.to(device), d_batch.to(device)

                q_vals = dqn(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_dqn(s2_batch).max(1)[0]
                q_target = r_batch + gamma * q_next * (1 - d_batch)
                loss = (q_vals - q_target).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if ep % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        volume_util = used_volume / (GRID_DIMS[0] * GRID_DIMS[1] * GRID_DIMS[2])
        episode_rewards.append(total_reward)
        volume_utilizations.append(volume_util)

        with open(log_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep, total_reward, volume_util, round(epsilon, 4)])

        print(f"Ep {ep} | Reward: {total_reward:.2f} | Utilization: {volume_util:.3f} | Epsilon: {epsilon:.3f}")

    torch.save(dqn.state_dict(), "model_weights/dqn_binpacking.pth")

    # Plot reward trend
    plt.plot(episode_rewards, label='Reward')
    plt.plot(volume_utilizations, label='Volume Utilization')
    plt.xlabel("Episode")
    plt.ylabel("Score / Util")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()


if __name__ == "__main__":
    dqn_train()
