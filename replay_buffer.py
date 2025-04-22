import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            lambda x: torch.tensor(x, dtype=torch.float32), zip(*samples)
        )
        return state, action.long(), reward, next_state, done

    def __len__(self):
        return len(self.buffer)