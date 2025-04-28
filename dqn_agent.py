import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def select_action(self, state_tensor, feasible_actions):
        """
        Given the current state and feasible actions, select the best action.
        Args:
            state_tensor (torch.Tensor): The input state tensor.
            feasible_actions (list): List of available actions.
        Returns:
            chosen_action (tuple): Selected action (x, y, z, orientation).
        """
        with torch.no_grad():
            q_values = self(state_tensor.unsqueeze(0)).squeeze()
            action_idx = torch.argmax(q_values[:len(feasible_actions)]).item()
        return feasible_actions[action_idx]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

