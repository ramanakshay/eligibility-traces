import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim

        self.network = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, out_dim),
                                        nn.Tanh())


    def forward(self, obs):
        # NOTE: outputs in batches only
        obs = obs.reshape([-1, self.in_dim])
        if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)

        output = self.network(obs)
        return output

class LargeMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim

        self.network = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, 2*hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(2*hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, out_dim),
                                        nn.Tanh())

    def forward(self, obs):
        # NOTE: outputs in batches only
        obs = obs.reshape([-1, self.in_dim])
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        output = self.network(obs)
        return output