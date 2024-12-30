import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim
        self.network = nn.Sequential(
           nn.Linear(in_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, inp):
        inp = inp.reshape([-1, self.in_dim])
        if isinstance(inp, np.ndarray):
            inp = torch.tensor(inp, dtype=torch.float)
        out = self.network(inp)
        return out