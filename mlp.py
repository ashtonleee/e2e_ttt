import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.gate = nn.Linear(dim, dim_hidden)
        self.silu = nn.SiLU()
        self.up = nn.Linear(dim, dim_hidden)
        self.down = nn.Linear(dim_hidden, dim)
    
    def forward(self, x):
        return self.down(self.silu(self.gate(x)) * self.up(x))