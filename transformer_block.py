from torch import nn
from mlp import MultiLayerPerceptron

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_hidden,
                 p_drop):
        super().__init__()
        self.mlp = MultiLayerPerceptron(dim, dim_hidden)
        self.RMS = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.dropout(self.mlp(self.RMS(x)))