from torch import nn
from transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self,
                 dim: int,
                 dim_hidden: int,
                 p_drop: float,
                 num_blocks: int,
                 vocab_size: int,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, dim_hidden, p_drop) for _ in range(num_blocks)])
        self.RMS = nn.RMSNorm(dim)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.RMS(x) @ self.embedding.weight.T