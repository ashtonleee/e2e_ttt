class Config:
    def __init__(self, dim, dim_hidden, p_drop, num_blocks, vocab_size, mini_batch_size):
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.p_drop = p_drop
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.mini_batch_size = mini_batch_size