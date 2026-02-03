from torch import nn
import torch
from transformer import Transformer
from utils import Config
from torch.func import functional_call
import torch.nn.functional as F
from torch.autograd import grad

class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.transformer = Transformer(cfg.dim,
                                  cfg.dim_hidden,
                                  cfg.p_drop,
                                  cfg.num_blocks,
                                  cfg.vocab_size,
                                  )
        self.cfg = cfg
        self.lr = 3e-4
    
    def forward(self, x):
        # x: (B, T)
        #   - IDs
        # out: (B, T, V)
        #   - logits

        params = dict(self.transformer.named_parameters())

        dynamic = {k: v for k, v in params.items() if '.mlp.' in k}
        static = {k: v for k, v in params.items() if k not in dynamic}
        logit_chunks = []

        b = self.cfg.mini_batch_size
        T = x.size(1)
        # inner loop:
        for i in range(0, T-1, b): # indexing safeguard
            # chunk of b + 1?
            mini_batch = x[:, i:i+b+1]

            all_params = {**dynamic, **static}
            logits = functional_call(self.transformer, all_params, (mini_batch[:,:-1],)) # wrap in tensor

            # logits: (B*b, V), mini_batch: (B*b)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), mini_batch[:,1:].reshape(-1))

            # create graph so we can grad of grad!
            grads = grad(loss, list(dynamic.values()), create_graph=True)
            dynamic = {k : (v - self.lr * g) for (k, v), g in zip(dynamic.items(), grads)}

            logit_chunks.append(logits)

        return torch.cat(logit_chunks, dim=1)

if __name__ == '__main__':
    cfg = Config(dim=16,
                 dim_hidden=32,
                 p_drop=0,
                 num_blocks=2,
                 vocab_size=50,
                 mini_batch_size=4)
    model = Model(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 17), dtype=torch.long)
    logits = model(x)
    print('logits:', logits.shape)

    S = logits.size(1)
    targets = x[:, 1:1+S] # the matching targets: (B, S)

    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    print('loss', float(loss))
