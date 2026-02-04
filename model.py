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
    
    def ttt_inner_step(self, mini_batch, dynamic, static):
        # mini_batch: (B x b)

        all_params = {**dynamic, **static}
        logits = functional_call(self.transformer, all_params, (mini_batch[:,:-1],)) # wrap in tensor

        # logits: (B*b, V), mini_batch: (B*b)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), mini_batch[:,1:].reshape(-1))

        # create graph so we can grad of grad!
        grads = grad(loss, list(dynamic.values()), create_graph=True)
        updated_dynamic = {k : (v - self.lr * g) for (k, v), g in zip(dynamic.items(), grads)}

        return updated_dynamic, logits, loss

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
        V = self.cfg.vocab_size
        B, T = x.shape
        # inner loop:
        for i in range(0, T-1, b): # indexing safeguard
            # chunk of b + 1?
            mini_batch = x[:, i:i+b+1]

            # inner loop step
            dynamic, logit_chunk, loss = self.ttt_inner_step(mini_batch, dynamic, static)

            # concat logits
            logit_chunks.append(logit_chunk)

        logits = torch.cat(logit_chunks, dim=1)
        assert logits.shape == [B, (T-1)//b, V]

        return logits