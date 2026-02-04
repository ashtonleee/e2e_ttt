from model import Model
from utils import Config
import torch

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