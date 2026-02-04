from model import Model
from utils import Config
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch

# TODO:
#   - test loop
#       - validate loss down
#       - lr scheduler, etc.
#       - add validate
#   - device stuff?
#   - add attn


def run(device: str,
        batch_size: int,
        context_length: int,
        mini_batch_size: int,
        num_workers: int = 0):
    dclm = load_dataset("HuggingFaceTB/dclm-edu", split="train", streaming=True)
    loader = DataLoader(dclm,
                        num_workers=num_workers,
                        batch_size=batch_size,
                        pin_memory=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # manually specify
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token


    cfg = Config(dim=16,
                 dim_hidden=32,
                 p_drop=0,
                 num_blocks=2,
                 vocab_size=tokenizer.vocab_size,
                 mini_batch_size=mini_batch_size)
    model = Model(cfg=cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()

    for batch in loader:
        b = mini_batch_size
        T = context_length

        text = batch['text']

        encoded = tokenizer(text,
                            padding = 'max_length',
                            truncation = True,
                            return_tensors = "pt", # pytorch
                            max_length = context_length + 1) # for target
        assert encoded["input_ids"].size(1) == T + 1

        tok_ids = encoded["input_ids"].to(device) # (B,T+1)
        pad_mask = encoded["attention_mask"].to(device) # (B,T+1)

        x = tok_ids[:,:-1] # (B,T)
        targets = tok_ids[:,1:] # (B,T)

        logits = model(x) # (B,S:=((T-1)//b)*b,V)

        B, S, V = logits.shape
        assert (B == tok_ids.size(0)) and (S == ((T-1)//b)*b) and (V == tokenizer.vocab_size)

        # logits: (B,S,V), targets: (B,S), loss: (B*S)
        targets = targets[:,:S] # (B,S)
        loss = F.cross_entropy(logits.reshape(-1, V),
                               targets.reshape(-1),
                               reduction="none")
        loss = loss.reshape(B, S)
        
        # manual reduce for padded
        pad_mask = pad_mask[:,1:1+S] #(B,S)
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    BATCH_SIZE = 256
    CONTEXT_LENGTH = 128
    MINI_BATCH_SIZE = 1

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    run(device=device,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
        mini_batch_size = MINI_BATCH_SIZE)