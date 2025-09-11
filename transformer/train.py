#!/usr/bin/env python3
# Train TinyGPT on tiny Shakespeare using a compact SentencePiece vocab (8k).
# pip install sentencepiece torch

import os, math, requests, torch, torch.nn as nn
from prospero import TinyGPT
import sentencepiece as spm
from tqdm import tqdm


# ---------------- config ----------------
batch_size   = 64
block_size   = 256
n_embd       = 384
n_head       = 6
n_layer      = 6
dropout      = 0.20
max_iters    = 15000
eval_interval= 4000
learning_rate= 3e-4
eval_batches = 100

sp_model     = "sp.model"   # trained on input.txt
sp_vocab     = 8000         # try 4000–8000
ckpt_path    = "shakespeare_gpt_sp.pt"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)

# -------------- data --------------------
# if not os.path.exists("input.txt"):
#     print("Downloading tiny Shakespeare...")
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     txt = requests.get(url, timeout=30).text
#     with open("input.txt", "w", encoding="utf-8") as f:
#         f.write(txt)
# with open("input.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# -------------- tokenizer (SentencePiece) --------------

if not os.path.exists(sp_model):
    print(f"Training SentencePiece tokenizer ({sp_vocab} unigram, byte_fallback=True)...")
    spm.SentencePieceTrainer.Train(
        input="input.txt",
        model_prefix=os.path.splitext(sp_model)[0],
        vocab_size=sp_vocab,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1
    )

sp = spm.SentencePieceProcessor(model_file=sp_model)
def encode(s: str): return sp.encode(s, out_type=int)
def decode(ids):    return sp.decode(ids)
vocab_size = sp.get_piece_size()

ids = torch.tensor(encode(raw_text), dtype=torch.long)
n = int(0.9 * len(ids))
train_data, val_data = ids[:n], ids[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -------------- model & optim --------------
model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.05)

warmup_steps = 400
final_lr = 1e-4
def lr_lambda(it):
    if it < warmup_steps: return (it + 1) / warmup_steps
    progress = (it - warmup_steps) / max(1, (max_iters - warmup_steps))
    return (final_lr/learning_rate) + 0.5 * (1 - math.cos(math.pi * progress)) * (1 - (final_lr/learning_rate))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train","val"]:
        losses = []
        for _ in range(eval_batches):
            x,y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses)
    model.train()
    return out

print(f"Device: {device}, vocab_size={vocab_size}, total_tokens={len(ids):,}")

progress = tqdm(range(1, max_iters + 1), ncols=100, desc="training", unit="step")
try:
    for step in progress:
        x, y = get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        # live stats on the bar
        lr_now = scheduler.get_last_lr()[0]
        progress.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_now:.2e}")

        if step % eval_interval == 0:
            # use tqdm.write so it doesn't break the bar line
            est = estimate_loss()
            tqdm.write(f"step {step:5d} | train_loss {est['train']:.3f} | val_loss {est['val']:.3f}")

            # quick sample & save
            start = torch.zeros((1,1), dtype=torch.long, device=device)
            sample_ids = model.generate(start, 300, temperature=0.9, top_k=50)[0].tolist()
            tqdm.write("\n---- SAMPLE ----\n" + decode(sample_ids) + "\n")

            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size, "n_embd": n_embd, "n_head": n_head,
                    "n_layer": n_layer, "block_size": block_size, "dropout": dropout
                },
                "tokenizer": {"type": "sentencepiece", "model_file": sp_model},
            }, ckpt_path)
            tqdm.write(f"[saved] {ckpt_path}")
finally:
    progress.close()

# final save
torch.save({
    "state_dict": model.state_dict(),
    "config": {
        "vocab_size": vocab_size, "n_embd": n_embd, "n_head": n_head,
        "n_layer": n_layer, "block_size": block_size, "dropout": dropout
    },
    "tokenizer": {"type": "sentencepiece", "model_file": sp_model},
}, ckpt_path)
print(f"Done. Checkpoint at: {ckpt_path}")
