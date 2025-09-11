#!/usr/bin/env python3
# Generate/complete Shakespeare-style text from a prompt using the SP-trained TinyGPT.
# Usage:
#   python generate_shakespeare_sp.py "O Romeo, Romeo!"
#   python generate_shakespeare_sp.py --ckpt shakespeare_gpt_sp.pt --max_new 400 --temp 0.9 --top_k 50 "To be, or not to be"

import argparse, os, torch
from prospero import TinyGPT
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="", help="initial text to continue (optional)")
    ap.add_argument("--ckpt", default="shakespeare_gpt_sp.pt", help="path to checkpoint")
    ap.add_argument("--max_new", type=int, default=400, help="tokens to generate")
    ap.add_argument("--temp", type=float, default=0.9, help="sampling temperature")
    ap.add_argument("--top_k", type=int, default=50, help="top-k sampling (set 0 to disable)")
    args = ap.parse_args()

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    tok = ckpt.get("tokenizer", {})
    assert tok.get("type") == "sentencepiece", f"Expected SentencePiece tokenizer in ckpt, got: {tok}"

    sp = spm.SentencePieceProcessor(model_file=tok["model_file"])
    def encode(s): return sp.encode(s, out_type=int)
    def decode(ids): return sp.decode(ids)
    vocab_size = sp.get_piece_size()

    cfg = ckpt["config"]
    model = TinyGPT(vocab_size, cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["block_size"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    # Build input ids (empty prompt okay → it will free-generate)
    ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k > 0 else None

    out = model.generate(ids, max_new_tokens=args.max_new, temperature=args.temp, top_k=top_k)[0].tolist()
    gen = decode(out[len(encode(args.prompt)):])

    # Print: prompt + completion
    if args.prompt:
        print(args.prompt, end="")
    print(gen)

if __name__ == "__main__":
    main()
