#!/usr/bin/env python3
# Build input.txt from Wikipedia math & science pages, with robust logging.
# Usage:
#   python make_wiki_input.py
#   python make_wiki_input.py --out data/input.txt --min 30 --max 900 --titles topics.txt

import argparse, os, re, time, requests
from pathlib import Path

DEFAULT_TITLES = [
    "Calculus","Probability","Statistics","Optimization",
    "Number theory","Group theory","Combinatorics","Graph theory","Topology",
    "Partial differential equation","Dynamical system",
    "Classical mechanics","Electromagnetism","Thermodynamics",
    "Quantum mechanics","Special relativity","General relativity","Statistical mechanics",
    "Algorithm","Data structure","Machine learning","Neural network",
    "Chemistry","Organic chemistry","Biochemistry","Genetics","Cell biology","Evolution"
]

API_TMPL = "https://{lang}.wikipedia.org/w/api.php"
UA = "TinyGPT-Dataset-Builder/1.0 (contact: you@example.com)"  # put your contact if you like

def fetch_extract(title, lang="en", retries=3):
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "redirects": 1, "titles": title, "format": "json", "formatversion": "2"
    }
    url = API_TMPL.format(lang=lang)
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", [])
            if not pages or "missing" in pages[0]:
                return ""
            return pages[0].get("extract", "") or ""
        except Exception as e:
            if i == retries - 1:
                print(f"[warn] {title}: {e}")
            time.sleep(0.7 * (i + 1))
    return ""

def normalize_text(txt: str) -> str:
    txt = re.sub(r"\r\n?", "\n", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    # Drop “See also / References / External links / Further reading”
    txt = re.split(r"\n==\s*(See also|References|External links|Further reading)\s*==\n", txt, 1)[0]
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def split_paragraphs(txt: str, min_len: int, max_len: int):
    for para in txt.split("\n\n"):
        p = para.strip()
        if min_len <= len(p) <= max_len:
            yield p

def load_titles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", help="File with one Wikipedia title per line")
    ap.add_argument("--out", default="input.txt")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--min", type=int, default=50)
    ap.add_argument("--max", type=int, default=800)
    args = ap.parse_args()

    titles = load_titles(args.titles) if args.titles else DEFAULT_TITLES
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    print(f"[info] Writing to: {out_path.resolve()}")
    with open(out_path, "w", encoding="utf-8") as out:
        for t in titles:
            print(f"[fetch] {t} …", end="", flush=True)
            txt = fetch_extract(t, lang=args.lang)
            if not txt:
                print(" (no content)")
                continue
            txt = normalize_text(txt)
            n_before = kept
            for p in split_paragraphs(txt, args.min, args.max):
                out.write(p + "\n")
                kept += 1
            print(f" kept {kept - n_before} paras")

    # If we ended up with nothing, relax and retry once.
    if kept == 0:
        print("[warn] No paragraphs kept with current thresholds; relaxing to 10–1200 and retrying core topics…")
        core = titles[:10] if len(titles) > 10 else titles
        with open(out_path, "w", encoding="utf-8") as out:
            for t in core:
                txt = fetch_extract(t, lang=args.lang)
                txt = normalize_text(txt)
                for p in split_paragraphs(txt, 10, 1200):
                    out.write(p + "\n")
                    kept += 1

    # Absolute last resort: write a tiny seed so your pipeline doesn’t break.
    if kept == 0:
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("This is a small modern-English seed paragraph about calculus and physics.\n")
        kept = 1
        print("[info] Wrote fallback seed.")

    print(f"[done] {kept} paragraph(s) written to {out_path.resolve()}")

if __name__ == "__main__":
    main()
