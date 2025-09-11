#!/usr/bin/env python3
# Usage:
#   python clean_input.py raw.txt input.txt
# or to clean in-place:
#   python clean_input.py input.txt input.txt

import re, sys, unicodedata

if len(sys.argv) != 3:
    print("usage: python clean_input.py <in> <out>")
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]
text = open(src, "r", encoding="utf-8").read()

# 1) Unicode normalize & remove weird spacing
text = unicodedata.normalize("NFKC", text)
# Thin spaces, non-breaking spaces, etc. -> regular space
text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)

# 2) Drop Wikipedia section headers like "=== Title ==="
text = re.sub(r"^\s*=+\s.*?=+\s*$", "", text, flags=re.MULTILINE)

# 3) Remove MathJax/TeX-ish blocks and inline display fragments
#    e.g. {\displaystyle ...}, \( ... \), \[ ... \]
text = re.sub(r"\{\s*\\?displaystyle\b.*?\}", "", text, flags=re.DOTALL)
text = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", text)
# kill leftover TeX-like braces clutter
text = re.sub(r"\$[^$]*\$", " ", text)

# 4) Collapse bracketed citation leftovers like [1], [12]
text = re.sub(r"\[\d{1,3}\]", "", text)

# 5) Normalize newlines & spaces; keep paragraph breaks
text = text.replace("\r\n", "\n").replace("\r", "\n")
# collapse 3+ newlines -> 2 (paragraph break)
text = re.sub(r"\n{3,}", "\n\n", text)
# collapse runs of spaces
text = re.sub(r"[ \t]{2,}", " ", text)
# trim spaces around newlines
text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

# 6) Filter paragraphs to reasonable length (optional but helpful)
paras = []
for p in text.split("\n\n"):
    p = p.strip()
    if 40 <= len(p) <= 800:
        paras.append(p)

# Fallback if we filtered everything
if not paras:
    paras = [line.strip() for line in text.split("\n") if len(line.strip()) > 0]

# 7) Write out
with open(dst, "w", encoding="utf-8") as f:
    for p in paras:
        f.write(p + "\n")

print(f"[cleaned] wrote {len(paras)} paragraphs to {dst}")
