import unicodedata

for fname in ["src/train_temporal_v2.py", "src/evaluate_temporal_v2.py"]:
    with open(fname, "r", encoding="utf-8") as f:
        content = f.read()
    clean = ""
    for ch in content:
        if ord(ch) < 128:
            clean += ch
        else:
            norm = unicodedata.normalize("NFKD", ch)
            ascii_ch = norm.encode("ascii", "ignore").decode("ascii")
            clean += ascii_ch if ascii_ch else " "
    with open(fname, "w", encoding="utf-8") as f:
        f.write(clean)
    print(f"Cleaned: {fname}")
