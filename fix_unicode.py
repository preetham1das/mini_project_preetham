import re

with open("src/train_temporal_v2.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace non-ASCII decorative characters with ASCII equivalents
content = content.replace("\u2192", "->")     # right arrow
content = content.replace("\u2500", "-")      # box drawing
content = content.replace("\u03b3", "g")      # gamma
content = content.replace("\u03b1", "a")      # alpha
content = content.replace("\u2014", "--")     # em dash
content = content.replace("\u2013", "-")      # en dash
content = content.replace("\u2192", "->")

# Remove remaining non-ASCII by replacing with closest ASCII
import unicodedata
clean = ""
for ch in content:
    if ord(ch) < 128:
        clean += ch
    else:
        try:
            norm = unicodedata.normalize("NFKD", ch)
            ascii_ch = norm.encode("ascii", "ignore").decode("ascii")
            clean += ascii_ch if ascii_ch else " "
        except Exception:
            clean += " "

with open("src/train_temporal_v2.py", "w", encoding="utf-8") as f:
    f.write(clean)

print("Done. Checking for remaining non-ASCII:")
for i, line in enumerate(clean.split("\n"), 1):
    for ch in line:
        if ord(ch) >= 128:
            print(f"  Line {i}: {repr(line)}")
            break
print("OK")
