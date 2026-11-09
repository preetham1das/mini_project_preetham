# Read the file in UTF-8
with open("src/train_temporal_v2.py", "r", encoding="utf-8") as f:
    content = f.read()

# Encode to ASCII, replacing any non-ASCII with ?
ascii_content = content.encode("ascii", errors="replace").decode("ascii")

# Write back as UTF-8 (but now all chars are ASCII)
with open("src/train_temporal_v2.py", "w", encoding="utf-8") as f:
    f.write(ascii_content)

# Verify
bad = [i+1 for i, ch in enumerate(ascii_content) if ord(ch) > 127]
print(f"Non-ASCII chars remaining: {len(bad)}")
if bad:
    print(f"At positions: {bad[:10]}")
else:
    print("File is clean ASCII.")
