import sys

bad = False
for path in sys.argv[1:]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        continue
    text = data.decode("utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(ord(c) > 127 for c in line):
            print(f"Non-ASCII characters found: {path}:{lineno}")
            bad = True

sys.exit(1 if bad else 0)
