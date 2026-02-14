import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "notebooks" / "01_Single_Variable_Linear_Regression.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
c = nb["cells"][0]
src = c["source"]
print("Type of source:", type(src))
print("Number of items:", len(src))
print("Type of first 3 items:", [type(x) for x in src[:3]])
# Join as in the script
text = "".join(s if isinstance(s, str) else "" for s in src)
print("Joined length:", len(text))
print("Has ```:", "```" in text)
# Show around first backtick
i = text.find("`")
print("First backtick at:", i)
if i >= 0:
    print("Repr around it:", repr(text[i : i + 25]))

# List all cells
print("\nAll cells:")
for i, c in enumerate(nb["cells"]):
    src = c.get("source", [])
    text = "".join(s if isinstance(s, str) else "" for s in src)
    print(f"  {i} {c['cell_type']} lines={len(src)} chars={len(text)} has_fence={'```' in text}")

# Show start of cell 2
if len(nb["cells"]) > 2:
    t = "".join(s if isinstance(s, str) else "" for s in nb["cells"][2]["source"])
    print("\nCell 2 first 400 chars:", repr(t[:400]))

# Show cell 1 first and last lines
print("\nCell 1 (code) first 3 source lines:")
for line in nb["cells"][1]["source"][:3]:
    print(" ", repr(line[:60]))
print("Cell 1 last 3 source lines:")
for line in nb["cells"][1]["source"][-3:]:
    print(" ", repr(line[:60]))
