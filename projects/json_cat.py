import json, pathlib
root = pathlib.Path("/scratch/bwang25/app_data/mppi_heurstic")
start, end = 0, 300  # [start, end)
items = []
for i in range(start, end):
    p = root / f"actor_{i}" / f"actor_{i}.json"
    if not p.exists():
        print(f"[SKIP] {p} not found")
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    items.extend(data)
out = root / f"all_{start}_{end-1}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False)
print(f"merged {len(items)} samples into {out}")

