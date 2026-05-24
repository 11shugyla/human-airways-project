import numpy as np
import pandas as pd
import struct
import json
import os
from pathlib import Path

def read_bin(path):
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64).reshape(-1, 3)

# Загружаем базовую сетку и настройки
print("Читаю points.bin...")
pts = read_bin("points.bin")

with open("settings.json") as f:
    ns = json.load(f)["namedSelections"]

# Ключевые анатомические регионы
key_regions = [
    "epiglotis", "glotis", "larynx", "mouth_region",
    "upper_trachea_bottom", "upper_trachea_middle", "upper_trachea_top",
    "gl", "gr", "glr", "grr"
]

# Читаем все снапшоты
snap_files = sorted(Path("snapshots").glob("snapshot*.bin"))
print(f"Найдено снапшотов: {len(snap_files)}")

results = []
for snap_path in snap_files:
    snap = read_bin(snap_path)
    mag = np.linalg.norm(snap - pts, axis=1)
    
    row = {"snapshot": snap_path.stem}
    for region in key_regions:
        start, _, end = ns[region]
        row[region + "_max"] = round(mag[start:end+1].max() * 1e3, 3)
        row[region + "_mean"] = round(mag[start:end+1].mean() * 1e3, 3)
    row["global_max"] = round(mag.max() * 1e3, 3)
    row["global_mean"] = round(mag.mean() * 1e3, 3)
    results.append(row)
    print(f"  {snap_path.stem}: global_max={row['global_max']} мм")

# Сохраняем результат
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("\nГотово! Сохранено в results.csv")
print(df[["snapshot", "global_max", "global_mean", "glotis_max", "larynx_max"]].to_string(index=False))