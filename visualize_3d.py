import numpy as np
import struct
import json
import pyvista as pv
from pathlib import Path

def read_bin(path):
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64).reshape(-1, 3)

print("Загружаю базовую сетку...")
pts = read_bin("points.bin")

with open("settings.json") as f:
    ns = json.load(f)["namedSelections"]

snap_files = sorted(Path("snapshots").glob("snapshot*.bin"),
                    key=lambda p: int(p.stem.replace("snapshot", "")))
print(f"Найдено снапшотов: {len(snap_files)}")

# Загружаем все снапшоты
print("Загружаю все снапшоты (подожди ~30 сек)...")
snaps = []
for p in snap_files:
    snaps.append(read_bin(p))
print("Готово!")

# Метки регионов для каждой точки
region_ids = np.zeros(len(pts), dtype=np.int32)
region_names = ["other"]
for i, (name, (start, _, end)) in enumerate(ns.items(), 1):
    region_ids[start:end+1] = i
    region_names.append(name)

# Создаём PyVista облако точек
cloud = pv.PolyData(pts.astype(np.float32))
cloud["region"] = region_ids

# Считаем деформацию для первого снапшота
def get_displacement(snap_idx):
    mag = np.linalg.norm(snaps[snap_idx] - pts, axis=1)
    return (mag * 1e3).astype(np.float32)

cloud["displacement_mm"] = get_displacement(0)

# Плоттер
pl = pv.Plotter(title="Human Airways — Displacement Viewer")
pl.add_mesh(
    cloud,
    scalars="displacement_mm",
    cmap="plasma",
    point_size=2,
    render_points_as_spheres=True,
    clim=[0, 25],
    scalar_bar_args={"title": "Деформация (мм)", "fmt": "%.1f"}
)
pl.add_text("Human Airways · snapshot 1/100", position="upper_left",
            font_size=10, color="white")

current = [0]

def next_snap():
    current[0] = min(current[0] + 1, len(snaps) - 1)
    update(current[0])

def prev_snap():
    current[0] = max(current[0] - 1, 0)
    update(current[0])

def update(i):
    cloud["displacement_mm"] = get_displacement(i)
    pl.actors["displacement_mm"] = None
    pl.add_mesh(
        cloud,
        scalars="displacement_mm",
        cmap="plasma",
        point_size=2,
        render_points_as_spheres=True,
        clim=[0, 25],
        scalar_bar_args={"title": "Деформация (мм)", "fmt": "%.1f"},
        name="cloud"
    )
    pl.textactor.SetInput(f"Human Airways · snapshot {i+1}/100")
    pl.render()

pl.add_key_event("Right", next_snap)
pl.add_key_event("Left", prev_snap)

print("\n3D окно открыто!")
print("← → стрелки на клавиатуре = переключать снапшоты")
print("Мышь = вращать / зумить")
print("Q = закрыть")

pl.show()