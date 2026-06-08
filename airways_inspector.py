# -*- coding: utf-8 -*-
"""
airways_inspector.py
====================
Interactive 3D inspector for Human Airways snapshots.

Inspired by hull_inspector.py (Prof. Marco E. Biancolini, Tor Vergata).
Adapted for the Human Airways Digital Twin dataset (DiTiDE / EuroHPC).

What this script does:
  - Loads the baseline mesh (points.bin) and all geometry/pressure snapshots.
  - Opens a PyVista window showing the 3D point cloud of the airways.
  - Allows real-time switching between:
      D  → Displacement colour map (mm) — how much each patient differs
           from the baseline geometry (FEA result)
      P  → Pressure colour map (Pa) — static air pressure on the walls
           (CFD result)
  - Arrow keys ← → navigate across all 100 patient snapshots.

Controls
--------
  ← →    navigate snapshots (patients)
  D      switch to Displacement view
  P      switch to Pressure view
  Mouse  rotate / zoom
  Q      quit

Usage
-----
    python airways_inspector.py

Dependencies
------------
    pip install numpy pyvista
"""

import numpy as np
import struct
import json
from pathlib import Path
import pyvista as pv

# ── CONFIG ────────────────────────────────────────────────────────────────────
POINTS_BIN    = "points.bin"
SETTINGS_JSON = "settings.json"
GEOM_DIR      = Path("snapshots")
PRES_DIR      = Path("Pressure/snapshots_pressure")


# ── I/O HELPERS ───────────────────────────────────────────────────────────────
def read_bin_vector(path):
    """Read geometry .bin → (N, 3) float64 array (XYZ coordinates)."""
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64).reshape(-1, 3)


def read_bin_scalar(path):
    """Read pressure .bin → (N,) float64 array (one value per point)."""
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("[INFO] Loading base mesh...")
pts = read_bin_vector(POINTS_BIN)

with open(SETTINGS_JSON) as f:
    ns = json.load(f)["namedSelections"]

geom_files = sorted(GEOM_DIR.glob("snapshot*.bin"),
                    key=lambda p: int(p.stem.replace("snapshot", "")))
pres_files = sorted(PRES_DIR.glob("snapshot*.bin"),
                    key=lambda p: int(p.stem.replace("snapshot", "")))

print(f"[INFO] Geometry snapshots : {len(geom_files)}")
print(f"[INFO] Pressure snapshots : {len(pres_files)}")
print("[INFO] Loading all snapshots (~30 sec)...")

geom_snaps = [read_bin_vector(p) for p in geom_files]
pres_snaps = [read_bin_scalar(p) for p in pres_files]
print("[INFO] Done!")

# ── VIEWER ────────────────────────────────────────────────────────────────────
mode    = ["displacement"]   # current colour mode
current = [0]                # current snapshot index


def get_points(i):
    """Return 3D coordinates for snapshot i."""
    if mode[0] == "displacement":
        return geom_snaps[i].astype(np.float32)
    return pts.astype(np.float32)


def get_scalars(i):
    """Return scalar field, label, and colour limits for snapshot i."""
    if mode[0] == "displacement":
        mag = np.linalg.norm(geom_snaps[i] - pts, axis=1)
        return (mag * 1e3).astype(np.float32), "Displacement (mm)", [0, 25], "plasma"
    else:
        return (pres_snaps[i].astype(np.float32),
                "Pressure (Pa)", [-0.35, 0.41], "RdBu_r")


def update(i):
    scalars, label, clim, cmap = get_scalars(i)
    cloud = pv.PolyData(get_points(i))
    cloud[label] = scalars
    pl.add_mesh(cloud,
                scalars=label,
                cmap=cmap,
                point_size=2,
                render_points_as_spheres=True,
                clim=clim,
                scalar_bar_args={"title": label, "fmt": "%.2f"},
                name="cloud")
    pl.actors["info"].SetInput(
        f"Human Airways  |  snapshot {i+1}/{len(geom_snaps)}  "
        f"|  mode: {mode[0]}  |  [D] displacement  [P] pressure  "
        f"|  ← → navigate  |  Q quit"
    )
    pl.render()


# Initial cloud
scalars0, label0, clim0, cmap0 = get_scalars(0)
cloud0 = pv.PolyData(get_points(0))
cloud0[label0] = scalars0

pl = pv.Plotter(title="Human Airways — Displacement & Pressure Inspector")
pl.add_mesh(cloud0,
            scalars=label0,
            cmap=cmap0,
            point_size=2,
            render_points_as_spheres=True,
            clim=clim0,
            scalar_bar_args={"title": label0, "fmt": "%.2f"},
            name="cloud")
pl.add_text(
    f"Human Airways  |  snapshot 1/{len(geom_snaps)}  "
    f"|  mode: displacement  |  [D] displacement  [P] pressure  "
    f"|  ← → navigate  |  Q quit",
    position="upper_left", font_size=8, color="white", name="info"
)


def next_snap():
    current[0] = min(current[0] + 1, len(geom_snaps) - 1)
    update(current[0])


def prev_snap():
    current[0] = max(current[0] - 1, 0)
    update(current[0])


def switch_displacement():
    mode[0] = "displacement"
    update(current[0])
    print(f"[INFO] Mode → displacement  (snapshot {current[0]+1})")


def switch_pressure():
    mode[0] = "pressure"
    update(current[0])
    print(f"[INFO] Mode → pressure  (snapshot {current[0]+1})")


pl.add_key_event("Right", next_snap)
pl.add_key_event("Left",  prev_snap)
pl.add_key_event("d",     switch_displacement)
pl.add_key_event("p",     switch_pressure)

print()
print("3D window open!")
print("  ← →   navigate snapshots")
print("  D      displacement view")
print("  P      pressure view")
print("  Mouse  rotate / zoom")
print("  Q      quit")

pl.show()