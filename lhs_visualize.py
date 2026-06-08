# -*- coding: utf-8 -*-
"""
lhs_visualize.py
================
3D visualisation of Latin Hypercube Sampling in parameter space.

Inspired by LHS_PyVista.py (Prof. Marco E. Biancolini, Tor Vergata).
Adapted for the Human Airways Digital Twin dataset.

Shows 100 patient anatomies as a 3D point cloud in the space of
the three most influential DOE parameters (from correlation analysis):
  - A_glotis   : glottis cross-sectional area (mm²)
  - l_trachea  : trachea length (mm)
  - r_curvature: trachea curvature radius (mm)

Controls
--------
  R  → regenerate a new LHS cloud (demo mode)
  Q  → quit

Usage
-----
    python lhs_visualize.py

Dependencies
------------
    pip install numpy scipy pyvista pandas
"""

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.stats import qmc

# ── CONFIG ────────────────────────────────────────────────────────────────────
N        = 100
SEED     = 42
DOE_CSV  = "doe.csv"

# Parameter names and ranges (from doe.csv min/max)
PARAM_NAMES = ["A_glotis (mm²)", "l_trachea (mm)", "r_curvature (mm)"]
LOWS        = np.array([86.0,  80.0, 30.0])
HIGHS       = np.array([230.0, 150.0, 70.0])


# ── LHS GENERATOR ─────────────────────────────────────────────────────────────
def generate_lhs(n=N, seed=SEED):
    """Generate LHS sample in physical parameter space."""
    sampler = qmc.LatinHypercube(d=3, seed=seed, optimization="random-cd")
    U = sampler.random(n=n)
    return LOWS + (HIGHS - LOWS) * U


# ── LOAD ACTUAL DOE ────────────────────────────────────────────────────────────
def load_doe():
    try:
        doe = pd.read_csv(DOE_CSV)
        return doe[["A_glotis", "l_trachea", "r_curvature"]].values
    except FileNotFoundError:
        print(f"[WARN] {DOE_CSV} not found — showing synthetic LHS only")
        return None


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    doe_pts = load_doe()

    # Build PyVista plotter
    pl = pv.Plotter(window_size=(1100, 800),
                    title="Human Airways — LHS Parameter Space")
    pl.set_background("white")

    # Bounding box (parameter space)
    box = pv.Box(bounds=(LOWS[0], HIGHS[0],
                          LOWS[1], HIGHS[1],
                          LOWS[2], HIGHS[2])).extract_all_edges()
    pl.add_mesh(box, line_width=2, color="gray")

    # Actual DOE points (from doe.csv)
    if doe_pts is not None:
        cloud_doe = pv.PolyData(doe_pts.astype(np.float32))
        cloud_doe["patient"] = np.arange(N)
        pl.add_mesh(cloud_doe,
                    scalars="patient",
                    cmap="plasma",
                    render_points_as_spheres=True,
                    point_size=14,
                    scalar_bar_args={"title": "Patient index"},
                    label="Actual DOE patients")
        title_text = (f"Human Airways — 100 Patients (DOE)\n"
                      f"X: {PARAM_NAMES[0]}  "
                      f"Y: {PARAM_NAMES[1]}  "
                      f"Z: {PARAM_NAMES[2]}\n"
                      f"Press R to show synthetic LHS  |  Q to quit")
    else:
        X = generate_lhs()
        cloud = pv.PolyData(X.astype(np.float32))
        cloud["idx"] = np.arange(N)
        pl.add_mesh(cloud,
                    scalars="idx",
                    cmap="plasma",
                    render_points_as_spheres=True,
                    point_size=14)
        title_text = (f"LHS — {N} synthetic samples\n"
                      f"X: {PARAM_NAMES[0]}  "
                      f"Y: {PARAM_NAMES[1]}  "
                      f"Z: {PARAM_NAMES[2]}\n"
                      f"Press R to regenerate  |  Q to quit")

    pl.add_text(title_text, position="upper_left", font_size=9, color="black")

    # Key R: regenerate synthetic LHS (demo)
    synthetic_cloud = [None]

    def regenerate():
        if synthetic_cloud[0] is not None:
            pl.remove_actor(synthetic_cloud[0])
        seed = int(np.random.SeedSequence().entropy) % 10000
        X    = generate_lhs(seed=seed)
        c    = pv.PolyData(X.astype(np.float32))
        c["idx"] = np.arange(N)
        synthetic_cloud[0] = pl.add_mesh(
            c, scalars="idx", cmap="viridis",
            render_points_as_spheres=True, point_size=10,
        )
        pl.render()
        print(f"[INFO] New LHS generated (seed={seed})")

    pl.add_key_event("r", regenerate)

    # Axes labels
    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    main()