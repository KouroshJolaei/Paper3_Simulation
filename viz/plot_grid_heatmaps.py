"""
plot_grid_heatmaps.py — Plot the tactile heatmaps from a grid run.

Reads all the *_tactile_maps.csv files from a grid run folder and lays out
each grid point's contact heatmap in its (row, col) position, so you can SEE
how the contact shifts as the gripper scans across the cylinder.

THIS RUNS IN PYCHARM (normal Python), NOT in Isaac Sim.
You need: numpy, pandas, matplotlib   (pip install numpy pandas matplotlib)

USAGE
-----
Option A — point it at a specific run folder:
    python plot_grid_heatmaps.py /home/kourosh/Paper3_Simulation/Data/grid_20260626_120000

Option B — no argument: it auto-picks the NEWEST grid_* folder in Data/.

The tactile map is 28 values (pred_0..pred_27) = the TSF-85 sensor's 7x4 taxels.
For each grid point we take the PEAK contact frame (the frame with the highest
total pressure) and show it as a 7x4 heatmap.

LIVE MODE
---------
Set LIVE_PLOT = True to watch heatmaps appear as CSVs land DURING a run.
Set LIVE_PLOT = False (default) to plot once after the run finishes.
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT  = "/home/kourosh/Paper3_Simulation/Data"
SENSOR     = "s1"          # which sensor to plot: "s1" (right) or "s2" (left)
TAXEL_ROWS = 7             # TSF-85 taxel grid is 7 x 4
TAXEL_COLS = 4
LIVE_PLOT  = False         # False = plot once after run; True = live during run
LIVE_REFRESH_SEC = 3.0     # how often to refresh in live mode

# ============================================================
# Helpers
# ============================================================
def find_latest_grid_dir():
    """Return the newest Data/grid_* folder."""
    dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "grid_*")))
    if not dirs:
        return None
    return dirs[-1]

def peak_tactile_frame(csv_path):
    """Load a *_tactile_maps.csv and return ONE 28-value tactile map by
    AVERAGING the frames in the stable 'hold' window (high-contact frames),
    instead of taking a single peak frame. This is cleaner and less noisy."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  could not read {csv_path}: {e}")
        return None
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols or len(df) == 0:
        return None
    vals = df[pred_cols].to_numpy()
    totals = vals.sum(axis=1)

    # The hold window = frames where contact is real (not the open-gripper
    # baseline at start/end). We threshold at 50% of the peak total pressure,
    # which cleanly separates the ~12000 contact frames from the ~250 baseline.
    peak = totals.max()
    in_contact = totals >= 0.5 * peak
    if in_contact.sum() == 0:
        # fallback: no clear contact, just use the single peak frame
        return vals[int(np.argmax(totals))]
    # average the 28 values over all in-contact frames
    return vals[in_contact].mean(axis=0)

def parse_row_col(filename):
    """Extract (row, col) from a name like grid_r01_c02_s1_tactile_maps.csv."""
    base = os.path.basename(filename)
    r = c = None
    for token in base.split("_"):
        if token.startswith("r") and token[1:].isdigit():
            r = int(token[1:])
        if token.startswith("c") and token[1:].isdigit():
            c = int(token[1:])
    return r, c

def collect_grid(run_dir):
    """Find all tactile_maps CSVs for the chosen sensor, return dict
    {(row,col): 28-vector} plus the max row/col seen."""
    pattern = os.path.join(run_dir, f"grid_*_{SENSOR}_tactile_maps.csv")
    files = sorted(glob.glob(pattern))
    grid = {}
    max_r = max_c = 0
    for f in files:
        r, c = parse_row_col(f)
        if r is None or c is None:
            continue
        vec = peak_tactile_frame(f)
        if vec is None:
            continue
        grid[(r, c)] = vec
        max_r = max(max_r, r)
        max_c = max(max_c, c)
    return grid, max_r, max_c

def draw(run_dir, fig=None, axes=None):
    """Draw the whole grid of heatmaps. Returns (fig, axes)."""
    grid, max_r, max_c = collect_grid(run_dir)
    if not grid:
        print(f"No tactile data found yet in {run_dir} for sensor {SENSOR}.")
        return fig, axes

    nrows, ncols = max_r + 1, max_c + 1

    # find a common color scale across all points
    allvals = np.concatenate([v for v in grid.values()])
    vmin, vmax = float(allvals.min()), float(allvals.max())

    if fig is None:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.2 * ncols, 2.2 * nrows),
                                 squeeze=False)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            if (r, c) in grid:
                img = grid[(r, c)].reshape(TAXEL_ROWS, TAXEL_COLS)
                ax.imshow(img, vmin=vmin, vmax=vmax, cmap="jet", aspect="auto")
                ax.set_title(f"r{r} c{c}", fontsize=9)
            else:
                ax.set_title(f"r{r} c{c} (none)", fontsize=8, color="gray")

    fig.suptitle(f"Grid tactile heatmaps — sensor {SENSOR}\n{os.path.basename(run_dir)}",
                 fontsize=11)
    fig.tight_layout()
    return fig, axes

# ============================================================
# Main
# ============================================================
def main():
    # pick the run dir: command-line arg, or newest grid_* folder
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_grid_dir()
    if not run_dir or not os.path.isdir(run_dir):
        print(f"No grid run folder found. Looked in {DATA_ROOT}/grid_*")
        print("Pass a folder explicitly: python plot_grid_heatmaps.py <folder>")
        return

    print(f"Plotting grid run: {run_dir}  (sensor {SENSOR})")

    if not LIVE_PLOT:
        fig, axes = draw(run_dir)
        if fig is not None:
            plt.show()
        return

    # LIVE MODE: refresh on a timer until you close the window
    plt.ion()
    fig = axes = None
    try:
        while True:
            fig, axes = draw(run_dir, fig, axes)
            if fig is not None:
                plt.pause(0.5)
            time.sleep(LIVE_REFRESH_SEC)
    except KeyboardInterrupt:
        print("Stopped live plotting.")
        plt.ioff()
        if fig is not None:
            plt.show()

if __name__ == "__main__":
    main()
