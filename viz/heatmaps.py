"""
heatmaps.py — Per-grasp HOLD-AVERAGE heatmaps, s1 | s2 side by side.

The representative map of a grasp is the HOLD-AVERAGE (project convention):
the mean of all frames whose taxel SUM is >= 0.5 x that grasp's peak sum.

For every grasp in a run folder this makes ONE figure:
    left = s1, right = s2, shared color scale, peak sum in each title,
and saves it to  <run_dir>/Heatmaps/heatmap_<tag>.png.

Pure post-processing (PyCharm side). Uses a standalone Figure + Agg canvas —
it NEVER touches the global matplotlib backend (GUI safety rule).

Usage:
  from heatmaps import plot_run
  pngs = plot_run(run_dir)
or standalone:
  python3 heatmaps.py <run_dir>
"""

import os, sys, glob
import numpy as np
import pandas as pd

HOLD_FRAC = 0.5      # frames with sum >= 0.5*peak define the hold window
MIRROR_S2 = True     # show s2 mirrored L-R (the two pads face each other)


def hold_average(csv_path):
    """Return (map_7x4, n_hold_frames, peak_sum) for one tactile CSV."""
    df = pd.read_csv(csv_path)
    pred = [c for c in df.columns if c.startswith("pred_")]
    v = df[pred].to_numpy()
    s = v.sum(1)
    peak = float(s.max()) if len(s) else 0.0
    if peak <= 0:
        return np.zeros((7, 4)), 0, 0.0
    mask = s >= HOLD_FRAC * peak
    return v[mask].mean(0).reshape(7, 4), int(mask.sum()), peak


def plot_run(run_dir):
    """One heatmap figure per grasp (s1 left | s2 right) -> Heatmaps/*.png.
    Returns the list of saved PNG paths (empty if no tactile files)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s1_files = sorted(glob.glob(os.path.join(run_dir, "*_pt*_s1_tactile_maps.csv")))
    if not s1_files:
        return []
    out_dir = os.path.join(run_dir, "Heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    made = []
    for s1 in s1_files:
        tag = os.path.basename(s1).split("_s1_")[0]        # e.g. gui_pt00
        s2 = s1.replace("_s1_", "_s2_")

        panels = []                                        # (name, map, n, peak, ok)
        for name, path in [("s1", s1), ("s2", s2)]:
            if os.path.exists(path):
                m, nfr, peak = hold_average(path)
                panels.append((name, m, nfr, peak, True))
            else:
                panels.append((name, np.zeros((7, 4)), 0, 0.0, False))

        vmax = max(p[1].max() for p in panels)
        if vmax <= 0:
            vmax = 1.0

        fig = Figure(figsize=(7.6, 4.0))
        FigureCanvasAgg(fig)
        axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
        im = None
        for ax, (name, m, nfr, peak, ok) in zip(axes, panels):
            # s2 faces the opposite way from s1, so show it MIRRORED L-R
            # (facing-pad view) — display only; the stored data is untouched.
            disp = m[:, ::-1] if (name == "s2" and MIRROR_S2) else m
            im = ax.imshow(disp, cmap="jet", aspect="auto", vmin=0.0, vmax=vmax)
            mtag = " [mirrored]" if (name == "s2" and MIRROR_S2) else ""
            ttl = (f"{name}{mtag} — hold-avg of {nfr} frames\npeak sum {peak:.0f}"
                   if ok else f"{name} — FILE MISSING")
            ax.set_title(ttl, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.set_label("pressure (a.u.)", fontsize=8)
        fig.suptitle(f"Grasp {tag} — hold-average heatmaps (s1 | s2)", fontsize=11)

        png = os.path.join(out_dir, f"heatmap_{tag}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        made.append(png)
        print(f"saved {png}")
    return made


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Paper3_Simulation/Data/gui_run")
    if not plot_run(rd):
        print(f"no *_s1_tactile_maps.csv found in {rd}")
