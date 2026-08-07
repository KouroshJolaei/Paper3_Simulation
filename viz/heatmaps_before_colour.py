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

HOLD_FRAC = 0.9      # FALLBACK ONLY — hold_average() below delegates to
                     # stitching.hold_average whenever stitching imports.
                     # Kept in step with stitching.HOLD_FRAC so the fallback
                     # cannot silently disagree with the real pipeline.

# ---- tolerate the tactile-writer race (see stitching._read_tactile_csv) ----
# Berith's per-frame writer occasionally collides two writes into one line,
# giving 31 fields instead of 30. One bad frame out of ~325 is irrelevant
# here, so skip it rather than abort the whole figure.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# hold_average is imported from stitching too: keeping a second copy here
# meant heatmaps and stitches could silently disagree (this file used
# 0.5*peak as the hold threshold, stitching uses min + 0.5*range, and only
# stitching knew about baseline subtraction).
try:
    from stitching import _read_tactile_csv as _read_csv_tolerant
    from stitching import hold_average as _hold_average_shared
except Exception:                                    # standalone fallback
    _hold_average_shared = None

    def _read_csv_tolerant(path):
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except TypeError:                            # pandas < 1.3
            return pd.read_csv(path, error_bad_lines=False,
                               warn_bad_lines=False)

MIRROR_S2 = False     # show s2 mirrored L-R (the two pads face each other)


def hold_average(csv_path):
    """Return (map_7x4, n_hold_frames, peak_sum) for one tactile CSV.

    Delegates to stitching.hold_average so the heatmap you look at is
    EXACTLY the map that gets stitched — same hold window, same baseline
    handling. The local version below is only a standalone fallback."""
    if _hold_average_shared is not None:
        return _hold_average_shared(csv_path)
    df = _read_csv_tolerant(csv_path)
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
            # im = ax.imshow(disp, cmap="jet", aspect="auto", vmin=0.0, vmax=vmax)
            im = ax.imshow(disp, cmap="jet", aspect="auto", vmin=0.0, vmax=vmax, origin="lower")
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

    # the 28 numbers behind every figure above, in one CSV
    try:
        from stitching import save_hold_averages
        save_hold_averages(run_dir)
    except Exception as e:
        print(f"[maps] could not save hold_average_maps.csv ({e})")
    return made


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Paper3_Simulation/Data/gui_run")
    if not plot_run(rd):
        print(f"no *_s1_tactile_maps.csv found in {rd}")
