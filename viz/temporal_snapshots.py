"""
temporal_snapshots.py — Extract the 4 temporal tactile snapshots per grasp.

Matches Roberge/Duchaine 2021 ("Tactile-Based Object Recognition Using a
Grasp-Centric Exploration"): during the squeeze, capture 4 static pressure maps
at the moments when the taxel SUM reaches:
    #1  5% of max   (first light contact)
    #2 50% of max   (mid compression)
    #3 95% of max   (near full compression)
    #4  3 s after squeeze start   (post-squeeze creep: rigid vs soft object)

Pure post-processing of the tactile CSVs we already record (each grasp already
stores the FULL close->hold->open curve, ~175 frames). We just select the 4
frames at the right pressure levels — no new collection needed.

NOTE (honest): snapshot #4 only means something if the gripper HOLDS CLOSED for
~3 s. The current collector releases quickly, so #4 falls after release until
we lengthen the hold. #1-#3 are always valid.

Usage:
  from temporal_snapshots import extract_snapshots
  snaps, meta = extract_snapshots("gui_pt00_s1_tactile_maps.csv")
or run:  python3 temporal_snapshots.py <run_dir>
"""

import os, sys, glob, json
import numpy as np

MIRROR_S2 = False  # s2 shown mirrored L-R (pads face each other)
import pandas as pd

POST_SECONDS = 3.0   # snapshot #4 : 3 s after squeeze start

# ---- tolerate the tactile-writer race (see stitching._read_tactile_csv) ----
# Berith's per-frame writer occasionally collides two writes into one line,
# giving 31 fields instead of 30. One bad frame out of ~325 is irrelevant
# here, so skip it rather than abort the whole figure.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from stitching import _read_tactile_csv as _read_csv_tolerant
except Exception:                                    # standalone fallback
    def _read_csv_tolerant(path):
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except TypeError:                            # pandas < 1.3
            return pd.read_csv(path, error_bad_lines=False,
                               warn_bad_lines=False)



def extract_snapshots(csv_path):
    """Return (snaps, meta) for one tactile CSV.
    snaps: dict of (7,4) arrays keyed p05/p50/p95/post3s.
    meta:  per-snapshot frame index/time/sum + peak + post3s_valid flag."""
    df = _read_csv_tolerant(csv_path)
    pred = [c for c in df.columns if c.startswith("pred_")]
    v = df[pred].to_numpy()
    s = v.sum(1)
    t = df["time_sec"].to_numpy() if "time_sec" in df.columns else np.arange(len(s))/60.0
    smax = float(s.max()) if len(s) and s.max() > 0 else 1.0

    def first_cross(frac):
        return int(np.argmax(s >= frac * smax))

    idx = {"p05": first_cross(0.05),
           "p50": first_cross(0.50),
           "p95": first_cross(0.95)}
    t_start = t[idx["p05"]]
    post_valid = (t[-1] - t_start) >= POST_SECONDS
    idx["post3s"] = int(np.argmin(np.abs(t - (t_start + POST_SECONDS)))) if post_valid else len(s) - 1

    snaps, meta = {}, {}
    for key, i in idx.items():
        snaps[key] = v[i].reshape(7, 4)
        meta[key] = {"frame_index": int(i), "time_sec": float(t[i]), "sum": float(s[i])}
    meta["peak_sum"] = smax
    meta["post3s_valid"] = bool(post_valid)
    return snaps, meta


def process_run(run_dir):
    """Extract snapshots for every grasp (s1+s2) in a run; write JSON summary."""
    s1_files = sorted(glob.glob(os.path.join(run_dir, "*_pt*_s1_tactile_maps.csv")))
    results = {}
    for s1 in s1_files:
        tag = os.path.basename(s1).split("_s1_")[0]
        s2 = s1.replace("_s1_", "_s2_")
        entry = {}
        for name, path in [("s1", s1), ("s2", s2)]:
            if os.path.exists(path):
                snaps, meta = extract_snapshots(path)
                entry[name] = {"snapshots": {k: v.tolist() for k, v in snaps.items()},
                               "meta": meta}
        if entry:
            results[tag] = entry
    out = os.path.join(run_dir, "temporal_snapshots.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out} ({len(results)} grasps)")
    return results


def plot_run(run_dir, out_png=None):
    """Grid plot: rows=grasps, cols=the 4 snapshots. One PNG per sensor."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    results = process_run(run_dir)
    tags = sorted(results.keys())
    if not tags:
        print("no grasps found"); return []
    labels = ["p05", "p50", "p95", "post3s"]
    titles = ["5% (contact)", "50%", "95% (full)", "+3s"]
    made = []
    for sensor in ("s1", "s2"):
        rows = [tg for tg in tags if sensor in results[tg]]
        if not rows:
            continue
        # ONE colour scale across every grasp and every stage of this
        # sensor. Per-panel autoscaling made all four squeeze stages look
        # identical, which is exactly the objection raised on 21 July.
        vmax = 0.0
        for tg in rows:
            for key in labels:
                vmax = max(vmax, float(np.max(results[tg][sensor]["snapshots"][key])))
        if vmax <= 0:
            vmax = 1.0
        fig = Figure(figsize=(2.2*4 + 1.8, 2.0*len(rows) + 1))
        FigureCanvasAgg(fig)
        _im = None
        for r, tg in enumerate(rows):
            snaps = results[tg][sensor]["snapshots"]
            valid = results[tg][sensor]["meta"]["post3s_valid"]
            for c, key in enumerate(labels):
                ax = fig.add_subplot(len(rows), 4, r*4 + c + 1)
                _mp = np.array(snaps[key])
                if sensor == "s2" and MIRROR_S2:
                    _mp = _mp[:, ::-1]
                # ax.imshow(_mp, cmap="jet", aspect="auto")
                _im = ax.imshow(_mp, cmap="jet", aspect="auto",
                                origin="lower", vmin=0.0, vmax=vmax)  # plot_run

                if r == 0:
                    ttl = titles[c] + ("" if (key != "post3s" or valid) else "\n(needs 3s hold)")
                    ax.set_title(ttl, fontsize=8)
                if c == 0:
                    ax.set_ylabel(tg, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
        if _im is not None:
            cb = fig.colorbar(_im, ax=fig.axes, shrink=0.85)
            cb.set_label(f"pressure (a.u.) — shared scale, vmax={vmax:.0f}",
                         fontsize=8)
        fig.suptitle(f"Temporal snapshots — {sensor}{' [mirrored]' if (sensor=='s2' and MIRROR_S2) else ''} (rows=grasps, cols=squeeze stages; ONE colour scale)",
                     fontsize=11)
        png = os.path.join(run_dir, f"temporal_snapshots_{sensor}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        made.append(png)
        print(f"saved {png}")
    made.extend(plot_per_grasp(run_dir, results))   # per-grasp 2x4 figures
    return made


def plot_per_grasp(run_dir, results=None):
    """One figure PER GRASP: 2 rows (s1 top, s2 bottom) x 4 squeeze stages,
    shared color scale within the grasp so the stages are comparable.
    Saves <run_dir>/Temporal_Per_Grasp/temporal_<tag>.png; returns paths."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if results is None:
        results = process_run(run_dir)
    labels = ["p05", "p50", "p95", "post3s"]
    titles = ["5% (contact)", "50%", "95% (full)", "+3 s"]
    out_dir = os.path.join(run_dir, "Temporal_Per_Grasp")
    made = []
    for tg in sorted(results.keys()):
        entry = results[tg]
        os.makedirs(out_dir, exist_ok=True)
        vmax = max((float(np.max(entry[s]["snapshots"][k]))
                    for s in entry for k in labels), default=1.0)
        if vmax <= 0:
            vmax = 1.0
        fig = Figure(figsize=(2.3*4 + 0.8, 2.2*2 + 0.9))
        FigureCanvasAgg(fig)
        for row, sensor in enumerate(("s1", "s2")):
            if sensor not in entry:
                continue
            snaps = entry[sensor]["snapshots"]
            meta = entry[sensor]["meta"]
            for c, key in enumerate(labels):
                ax = fig.add_subplot(2, 4, row*4 + c + 1)
                _mp = np.array(snaps[key])
                if sensor == "s2" and MIRROR_S2:
                    _mp = _mp[:, ::-1]
                # ax.imshow(_mp, cmap="jet", aspect="auto",
                #           vmin=0.0, vmax=vmax)
                ax.imshow(_mp, cmap="jet", aspect="auto", vmin=0.0, vmax=vmax, origin="lower")  # plot_per_grasp
                ttl = f"{sensor}  {titles[c]}\nsum {meta[key]['sum']:.0f}"
                if key == "post3s" and not meta["post3s_valid"]:
                    ttl += " (needs 3s hold)"
                ax.set_title(ttl, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{tg} — temporal snapshots (top=s1, bottom=s2)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        png = os.path.join(out_dir, f"temporal_{tg}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        made.append(png)
        print(f"saved {png}")
    return made


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Paper3_Simulation/Data/gui_run")
    plot_run(rd)
