"""
grid_accuracy.py — designed grid vs where the robot ACTUALLY put the pad.

Answers one question, with numbers: when the GUI asks for a pad centre at
(Y, Z), how close does the arm get? Nothing here re-derives geometry — the
commanded points come from the run's own config copy and the measured ones
from stitching.load_offsets, i.e. the SAME positions the stitcher paints with,
so this figure cannot disagree with the stitched map.

Four panels:
  1. overlay      commanded (open circles) vs measured (filled), joined by an
                  error line drawn at x50 so a 0.2 mm miss is visible at all
  2. error vs Y   signed dY and dZ against commanded Y
  3. error vs Z   signed dY and dZ against commanded Z
  4. scatter      dY vs dZ, with the mean (bias) and the 95% circle

Reported separately: BIAS (the mean offset — a systematic frame or calibration
error) and SCATTER (the std — per-point repeatability). They mean different
things and a single "error" number hides both.

Usage:
    from grid_accuracy import plot_run
    png, stats = plot_run(run_dir)
or:
    python3 grid_accuracy.py <run_dir>
"""

import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _commanded(run_dir):
    """{ptNN: (y_mm, z_mm)} the GUI ASKED for, in world/GUI coordinates.

    object centre + pad_offset, which is exactly what the collector turns into
    a target. Reads the run's OWN config copy, never the live one, so an
    edited GUI cannot silently redefine what an old run was asked to do."""
    for name in ("gui_config_used.json", "gui_config.json"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            c = json.load(f)
        ctr = c["object"]["center_world_mm"]
        out = {}
        for pt in c.get("points", []):
            out[f"pt{int(pt['index']):02d}"] = (
                float(ctr[1]) + float(pt["pad_offset_y_mm"]),
                float(ctr[2]) + float(pt["pad_offset_z_mm"]))
        if out:
            return out, name, c
    return {}, None, {}


def measure_run(run_dir):
    """(rows, stats, meta). rows = [(key, cy, cz, my, mz, dy, dz, d), ...]
    sorted by point index; d is the Euclidean miss in mm."""
    import stitching as ST

    cmd, cfg_name, cfg = _commanded(run_dir)
    if not cmd:
        raise RuntimeError(f"no config with points found in {run_dir}")
    meas, src = ST.load_offsets(run_dir)
    if not meas:
        raise RuntimeError(f"no measured pad positions found in {run_dir}")
    if "config" in (src or ""):
        # load_offsets fell all the way through to the commanded points, so
        # measured == commanded by construction and the plot would show a
        # perfect zero that means nothing. Refuse rather than mislead.
        raise RuntimeError(
            f"measured positions unavailable: load_offsets fell back to "
            f"{src!r}, which IS the commanded grid. This run has no usable "
            f"pose_history.json, so there is nothing to compare against.")

    keys = sorted(set(cmd) & set(meas),
                  key=lambda k: int(k.replace("pt", "")))
    rows = []
    for k in keys:
        cy, cz = cmd[k]
        my, mz = meas[k]
        dy, dz = my - cy, mz - cz
        rows.append((k, cy, cz, my, mz, dy, dz, float(np.hypot(dy, dz))))

    dY = np.array([r[5] for r in rows])
    dZ = np.array([r[6] for r in rows])
    D = np.array([r[7] for r in rows])
    stats = {
        "n_commanded": len(cmd), "n_measured": len(meas), "n_compared": len(rows),
        "missing": sorted(set(cmd) - set(meas),
                          key=lambda k: int(k.replace("pt", ""))),
        "bias_y_mm": float(dY.mean()) if len(dY) else 0.0,
        "bias_z_mm": float(dZ.mean()) if len(dZ) else 0.0,
        "std_y_mm": float(dY.std(ddof=1)) if len(dY) > 1 else 0.0,
        "std_z_mm": float(dZ.std(ddof=1)) if len(dZ) > 1 else 0.0,
        "mean_miss_mm": float(D.mean()) if len(D) else 0.0,
        "median_miss_mm": float(np.median(D)) if len(D) else 0.0,
        "max_miss_mm": float(D.max()) if len(D) else 0.0,
        "max_miss_at": rows[int(np.argmax(D))][0] if len(D) else None,
        "p95_miss_mm": float(np.percentile(D, 95)) if len(D) else 0.0,
        "offset_source": src,
        "config": cfg_name,
        "roll_deg": float((cfg.get("pad") or {}).get("rotation_deg", 0.0)),
        "step_mm": float((cfg.get("grid") or {}).get("step_mm", 0.0)),
        "step_frame": str((cfg.get("grid") or {}).get("step_frame", "world")),
    }
    return rows, stats, cfg


def format_report(run_dir, rows, stats):
    """Plain-text report — the same numbers as the figure, for the log."""
    L = [f"GRID ACCURACY — commanded vs measured pad centre",
         f"run   : {run_dir}",
         f"source: {stats['offset_source']}",
         f"config: {stats['config']}  (roll {stats['roll_deg']:+.1f} deg, "
         f"step {stats['step_mm']:.1f} mm, steps along {stats['step_frame']})",
         "-" * 66,
         f"{'point':>6} {'cmd Y':>9} {'cmd Z':>9} "
         f"{'dY':>8} {'dZ':>8} {'miss':>8}"]
    for k, cy, cz, _my, _mz, dy, dz, d in rows:
        L.append(f"{k:>6} {cy:9.2f} {cz:9.2f} {dy:+8.3f} {dz:+8.3f} {d:8.3f}")
    L += ["-" * 66,
          f"compared {stats['n_compared']} of {stats['n_commanded']} "
          f"commanded points",
          f"BIAS    (mean, systematic) : dY {stats['bias_y_mm']:+.3f}  "
          f"dZ {stats['bias_z_mm']:+.3f}  mm",
          f"SCATTER (std, per-point)   : dY {stats['std_y_mm']:.3f}  "
          f"dZ {stats['std_z_mm']:.3f}  mm",
          f"miss  mean {stats['mean_miss_mm']:.3f}  "
          f"median {stats['median_miss_mm']:.3f}  "
          f"p95 {stats['p95_miss_mm']:.3f}  "
          f"max {stats['max_miss_mm']:.3f} mm at {stats['max_miss_at']}"]
    if stats["missing"]:
        L.append(f"NOT REACHED ({len(stats['missing'])}): "
                 + ", ".join(stats["missing"]))
    L += ["",
          "BIAS is a constant offset — a frame or calibration error, and it",
          "moves the whole stitched map together, so it does NOT blur it.",
          "SCATTER is per-point repeatability — this is what smears the",
          "stitch, because neighbouring grasps land inconsistently.",
          "For reference, one taxel is 5.5 mm across and 5.29 mm up."]
    return "\n".join(L)


def plot_run(run_dir, out_png=None, err_scale=50.0):
    """Write Stitched/grid_accuracy.png (+ .txt). Returns (png, stats)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rows, stats, _cfg = measure_run(run_dir)
    out_dir = os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    png = out_png or os.path.join(out_dir, "grid_accuracy.png")

    cy = np.array([r[1] for r in rows]); cz = np.array([r[2] for r in rows])
    my = np.array([r[3] for r in rows]); mz = np.array([r[4] for r in rows])
    dy = np.array([r[5] for r in rows]); dz = np.array([r[6] for r in rows])
    d = np.array([r[7] for r in rows])

    fig = Figure(figsize=(15.0, 4.2))
    FigureCanvasAgg(fig)

    # ---- 1. overlay, error exaggerated so it is visible at all -------------
    ax = fig.add_subplot(1, 4, 1)
    for k, a, b, c_, e, *_ in rows:
        ax.plot([a, a + (c_ - a) * err_scale], [b, b + (e - b) * err_scale],
                "-", color="#d62728", lw=1.0, zorder=3)
    ax.plot(cy, cz, "o", mfc="none", mec="#06a", ms=7, mew=1.2,
            label="commanded", zorder=4)
    ax.plot(my, mz, ".", color="#d62728", ms=5, label="measured", zorder=5)
    for k, a, b, *_ in rows:
        if k in ("pt00",):
            ax.plot([a], [b], "s", mfc="none", mec="#7b2fbe", ms=12, mew=1.8,
                    zorder=6, label="pt00 (initial)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("world Y (mm)"); ax.set_ylabel("world Z (mm)")
    ax.set_title(f"commanded vs measured\n(error drawn x{err_scale:.0f})",
                 fontsize=9)
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3, lw=0.4)

    # ---- 2 & 3. signed error against each commanded axis -------------------
    for i, (axis, vals, lbl) in enumerate(
            [(2, cy, "commanded Y (mm)"), (3, cz, "commanded Z (mm)")]):
        a2 = fig.add_subplot(1, 4, axis)
        a2.axhline(0, color="0.6", lw=0.8)
        a2.plot(vals, dy, "o", ms=4, color="#06a", label="dY")
        a2.plot(vals, dz, "^", ms=4, color="#d62728", label="dZ")
        a2.set_xlabel(lbl); a2.set_ylabel("error (mm)")
        a2.set_title(f"error vs {lbl.split()[1]}", fontsize=9)
        a2.legend(fontsize=7); a2.grid(alpha=0.3, lw=0.4)

    # ---- 4. dY vs dZ, bias and spread --------------------------------------
    a4 = fig.add_subplot(1, 4, 4)
    a4.axhline(0, color="0.6", lw=0.8); a4.axvline(0, color="0.6", lw=0.8)
    a4.plot(dy, dz, "o", ms=5, color="#06a", alpha=0.75, label="points")
    a4.plot([stats["bias_y_mm"]], [stats["bias_z_mm"]], "x", color="#d62728",
            ms=11, mew=2.2, label="bias (mean)")
    th = np.linspace(0, 2 * np.pi, 200)
    r95 = stats["p95_miss_mm"]
    a4.plot(r95 * np.cos(th), r95 * np.sin(th), "--", color="#d62728", lw=1.0,
            label=f"p95 = {r95:.2f} mm")
    a4.set_aspect("equal", adjustable="datalim")
    a4.set_xlabel("dY (mm)"); a4.set_ylabel("dZ (mm)")
    a4.set_title("error scatter", fontsize=9)
    a4.legend(fontsize=7, loc="best"); a4.grid(alpha=0.3, lw=0.4)

    miss_txt = (f"   |   {len(stats['missing'])} NOT REACHED"
                if stats["missing"] else "")
    fig.suptitle(
        f"GRID ACCURACY — {stats['n_compared']} points   "
        f"bias (dY {stats['bias_y_mm']:+.2f}, dZ {stats['bias_z_mm']:+.2f}) mm   "
        f"scatter (dY {stats['std_y_mm']:.2f}, dZ {stats['std_z_mm']:.2f}) mm   "
        f"max miss {stats['max_miss_mm']:.2f} mm at {stats['max_miss_at']}   "
        f"[1 taxel = 5.5 x 5.29 mm]{miss_txt}",
        fontsize=10, y=1.03)

    fig.savefig(png, dpi=120, bbox_inches="tight")
    txt = os.path.splitext(png)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(format_report(run_dir, rows, stats) + "\n")
    print(f"saved {png}")
    print(f"saved {txt}")
    return png, stats


if __name__ == "__main__":
    import glob
    rd = (sys.argv[1] if len(sys.argv) > 1
          else os.path.expanduser("~/Paper3_Simulation/Data/gui_run"))
    if not glob.glob(os.path.join(rd, "pose_history.json")):
        runs = sorted(glob.glob(os.path.join(rd, "run_*")))
        if runs:
            rd = runs[-1]
    _png, _st = plot_run(rd)
    print()
    print(format_report(rd, *measure_run(rd)[:2]))
