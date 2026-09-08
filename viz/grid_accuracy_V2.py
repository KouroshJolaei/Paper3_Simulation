"""grid_accuracy.py — did the pad actually go where the grid said?

WHY THIS EXISTS
---------------
Every downstream number assumes the pad was where the design said it was.
The stitcher paints each 7x4 map at a commanded position; the expected-band
panel predicts contact from that position; the training pair is labelled with
it. If the arm misses, none of that announces itself -- the map is simply
painted in the wrong place and the stitch quietly blurs.

So this compares, per point, the pad centre the GUI asked for against the pad
centre forward kinematics says was reached, and reports the two failure modes
separately, because they do completely different damage:

    BIAS    a constant offset. A frame or calibration error. It translates
            the whole stitched map together, so it does NOT blur it -- every
            grasp is wrong by the same amount and their overlaps still agree.
    SCATTER per-point repeatability. THIS is what smears a stitch, because
            neighbouring grasps land inconsistently and their overlapping
            taxels disagree for a reason that has nothing to do with the
            object.

WHY THE FIGURE IS SHAPED LIKE THIS (rewritten 2026-09-02)
---------------------------------------------------------
The first version drew commanded-vs-measured with the error magnified x50,
then dY and dZ against Y and against Z, then a scatter. Three problems:

  1. At x50 a 0.03 mm error draws as 1.5 mm against a 6 mm grid pitch, so the
     measured dots sat inside the commanded circles and the panel showed
     nothing. It was a picture of an invisible quantity.
  2. dY/dZ vs Y and dY/dZ vs Z are the same 117 points plotted twice, and
     both were being read for one question -- is the error positional? A map
     answers that instantly; two scatter plots make you infer it.
  3. Nothing gave the numbers a scale. "0.074 mm" needs a division before it
     means anything. The unit that matters is ONE TAXEL: 5.5 x 5.29 mm.

Each panel now answers ONE question, and its title is that question:
    WHERE is the error?          spatial map, magnitude in colour, direction
                                 as arrows at a stated magnification
    HOW BIG is it, really?       drawn inside a true-scale taxel, with a
                                 magnified inset for the structure
    Does it DRIFT?               versus visit order, which is a different
                                 question from position and catches settling,
                                 thermal drift and one-off outliers
    Any bad ROW or COLUMN?       the aggregation that made the throat jam
                                 obvious in grid_quality: a fault at one grid
                                 coordinate is invisible in a 117-point list
                                 and unmissable once grouped

USAGE
    import grid_accuracy as GA
    png, stats = GA.plot_run(run_dir)
or
    python3 grid_accuracy.py <run_dir>
"""

import os
import sys
import json
import numpy as np

PAD_W, PAD_H = 22.0, 37.0
PITCH_Y = 5.5                  # mm, one taxel across
PITCH_Z = 37.0 / 7.0           # mm, one taxel up


def _commanded(run_dir):
    """{ptNN: (y_mm, z_mm)} the GUI asked for, in the same world frame the
    measured offsets use. The config stores pad offsets RELATIVE to the object
    centre, so the centre has to be added back."""
    for name in ("gui_config_used.json", "gui_config.json"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                cfg = json.load(f)
            cfg = cfg.get("config", cfg)
            c = cfg["object"]["center_world_mm"]
            cy, cz = float(c[1]), float(c[2])
            out = {}
            for q in cfg.get("points", []):
                tag = f"pt{int(q['index']):02d}"
                out[tag] = (cy + float(q["pad_offset_y_mm"]),
                            cz + float(q["pad_offset_z_mm"]))
            if out:
                return out, cfg
        except Exception:
            continue
    return {}, {}


def measure(run_dir):
    """Join commanded to measured. Returns (rows, stats)."""
    import stitching as ST
    cmd, cfg = _commanded(run_dir)
    if not cmd:
        raise RuntimeError("no commanded grid found (gui_config_used.json)")
    offs, src = ST.load_offsets(run_dir)

    rows, missing = [], []
    for tag in sorted(cmd, key=lambda t: int(t[2:])):
        cy, cz = cmd[tag]
        if tag not in offs:
            missing.append(tag)
            continue
        my, mz = offs[tag]
        dy, dz = my - cy, mz - cz
        rows.append({"tag": tag, "index": int(tag[2:]),
                     "cmd_y": cy, "cmd_z": cz, "meas_y": my, "meas_z": mz,
                     "dy": dy, "dz": dz, "miss": float(np.hypot(dy, dz))})
    if not rows:
        raise RuntimeError("no point matched between config and pose_history")

    dy = np.array([r["dy"] for r in rows])
    dz = np.array([r["dz"] for r in rows])
    miss = np.array([r["miss"] for r in rows])
    imax = int(np.argmax(miss))
    stats = {
        "n_commanded": len(cmd), "n_compared": len(rows),
        "bias_y_mm": float(dy.mean()), "bias_z_mm": float(dz.mean()),
        "std_y_mm": float(dy.std()), "std_z_mm": float(dz.std()),
        "mean_miss_mm": float(miss.mean()),
        "median_miss_mm": float(np.median(miss)),
        "p95_miss_mm": float(np.percentile(miss, 95)),
        "max_miss_mm": float(miss.max()), "max_miss_at": rows[imax]["tag"],
        "offset_source": src, "missing": missing,
        # the numbers that actually mean something: everything as a fraction
        # of ONE TAXEL, because that is the resolution the map is painted at
        "scatter_taxel_frac": float(max(dy.std() / PITCH_Y,
                                        dz.std() / PITCH_Z)),
        "max_miss_taxel_frac": float(miss.max() / PITCH_Y),
        "object": cfg.get("object", {}), "grid": cfg.get("grid", {})}
    return rows, stats


def report(run_dir):
    rows, st = measure(run_dir)
    L = ["GRID ACCURACY — commanded vs measured pad centre",
         f"run   : {os.path.abspath(run_dir)}",
         f"source: {st['offset_source']}",
         "-" * 66,
         " point     cmd Y     cmd Z       dY       dZ     miss"]
    for r in rows:
        L.append(f"  {r['tag']:>5s}  {r['cmd_y']:8.2f} {r['cmd_z']:9.2f}"
                 f"  {r['dy']:+7.3f}  {r['dz']:+7.3f}  {r['miss']:7.3f}")
    L += ["-" * 66,
          f"compared {st['n_compared']} of {st['n_commanded']} commanded points",
          f"BIAS    (mean, systematic) : dY {st['bias_y_mm']:+.3f}  "
          f"dZ {st['bias_z_mm']:+.3f}  mm",
          f"SCATTER (std, per-point)   : dY {st['std_y_mm']:.3f}  "
          f"dZ {st['std_z_mm']:.3f}  mm",
          f"miss  mean {st['mean_miss_mm']:.3f}  "
          f"median {st['median_miss_mm']:.3f}  "
          f"p95 {st['p95_miss_mm']:.3f}  "
          f"max {st['max_miss_mm']:.3f} mm at {st['max_miss_at']}",
          "",
          "IN TAXELS — the unit that decides whether any of this matters:",
          f"  scatter  = {100*st['scatter_taxel_frac']:.2f} % of one taxel",
          f"  max miss = {100*st['max_miss_taxel_frac']:.2f} % of one taxel",
          "",
          "BIAS is a constant offset — a frame or calibration error. It moves",
          "the whole stitched map together, so it does NOT blur it.",
          "SCATTER is per-point repeatability. This is what smears the stitch,",
          "because neighbouring grasps land inconsistently.",
          "One taxel is 5.50 mm across and 5.29 mm up."]
    if st["missing"]:
        L.append("")
        L.append(f"NOT REACHED ({len(st['missing'])}): "
                 + ", ".join(st["missing"]))
    return "\n".join(L)


def plot_run(run_dir, out_dir=None):
    """Four panels, each answering one question. Returns (png_path, stats)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle, Circle

    rows, st = measure(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)

    cy = np.array([r["cmd_y"] for r in rows])
    cz = np.array([r["cmd_z"] for r in rows])
    dy = np.array([r["dy"] for r in rows])
    dz = np.array([r["dz"] for r in rows])
    miss = np.array([r["miss"] for r in rows])
    idx = np.array([r["index"] for r in rows])

    fig = Figure(figsize=(16.0, 9.0))
    FigureCanvasAgg(fig)

    # ---- 1. WHERE is the error? ------------------------------------------
    # Magnitude in colour so a bad region is a coloured patch, direction as
    # arrows so a systematic pull is a combed field rather than a hedgehog.
    ax = fig.add_subplot(2, 2, 1)
    sc = ax.scatter(cy, cz, c=1000.0 * miss, s=150, marker="s",
                    cmap="magma_r", edgecolors="k", linewidths=0.3, zorder=2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("miss (µm)", fontsize=8)
    step = float((st["grid"] or {}).get("step_mm", 6.0)) or 6.0
    # Scale so the WORST arrow spans ~1.2 grid steps. A third of a step was
    # tried first and drew arrows ~2 mm long on a 90 mm plot -- invisible,
    # which is the exact failure the old x50 panel had. The magnification is
    # printed rather than left for the eye to guess.
    amp = (step * 1.2) / max(miss.max(), 1e-9)
    ax.quiver(cy, cz, dy * amp, dz * amp, angles="xy", scale_units="xy",
              scale=1.0, width=0.004, color="#1f77b4", zorder=3)
    ax.set_xlabel("commanded Y (mm)", fontsize=9)
    ax.set_ylabel("commanded Z (mm)", fontsize=9)
    ax.set_title(f"WHERE is the error?\ncolour = size, arrows = direction "
                 f"(x{amp:.0f})", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)

    # ---- 2. HOW BIG is it, really? ---------------------------------------
    # Drawn inside ONE TAXEL at true scale. If the cloud is a dot, that IS
    # the finding, and no amount of axis-stretching should be allowed to
    # disguise it. The inset then magnifies the dot for its structure.
    ax = fig.add_subplot(2, 2, 2)
    ax.add_patch(Rectangle((-PITCH_Y / 2, -PITCH_Z / 2), PITCH_Y, PITCH_Z,
                           fill=False, ec="#c0392b", lw=1.8, ls="--",
                           label="one taxel (5.50 x 5.29 mm)"))
    ax.scatter(dy, dz, s=14, c="#1f77b4", alpha=0.85, label="points")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlim(-PITCH_Y * 0.62, PITCH_Y * 0.62)
    ax.set_ylim(-PITCH_Z * 0.62, PITCH_Z * 0.62)
    ax.set_aspect("equal")
    ax.set_xlabel("dY (mm)", fontsize=9)
    ax.set_ylabel("dZ (mm)", fontsize=9)
    ax.set_title(f"HOW BIG is it, really?\nevery point inside one taxel — "
                 f"scatter {100*st['scatter_taxel_frac']:.2f} % of a taxel",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)

    lim = max(np.abs(dy).max(), np.abs(dz).max(), 1e-3) * 1.25
    ins = ax.inset_axes([0.56, 0.05, 0.40, 0.40], zorder=5)
    ins.set_facecolor("white")
    for _sp in ins.spines.values():
        _sp.set_edgecolor("#555")
    ins.scatter(dy, dz, s=9, c="#1f77b4", alpha=0.85)
    ins.add_patch(Circle((0, 0), st["p95_miss_mm"], fill=False, ec="#c0392b",
                         ls=":", lw=1.0))
    ins.scatter([st["bias_y_mm"]], [st["bias_z_mm"]], marker="x", s=60,
                c="#c0392b", lw=1.8)
    ins.axhline(0, color="#bbb", lw=0.6)
    ins.axvline(0, color="#bbb", lw=0.6)
    ins.set_xlim(-lim, lim)
    ins.set_ylim(-lim, lim)
    ins.set_aspect("equal")
    ins.tick_params(labelsize=6)
    ins.set_title(f"magnified x{PITCH_Y / (2 * lim):.0f}\nX = bias,  dotted = p95", fontsize=7)

    # ---- 3. Does it DRIFT during the run? --------------------------------
    # Position and time are different questions. A stitch collected over two
    # hours can drift without any spatial pattern at all, and a single
    # outlier is a spike here and invisible everywhere else.
    ax = fig.add_subplot(2, 2, 3)
    order = np.argsort(idx)
    ax.plot(idx[order], 1000.0 * miss[order], "-o", ms=3, lw=1.0,
            color="#1f77b4")
    ax.axhline(1000.0 * st["p95_miss_mm"], ls=":", lw=1.2, color="#c0392b",
               label=f"p95 = {1000*st['p95_miss_mm']:.0f} µm")
    ax.axhline(1000.0 * st["median_miss_mm"], ls="--", lw=1.0, color="#888",
               label=f"median = {1000*st['median_miss_mm']:.0f} µm")
    ax.annotate(st["max_miss_at"],
                xy=(idx[int(np.argmax(miss))], 1000.0 * miss.max()),
                xytext=(6, 6), textcoords="offset points", fontsize=8,
                color="#c0392b")
    ax.set_xlabel("visit order (point index)", fontsize=9)
    ax.set_ylabel("miss (µm)", fontsize=9)
    ax.set_title("Does it DRIFT during the run?\nflat = repeatable, "
                 "sloped = something is moving", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)

    # ---- 4. Any bad ROW or COLUMN? ---------------------------------------
    # The same aggregation that made the throat jam obvious in grid_quality.
    ax = fig.add_subplot(2, 2, 4)

    def _agg(coord):
        g = {}
        for k, m in zip(coord, miss):
            g.setdefault(round(float(k), 2), []).append(1000.0 * m)
        ks = sorted(g)
        return (np.array(ks),
                np.array([np.mean(g[k]) for k in ks]),
                np.array([np.max(g[k]) for k in ks]))

    ky, my_, xy_ = _agg(cy)
    kz, mz_, xz_ = _agg(cz)
    ax.plot(ky - ky.mean(), my_, "-o", ms=4, color="#1f77b4",
            label="by COLUMN (Y), mean")
    ax.plot(ky - ky.mean(), xy_, ":", lw=1.0, color="#1f77b4",
            alpha=0.5,
            label="by COLUMN, worst")
    ax.plot(kz - kz.mean(), mz_, "-s", ms=4, color="#e67e22",
            label="by ROW (Z), mean")
    ax.plot(kz - kz.mean(), xz_, ":", lw=1.0, color="#e67e22",
            alpha=0.5,
            label="by ROW, worst")
    ax.set_xlabel("distance from the sweep centre (mm)", fontsize=9)
    ax.set_ylabel("miss (µm)", fontsize=9)
    ax.set_title("Any bad ROW or COLUMN?\na whole line standing out is "
                 "geometry, not noise", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)

    o = st["object"] or {}
    head = (f"GRID ACCURACY — {st['n_compared']} of {st['n_commanded']} points"
            f"   |   scatter {100*st['scatter_taxel_frac']:.2f} % of a taxel"
            f"   |   worst miss {100*st['max_miss_taxel_frac']:.2f} % of a "
            f"taxel ({st['max_miss_at']})")
    sub = (f"bias dY {st['bias_y_mm']:+.3f}  dZ {st['bias_z_mm']:+.3f} mm "
           f"(shifts the map, does not blur it)   |   "
           f"scatter dY {st['std_y_mm']:.3f}  dZ {st['std_z_mm']:.3f} mm "
           f"(this is what blurs it)")
    if o:
        sub += f"   |   {o.get('shape', '?')} {o.get('diameter_mm', '?')}x" \
               f"{o.get('across_mm', o.get('diameter_mm', '?'))}x" \
               f"{o.get('length_mm', '?')} mm"
    fig.suptitle(head + "\n" + sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    png = os.path.join(out_dir, "grid_accuracy.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    with open(os.path.join(out_dir, "grid_accuracy.txt"), "w") as f:
        f.write(report(run_dir) + "\n")
    return png, st


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    _png, _st = plot_run(sys.argv[1])
    print(report(sys.argv[1]))
    print("\nsaved " + _png)
