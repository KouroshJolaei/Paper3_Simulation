"""
blob_axis.py — contact-blob orientation by weighted PCA (Paper 2's method).

WHY THIS EXISTS
---------------
The first tilt metric fitted a line through per-row pressure centroids. That
collapses each row to one number and then fits 7 points, and it needs a
threshold to decide which rows count — on a narrow contact only 3-5 rows
survive, and the answer moves with that choice.

Paper 2 already solved this, in virtual_search.generate_eigen_align:
    upsample 7x4 -> 70x40 (cubic), threshold, then a PRESSURE-WEIGHTED
    covariance in millimetres; the principal eigenvector is the contact's
    long axis. This file reproduces that chain so Paper 3 reports the same
    quantity Paper 2 does, rather than a new invented one.

THE ONE DELIBERATE DIFFERENCE
-----------------------------
Paper 2 stores maps with ROW 0 = TOP (network_imagination.plot_heatmaps2
does np.flipud before drawing with origin="lower"), so it builds physical y
as (centre_row - row). Paper 3 stores ROW 0 = BOTTOM — verified 2026-07-30
against a rod-tip grasp, where the rod touched only the pad's lowest 8.5 mm
and rows r0/r1 carried 59% of the signal. So here y grows WITH the row
index. Copying Paper 2's sign would negate every angle.

CONVENTIONS
-----------
  x = across the pad  (4 taxels, PITCH_Y = 5.50 mm)
  y = up the pad      (7 taxels, PITCH_Z = 5.286 mm), +y = toward row 6
  angle_from_vertical_deg : 0 = contact line runs straight up the pad,
      positive = leaning toward +x as you go up. Undirected, so it is
      wrapped to (-90, +90].

For a cylinder tilted by T degrees about X, the contact line follows the
cylinder axis, so the expected reading is |angle| = T.

Usage:
  from blob_axis import blob_axis
  info = blob_axis(map7x4)
or:
  python3 blob_axis.py <run_dir> [tilt_deg]
"""

import os, sys, glob, json
import numpy as np

PITCH_Y = 5.5                 # mm, across the pad (4 columns)
PITCH_Z = 37.0 / 7.0          # mm, up the pad (7 rows)
PAD_W, PAD_H = 22.0, 37.0
N_ROWS, N_COLS = 7, 4

# Paper 2's values (virtual_search.generate_eigen_align defaults)
UPSAMPLE = 10                 # set_Domain: nHR = 10  -> 70 x 40
T_THRESHOLD = 0.35
STRICTNESS = 1.25             # effective threshold = 0.4375 of the range
MIN_CELLS = 5                 # "if len(rows) < 5: return"


def _upsample(m, r=UPSAMPLE):
    """7x4 -> 70x40, cubic, exactly as network_imagination.High_Res."""
    if r <= 1:
        return np.asarray(m, float)
    try:
        from scipy.ndimage import zoom
        return zoom(np.asarray(m, float), (r, r), order=3)
    except Exception:
        return np.repeat(np.repeat(np.asarray(m, float), r, 0), r, 1)


def blob_axis(map7x4, upsample=UPSAMPLE, t_threshold=T_THRESHOLD,
              strictness=STRICTNESS):
    """Weighted-PCA orientation of the contact blob.

    Returns a dict, or {"ok": False, "reason": ...} when the contact is too
    small or too round for an axis to mean anything."""
    m0 = np.asarray(map7x4, float)
    if m0.shape != (N_ROWS, N_COLS):
        return {"ok": False, "reason": f"expected (7,4), got {m0.shape}"}
    P = _upsample(m0, upsample)
    H, W = P.shape

    eff = min(t_threshold * strictness, 0.95)
    lo, hi = float(P.min()), float(P.max())
    if hi - lo <= 1e-9:
        return {"ok": False, "reason": "flat map (no contact)"}
    thr = lo + eff * (hi - lo)
    rr, cc = np.where(P > thr)
    if rr.size < MIN_CELLS:
        return {"ok": False, "reason": f"only {rr.size} cells above threshold"}
    w = P[rr, cc]

    # physical mm. ROW 0 = BOTTOM here, so y grows WITH the row index —
    # the opposite of Paper 2's image-convention (pcy - rows).
    mm_x, mm_y = PAD_W / W, PAD_H / H
    x = (cc - (W - 1) / 2.0) * mm_x
    y = (rr - (H - 1) / 2.0) * mm_y
    pts = np.column_stack((x, y))

    cen = np.average(pts, axis=0, weights=w)
    cov = np.cov(pts - cen, rowvar=False, aweights=w)
    if not np.all(np.isfinite(cov)):
        return {"ok": False, "reason": "degenerate covariance"}
    vals, vecs = np.linalg.eigh(cov)          # ascending
    major = vecs[:, -1]

    raw = np.degrees(np.arctan2(major[1], major[0]))   # from +x axis
    # A line leaning `ang` from vertical has direction (sin ang, cos ang),
    # so raw = 90 - ang  ->  ang = 90 - raw. (Paper 2 writes raw - 90 because
    # it aligns the SENSOR to the blob; we are reporting the blob itself.)
    ang = 90.0 - raw
    ang = (ang + 90.0) % 180.0 - 90.0                  # axis is undirected

    maj = float(np.sqrt(max(vals[-1], 0.0)))
    mnr = float(np.sqrt(max(vals[0], 0.0)))
    elong = maj / mnr if mnr > 1e-9 else np.inf

    return {"ok": True,
            "angle_from_vertical_deg": float(ang),
            # same quantity as the old row-centroid fit, for comparison
            "slope_col_per_row": float(np.tan(np.radians(ang))
                                       * PITCH_Z / PITCH_Y),
            "major_std_mm": maj, "minor_std_mm": mnr,
            "elongation": float(elong),
            "n_cells": int(rr.size),
            "centroid_mm": [float(cen[0]), float(cen[1])],
            "threshold_frac": float(eff)}


# ------------------------------------------------------------------ run ----
def run_folder(run_dir, tilt_deg=None, sensor="s1"):
    """Blob axis for every grasp, plus the angle-vs-lateral-position fit."""
    import stitching as ST
    files = sorted(glob.glob(os.path.join(run_dir,
                                          f"*_pt*_{sensor}_tactile_maps.csv")))
    if not files:
        return None
    ph = {}
    try:
        with open(os.path.join(run_dir, "pose_history.json")) as f:
            for e in json.load(f)["points"]:
                ph[e["tag"]] = e["pad_desired_pos_m"][1] * 1000.0
    except Exception:
        pass

    L = [f"BLOB AXIS ({sensor})  —  Paper-2 weighted PCA, "
         f"upsample x{UPSAMPLE}, threshold {T_THRESHOLD*STRICTNESS:.3f}"]
    if tilt_deg is not None:
        L.append(f"  object tilt = {tilt_deg} deg  ->  expect "
                 f"|angle| = {abs(float(tilt_deg)):.1f} deg, "
                 f"slope {np.tan(np.radians(float(tilt_deg)))*PITCH_Z/PITCH_Y:.3f}")
    L.append("")
    L.append("  grasp    Y(mm)   angle(deg)  slope   elong  cells")
    Y, A = [], []
    for f in files:
        tag = os.path.basename(f).split("_")[1]
        m, _, _ = ST.hold_average(f)
        info = blob_axis(m)
        yy = ph.get(f"{tag}", np.nan)
        if not info["ok"]:
            L.append(f"   {tag}   {yy:7.1f}   -- {info['reason']}")
            continue
        L.append(f"   {tag}   {yy:7.1f}   {info['angle_from_vertical_deg']:+8.2f}"
                 f"  {info['slope_col_per_row']:+.3f}"
                 f"  {info['elongation']:6.2f}  {info['n_cells']:5d}")
        if np.isfinite(yy):
            Y.append(yy); A.append(info["angle_from_vertical_deg"])
    if len(Y) >= 3:
        Y = np.array(Y) - np.mean(Y); A = np.array(A)
        a, b = np.polyfit(Y, A, 1)
        r = float(np.corrcoef(Y, A)[0, 1])
        L.append("")
        L.append(f"  mean angle = {A.mean():+.2f} deg   "
                 f"(a real tilt shows up HERE, the same at every grasp)")
        L.append(f"  angle vs lateral Y: {a:+.4f} deg/mm, r = {r:+.3f}   "
                 f"(a position artifact shows up HERE)")
    return "\n".join(L)


# ------------------------------------------------- design vs measured ----
# WHICH ANGLE SHOULD THE CONTACT LINE HAVE?  (added 2026-08-05)
#
# Two rotations decide it, and they are NOT the same rotation:
#   * the ROD leans in the world by the scene tilt T (about world X)
#   * the PAD is rolled by rho, measured from pad_actual_R
# blob_axis reports the angle in the PAD's own frame, so what it should read
# is the rod's direction as seen from a rolled pad:
#
#       alpha_expected = rho - TILT_SIGN * T
#
# Derivation, so this is checkable rather than asserted:
#   stitching.py draws the rod's axis (column 3) as (-sin T, cos T) in world
#   (Y, Z), i.e. a world angle of -T measured from +Z toward +Y. A pad-frame
#   angle alpha maps to the world direction (sin(alpha - rho), cos(alpha -
#   rho)), so world = alpha - rho, hence alpha = world + rho = rho - T.
# Two checks:
#   rod upright (T=0), pad rolled -25  ->  alpha = -25   (pad sees a lean)
#   rod tilted 20, pad flat (rho=0)    ->  alpha = -20, |alpha| = 20, which
#                                          is the case blob_axis's own
#                                          docstring describes.
# The pad FOLLOWS the rod when rho = T, giving alpha = 0 — that null case is
# the cleanest single test of whether the CNN can see an oblique contact.
#
# TILT_SIGN encodes how the scene's tilt_deg maps to a world lean. It follows
# stitching.py's column-3 drawing, which has been checked against the GUI by
# eye. If a run ever shows the measured and expected angles equal and
# opposite at rho = 0, flip this to +1 — do not adjust anything else.
TILT_SIGN = 1.0


def expected_pad_angle_deg(roll_deg, tilt_deg):
    """Pad-frame angle the contact line should have, wrapped to (-90, +90]."""
    a = float(roll_deg) - TILT_SIGN * float(tilt_deg)
    return (a + 90.0) % 180.0 - 90.0


def _scene_tilt(run_dir):
    """(tilt_deg, axis) from the run's own config. 0 if not recorded."""
    for name in ("gui_config_used.json", "gui_config.json",
                 "pose_history.json"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            obj = (d.get("config", d)).get("object", {})
            if "tilt_deg" in obj:
                return float(obj["tilt_deg"]), str(obj.get("tilt_axis", "X"))
        except Exception:
            continue
    return 0.0, "X"


def measure_run(run_dir, sensor="s1"):
    """Per-grasp blob axis + the angle the design says it should have.

    Reads ONLY this run's own folder: the maps from its CSVs, the pad roll
    from its pose_history (measured pad_actual_R, via stitching's own
    loader — not a second copy of that maths), and the rod tilt from its
    config. Returns (rows, meta)."""
    import stitching as ST
    files = sorted(glob.glob(os.path.join(run_dir,
                                          f"*_pt*_{sensor}_tactile_maps.csv")))
    tilt, axis = _scene_tilt(run_dir)
    bases = ST.load_pad_bases(run_dir)
    rows = []
    for f in files:
        tag = ST._pt_key(os.path.basename(f))
        if tag is None:
            continue
        m, _n, _peak = ST.hold_average(f)
        info = blob_axis(m)
        roll = ST.pad_roll_deg(bases.get(tag))
        exp = expected_pad_angle_deg(roll, tilt)
        rows.append({"grasp": tag, "map": m, "roll_deg": roll,
                     "expected_deg": exp,
                     "measured_deg": (info["angle_from_vertical_deg"]
                                      if info["ok"] else None),
                     "info": info})
    return rows, {"tilt_deg": tilt, "tilt_axis": axis, "sensor": sensor}


def blob_report(run_dir):
    """Human-readable per-grasp table, both sensors."""
    L = ["BLOB AXIS — contact orientation vs grid design",
         f"run: {os.path.abspath(run_dir)}",
         f"method: Paper-2 weighted PCA (virtual_search.generate_eigen_align)"
         f" — upsample x{UPSAMPLE}, threshold "
         f"{T_THRESHOLD*STRICTNESS:.4f} of range, pressure-weighted "
         f"covariance in mm"]
    any_rows = False
    for sensor in ("s1", "s2"):
        rows, meta = measure_run(run_dir, sensor)
        if not rows:
            continue
        any_rows = True
        L.append("")
        L.append(f"  {sensor}:  rod tilt {meta['tilt_deg']:+.1f} deg about "
                 f"{meta['tilt_axis']}")
        L.append("     grasp    roll    expected   measured    error"
                 "    elong  cells")
        errs = []
        for r in rows:
            if r["measured_deg"] is None:
                L.append(f"     {r['grasp']}   {r['roll_deg']:+6.2f}"
                         f"   {r['expected_deg']:+8.2f}        --   "
                         f"({r['info']['reason']})")
                continue
            e = r["measured_deg"] - r["expected_deg"]
            e = (e + 90.0) % 180.0 - 90.0
            errs.append(e)
            L.append(f"     {r['grasp']}   {r['roll_deg']:+6.2f}"
                     f"   {r['expected_deg']:+8.2f}"
                     f"   {r['measured_deg']:+8.2f}"
                     f"  {e:+7.2f}"
                     f"  {r['info']['elongation']:6.2f}"
                     f"  {r['info']['n_cells']:5d}")
        if errs:
            a = np.array(errs)
            L.append(f"     MEAN error {a.mean():+.2f} deg   "
                     f"spread (std) {a.std():.2f} deg   "
                     f"|max| {np.abs(a).max():.2f} deg")
    if not any_rows:
        return "no per-grasp tactile CSVs found in " + os.path.abspath(run_dir)
    L.append("")
    L.append("roll     = pad roll, MEASURED from pad_actual_R (0 = upright)")
    L.append("expected = roll - tilt: the rod's own direction seen from the")
    L.append("           rolled pad. 0 means the pad was rolled to follow the")
    L.append("           rod, so the contact line should run straight up the")
    L.append("           pad. See expected_pad_angle_deg for the derivation.")
    L.append("measured = weighted-PCA principal axis of the contact blob.")
    L.append("error    = measured - expected. A LARGE, CONSISTENT error is")
    L.append("           the finding: the pad is where it should be (checked")
    L.append("           separately at 0.007 deg), so the contact line is not")
    L.append("           leaning the way the geometry says it must.")
    L.append("elong    = major/minor axis ratio. Below ~1.5 the blob is round")
    L.append("           and its angle is not meaningful — read those rows as")
    L.append("           'no axis', not as a small angle.")
    return "\n".join(L)


def blob_figure(run_dir, out_dir=None):
    """One panel per grasp: the map, the MEASURED axis (solid) and the
    EXPECTED axis (dashed). Saves <run>/Stitched/blob_axis_<sensor>.png."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    out_dir = out_dir or os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for sensor in ("s1", "s2"):
        rows, meta = measure_run(run_dir, sensor)
        if not rows:
            continue
        n = len(rows)
        fig = Figure(figsize=(2.7 * n + 1.0, 4.2))
        FigureCanvasAgg(fig)
        vmax = max(float(np.max(r["map"])) for r in rows) or 1.0
        ext = (-PAD_W / 2, PAD_W / 2, -PAD_H / 2, PAD_H / 2)
        for i, r in enumerate(rows):
            ax = fig.add_subplot(1, n, i + 1)
            ax.imshow(_upsample(r["map"]), cmap="jet", origin="lower",
                      extent=ext, aspect="equal", vmin=0, vmax=vmax)
            cx, cy = (r["info"]["centroid_mm"] if r["info"]["ok"] else (0, 0))
            half = PAD_H / 2.0
            for ang, style, lbl in (
                    (r["expected_deg"], dict(ls="--", lw=1.8, color="w"),
                     "expected"),
                    (r["measured_deg"], dict(ls="-", lw=2.0, color="k"),
                     "measured")):
                if ang is None:
                    continue
                dx, dy = np.sin(np.radians(ang)), np.cos(np.radians(ang))
                ax.plot([cx - dx * half, cx + dx * half],
                        [cy - dy * half, cy + dy * half],
                        label=lbl, **style)
            ttl = f"{r['grasp']}  roll {r['roll_deg']:+.1f}\n"
            ttl += (f"exp {r['expected_deg']:+.1f}  meas "
                    + (f"{r['measured_deg']:+.1f}"
                       if r["measured_deg"] is not None else "--"))
            ax.set_title(ttl, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.legend(fontsize=6, loc="lower left")
        fig.suptitle(f"BLOB AXIS [{sensor}] — contact line vs grid design "
                     f"(rod tilt {meta['tilt_deg']:+.1f} deg about "
                     f"{meta['tilt_axis']}); black = measured, "
                     f"white dashed = expected", fontsize=10)
        p = os.path.join(out_dir, f"blob_axis_{sensor}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        made.append(p)
    return made


def blob_and_save(run_dir, want_figure=True):
    """Report + figures, saved beside the stitch outputs. Returns (text, pngs)."""
    txt = blob_report(run_dir)
    out_dir = os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "blob_axis_report.txt"), "w") as f:
        f.write(txt + "\n")
    pngs = blob_figure(run_dir, out_dir) if want_figure else []
    print(txt)
    print(f"\nsaved {os.path.join(out_dir, 'blob_axis_report.txt')}")
    for p in pngs:
        print(f"saved {p}")
    return txt, pngs


def metric_selftest(widths=(4.0, 6.0, 9.0),
                    angles=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45)):
    """What does this metric read on a PERFECT line contact at a known angle?

    Pre-empts the obvious objection to any tilt finding — "maybe your angle
    estimator is the problem". Synthesises an ideal straight contact line at
    70x40, averages it down to the real 7x4, and reports what blob_axis
    recovers. Any shortfall here is the 7x4 quantisation plus the pad being
    taller than wide, NOT the sensor and NOT the CNN, so it is the floor a
    real measurement has to be judged against."""
    L = ["BLOB-AXIS METRIC SELF-TEST — ideal straight line contact",
         "(synthetic, no sensor and no CNN involved: this is what the METRIC",
         " alone does to a perfectly straight contact of known angle)",
         "",
         "  true    " + "".join(f"  w={w:.0f}mm" for w in widths)]
    H, W = 700, 400
    y = (np.arange(H) - (H - 1) / 2.0) * (PAD_H / H)
    x = (np.arange(W) - (W - 1) / 2.0) * (PAD_W / W)
    X, Y = np.meshgrid(x, y)
    for t in angles:
        cells = []
        for w in widths:
            a = np.radians(t)
            d = X * np.cos(a) - Y * np.sin(a)
            m = (400.0 * np.exp(-(d / w) ** 2)).reshape(7, 100, 4, 100
                                                        ).mean(axis=(1, 3))
            r = blob_axis(m)
            cells.append(f"{r['angle_from_vertical_deg']:+8.2f}"
                         if r["ok"] else "      --")
        L.append(f"  {t:+5.1f}   " + "".join(cells))
    L.append("")
    L.append("Read this as the metric's own transfer curve. It is close to")
    L.append("unbiased below ~25 deg and compresses above that, so a measured")
    L.append("angle far BELOW these values cannot be blamed on the metric.")
    return "\n".join(L)


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    args, nums = [], []
    for a in sys.argv[1:]:
        if a.startswith("--"):
            continue
        try:
            nums.append(float(a))
        except ValueError:
            args.append(a)
    if not args:
        print(__doc__); sys.exit(1)
    if "--selftest" in sys.argv:
        print(metric_selftest()); sys.exit(0)
    if "--legacy" in sys.argv:
        tilt = nums[0] if nums else None
        for sen in ("s1", "s2"):
            out = run_folder(args[0], tilt, sen)
            if out:
                print(out); print()
    else:
        blob_and_save(args[0])
