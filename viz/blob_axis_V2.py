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


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    args, nums = [], []
    for a in sys.argv[1:]:
        try:
            nums.append(float(a))
        except ValueError:
            args.append(a)
    if not args:
        print(__doc__); sys.exit(1)
    tilt = nums[0] if nums else None
    for sen in ("s1", "s2"):
        out = run_folder(args[0], tilt, sen)
        if out:
            print(out); print()
