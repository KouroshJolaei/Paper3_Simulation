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
# WHAT ANGLE SHOULD THE CONTACT BLOB HAVE?   (rewritten 2026-08-05)
#
# The first version computed the angle of the infinite contact LINE as
# roll - tilt. That is wrong twice over.
#
# 1. SIGN. blob_axis's x runs with the COLUMN INDEX, but _taxel_centers puts
#    column 0 at the LARGEST across-coordinate (+8.25 mm) and column 3 at the
#    smallest (-8.25). So blob_axis's x is MINUS the pad's across axis, and
#    any angle derived by hand in the (across, up) basis comes out negated.
#
# 2. THE LINE IS NOT THE BLOB. What the taxels see is the line CLIPPED by the
#    22x37 window. Slide the pad sideways and a tilted band runs off the edge,
#    so the pad sees a short, stubby piece of it whose principal axis is NOT
#    the line's angle — at the extreme it is nearly perpendicular to it. The
#    offsets therefore change the expected reading even though they never
#    change the true contact direction. (Kourosh's point, 2026-08-05.)
#
# So the expected angle is now MEASURED, not derived: build the contact patch
# the geometry implies, average it onto the real 7x4 taxel grid, and run the
# SAME blob_axis on it. Nothing is hand-transformed, so the sign cannot be
# got wrong, and the answer automatically carries the same 7x4 quantisation
# the real measurement has (see metric_selftest: a true 35 deg line reads
# ~31 deg). What is left in the error column is then neither geometry nor
# metric.
#
# THE MODEL
#   A flat pad pressing a cylinder touches along the generatrix nearest the
#   pad. Projected into the pad-facing (Y, Z) plane that line coincides with
#   the cylinder's own axis, so:
#       point     P0 = (cy, cz)          object centre
#       direction e  = (-sin T, cos T)   T = tilt about world X
#       normal    n  = ( cos T, sin T)
#   Contact is not a mathematical line: the rubber flattens, giving a BAND of
#   width w. Pressure across it is taken as Hertzian for a line contact,
#       p(r) = sqrt(1 - (2r/w)^2),
#   which is peaked at the line and falls to zero at the band edge. w is the
#   one free parameter — measure it per run with fit_band_width().
BAND_WIDTH_MM = 8.0      # default; fit_band_width() measures it from a run
SUBSAMPLE = 25           # sample points per taxel per axis when rasterising


def expected_patch_map(pad_yz, basis, scene, band_width_mm=BAND_WIDTH_MM,
                       cal=None, subsample=SUBSAMPLE):
    """The 7x4 a PERFECT sensor would report for this pad pose and object.

    pad_yz : (y_mm, z_mm) pad-face centre in world/GUI coordinates
    basis  : ((ay, az), (uy, uz)) measured pad axes, or None for upright
    scene  : dict from stitching._load_scene (cy, cz, tilt, axis, d, L)

    Every taxel is integrated over its own footprint on the SAME lattice
    stitching uses (_taxel_centers, so CAL flips are honoured), which is why
    no coordinate has to be transformed by hand anywhere in this file."""
    import stitching as ST
    cal = cal if cal is not None else ST.CAL["s1"]
    a, u = (ST.FLAT_BASIS if ST.is_flat(basis)
            else (np.asarray(basis[0], float), np.asarray(basis[1], float)))
    tax = ST._taxel_centers(cal)                    # (7,4,2) pad-local mm

    T = np.radians(float(scene.get("tilt", 0.0))
                   if str(scene.get("axis", "X")).upper() == "X" else 0.0)
    e = np.array([-np.sin(T), np.cos(T)])           # rod axis, world (Y, Z)
    n = np.array([np.cos(T), np.sin(T)])            # across the rod
    P0 = np.array([float(scene["cy"]), float(scene["cz"])])
    half_w = max(float(band_width_mm), 1e-6) / 2.0
    half_L = float(scene.get("L", 1e6)) / 2.0

    s = (np.arange(subsample) + 0.5) / subsample - 0.5      # -0.5 .. +0.5
    dA = s * PITCH_Y                                        # within a taxel
    dU = s * PITCH_Z
    A = tax[:, :, 0][:, :, None, None] + dA[None, None, None, :]
    U = tax[:, :, 1][:, :, None, None] + dU[None, None, :, None]

    C = np.asarray(pad_yz, float)
    Wy = C[0] + A * a[0] + U * u[0]                 # world Y of each sample
    Wz = C[1] + A * a[1] + U * u[1]                 # world Z of each sample
    r = (Wy - P0[0]) * n[0] + (Wz - P0[1]) * n[1]   # distance across the rod
    sA = (Wy - P0[0]) * e[0] + (Wz - P0[1]) * e[1]  # distance along the rod

    q = 1.0 - (r / half_w) ** 2
    p = np.where((q > 0) & (np.abs(sA) <= half_L), np.sqrt(np.clip(q, 0, None)),
                 0.0)
    m = p.mean(axis=(2, 3))                          # integrate each taxel
    if m.max() > 0:
        m = m / m.max() * 1000.0                     # arbitrary units
    return m


def expected_from_geometry(pad_yz, basis, scene, band_width_mm=BAND_WIDTH_MM,
                           cal=None):
    """Expected blob axis, via the patch the geometry implies.
    Returns the blob_axis dict with 'patch' and 'contact_frac' added."""
    m = expected_patch_map(pad_yz, basis, scene, band_width_mm, cal)
    info = blob_axis(m)
    info["patch"] = m
    info["contact_frac"] = float((m > 0).sum()) / m.size
    return info


def line_angle_deg(basis, scene):
    """Angle of the INFINITE contact line in blob_axis's frame, degrees.

    Kept as a reference column: comparing it with the patch angle shows how
    much of the expected value is set by clipping rather than by the rod.
    Derived numerically from two points on the rod axis, projected onto the
    same axes expected_patch_map uses, so it shares that function's
    conventions instead of re-deriving them."""
    import stitching as ST
    a, u = (ST.FLAT_BASIS if ST.is_flat(basis)
            else (np.asarray(basis[0], float), np.asarray(basis[1], float)))
    T = np.radians(float(scene.get("tilt", 0.0))
                   if str(scene.get("axis", "X")).upper() == "X" else 0.0)
    e = np.array([-np.sin(T), np.cos(T)])
    # blob_axis x runs with COLUMN INDEX, and _taxel_centers has the across-
    # coordinate DECREASING with column index -> x_blob = -(across).
    x_b = -float(e @ a)
    y_b = float(e @ u)
    ang = np.degrees(np.arctan2(x_b, y_b))
    return float((ang + 90.0) % 180.0 - 90.0)


def _across_profile(m, basis=None):
    """Collapse a map to its ACROSS-pad profile (one value per column).

    The band width only controls how pressure spreads ACROSS the pad. Real
    maps also vary strongly ALONG the rod — the 2026-08-06 upright run peaks
    in the middle rows and fades at rows 0 and 6, which the uniform-line
    model does not reproduce. Comparing full maps lets that vertical
    mismatch dominate the residual and swamp the width signal (it left the
    error curve flat from 2 to 11 mm). Collapsing along rows removes it."""
    p = np.asarray(m, float).mean(axis=0)
    return p / p.max() if p.max() > 0 else p


def fit_band_width(run_dir, sensor="s1", widths=None, verbose=True):
    """Measure the contact band width from the run's OWN maps.

    Sweeps w, generates the synthetic patch for each grasp's true pose, and
    keeps the w whose ACROSS-pad profile best matches the measured one.
    Best done on an UPRIGHT run, where the band is vertical and unclipped, so
    the fit is about width and nothing else. Returns (w_mm, table).

    Only 4 columns carry the profile, so this is inherently coarse — read
    the residual and the curvature, not just the argmin."""
    import stitching as ST
    scene = ST._load_scene(run_dir, verbose=False)
    if scene is None:
        return None, "no object geometry in this run's config"
    offs, _src = ST.load_offsets(run_dir)
    bases = ST.load_pad_bases(run_dir)
    files = sorted(glob.glob(os.path.join(run_dir,
                                          f"*_pt*_{sensor}_tactile_maps.csv")))
    obs = []
    for f in files:
        tag = ST._pt_key(os.path.basename(f))
        if tag is None or tag not in offs:
            continue
        m, _n, _p = ST.hold_average(f)
        if m.max() <= 0:
            continue
        obs.append((tag, offs[tag], bases.get(tag), _across_profile(m)))
    if not obs:
        return None, "no usable grasps"
    widths = widths if widths is not None else np.arange(2.0, 22.1, 0.5)
    rows = []
    for w in widths:
        err = 0.0
        for _tag, pyz, b, po in obs:
            ms = expected_patch_map(pyz, b, scene, w, ST.CAL[sensor])
            err += float(np.mean((_across_profile(ms) - po) ** 2))
        rows.append((float(w), err / len(obs)))
    best, best_err = min(rows, key=lambda t: t[1])
    worst_err = max(e for _w, e in rows)
    L = [f"BAND WIDTH FIT ({sensor}) — {len(obs)} grasp(s), "
         f"rod tilt {scene['tilt']:+.1f} deg, diameter {scene['d']:.0f} mm",
         "fitted on the ACROSS-pad profile (rows collapsed), so the fit sees "
         "only what the width controls",
         "  w(mm)   mean sq err (peak-normalised profile)"]
    for w, e in rows:
        L.append(f"  {w:5.1f}   {e:.5f}" + ("   <-- best" if w == best else ""))
    L.append(f"  -> band width {best:.1f} mm      "
             f"(residual {best_err:.5f}; equivalent indentation "
             f"{_depth_for_width(best, scene['d']):.2f} mm on this rod)")
    # honesty guards: say when the answer should not be trusted
    if best_err > 0.01:
        L.append("  !! RESIDUAL IS LARGE — the model does not match these maps")
        L.append("  !! at ANY width, so this is the least-bad fit rather than")
        L.append("  !! a measurement. Treat the number as indicative only.")
    if worst_err > 0 and (worst_err - best_err) / worst_err < 0.25:
        L.append("  !! ERROR CURVE IS FLAT — the data barely distinguishes one")
        L.append("  !! width from another, so the argmin is weakly determined.")
    L.append("  Fit on an UPRIGHT run for a clean answer; on a rolled or")
    L.append("  tilted run the clipping also moves and the fit is muddier.")
    txt = "\n".join(L)
    if verbose:
        print(txt)
    return best, txt


def _depth_for_width(w_mm, diameter_mm):
    """Indentation depth implied by a contact chord of width w on a cylinder:
    w = 2*sqrt(2Rd - d^2). A quick physical plausibility check on the fit."""
    R = max(float(diameter_mm), 1e-6) / 2.0
    h = min(float(w_mm) / 2.0, R)
    return float(R - np.sqrt(max(R * R - h * h, 0.0)))


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


def measure_run(run_dir, sensor="s1", band_width_mm=BAND_WIDTH_MM):
    """Per-grasp blob axis + the angle the geometry implies for that grasp.

    Reads ONLY this run's own folder: maps from its CSVs, pad pose and roll
    from its pose_history (measured), object geometry from its config."""
    import stitching as ST
    files = sorted(glob.glob(os.path.join(run_dir,
                                          f"*_pt*_{sensor}_tactile_maps.csv")))
    tilt, axis = _scene_tilt(run_dir)
    scene = ST._load_scene(run_dir, verbose=False)
    bases = ST.load_pad_bases(run_dir)
    try:
        offs, _src = ST.load_offsets(run_dir)
    except Exception:
        offs = {}
    rows = []
    for f in files:
        tag = ST._pt_key(os.path.basename(f))
        if tag is None:
            continue
        m, _n, _peak = ST.hold_average(f)
        info = blob_axis(m)
        basis = bases.get(tag)
        roll = ST.pad_roll_deg(basis)
        exp, expinfo, lang = None, None, None
        if scene is not None and tag in offs:
            expinfo = expected_from_geometry(offs[tag], basis, scene,
                                             band_width_mm, ST.CAL[sensor])
            exp = (expinfo["angle_from_vertical_deg"]
                   if expinfo["ok"] else None)
            lang = line_angle_deg(basis, scene)
        rows.append({"grasp": tag, "map": m, "roll_deg": roll,
                     "pad_yz": offs.get(tag),
                     "expected_deg": exp, "line_deg": lang,
                     "expected_info": expinfo,
                     "measured_deg": (info["angle_from_vertical_deg"]
                                      if info["ok"] else None),
                     "info": info})
    return rows, {"tilt_deg": tilt, "tilt_axis": axis, "sensor": sensor,
                  "scene": scene, "band_width_mm": band_width_mm}


def blob_report(run_dir, band_width_mm=BAND_WIDTH_MM):
    """Human-readable per-grasp table, both sensors."""
    L = ["BLOB AXIS — contact orientation vs grid design",
         f"run: {os.path.abspath(run_dir)}",
         f"method: Paper-2 weighted PCA (virtual_search.generate_eigen_align)"
         f" — upsample x{UPSAMPLE}, threshold "
         f"{T_THRESHOLD*STRICTNESS:.4f} of range, pressure-weighted "
         f"covariance in mm",
         f"expected: contact band of width {band_width_mm:.1f} mm, clipped by "
         f"the pad window, put through the SAME estimator"]
    any_rows = False
    for sensor in ("s1", "s2"):
        rows, meta = measure_run(run_dir, sensor, band_width_mm)
        if not rows:
            continue
        any_rows = True
        L.append("")
        L.append(f"  {sensor}:  rod tilt {meta['tilt_deg']:+.1f} deg about "
                 f"{meta['tilt_axis']}")
        L.append("     grasp    roll    line   expected   measured    error"
                 "    elong  exp_el  cover")
        errs = []
        for r in rows:
            ln = ("     --" if r["line_deg"] is None
                  else f"{r['line_deg']:+7.2f}")
            ex = ("      --" if r["expected_deg"] is None
                  else f"{r['expected_deg']:+8.2f}")
            xe = r["expected_info"]
            exel = ("    --" if xe is None or not xe["ok"]
                    else f"{xe['elongation']:6.2f}")
            cov = ("    --" if xe is None
                   else f"{100*xe['contact_frac']:5.0f}%")
            if r["measured_deg"] is None:
                L.append(f"     {r['grasp']}   {r['roll_deg']:+6.2f} {ln} {ex}"
                         f"        --          ({r['info']['reason']})")
                continue
            if r["expected_deg"] is None:
                L.append(f"     {r['grasp']}   {r['roll_deg']:+6.2f} {ln} {ex}"
                         f"   {r['measured_deg']:+8.2f}       --"
                         f"  {r['info']['elongation']:6.2f} {exel} {cov}")
                continue
            e = r["measured_deg"] - r["expected_deg"]
            e = (e + 90.0) % 180.0 - 90.0
            errs.append(e)
            L.append(f"     {r['grasp']}   {r['roll_deg']:+6.2f} {ln} {ex}"
                     f"   {r['measured_deg']:+8.2f}  {e:+7.2f}"
                     f"  {r['info']['elongation']:6.2f} {exel} {cov}")
        if errs:
            a = np.array(errs)
            L.append(f"     MEAN error {a.mean():+.2f} deg   "
                     f"spread (std) {a.std():.2f} deg   "
                     f"|max| {np.abs(a).max():.2f} deg")
    if not any_rows:
        return "no per-grasp tactile CSVs found in " + os.path.abspath(run_dir)
    L.append("")
    L.append("roll     = pad roll, MEASURED from pad_actual_R (0 = upright)")
    L.append("line     = angle of the infinite contact line. Depends only on")
    L.append("           rod tilt and pad roll, NOT on where the pad sits.")
    L.append("expected = angle of that line CLIPPED to the pad window and put")
    L.append("           through the same 7x4 estimator. This is what a")
    L.append("           perfect sensor would report at this pose. It differs")
    L.append("           from `line` exactly when the offset pushes the band")
    L.append("           off-centre so only a short piece of it is seen.")
    L.append("measured = weighted-PCA principal axis of the REAL contact blob.")
    L.append("error    = measured - expected, with geometry AND the estimator")
    L.append("           already accounted for. What is left is the sensor.")
    L.append("elong    = major/minor of the real blob; exp_el the same for the")
    L.append("           expected patch. Below ~1.5 a blob is round and its")
    L.append("           angle is not meaningful. If exp_el is low, the")
    L.append("           GEOMETRY makes this pose uninformative — that is a")
    L.append("           grid-design problem, not a sensor result.")
    L.append("cover    = fraction of the pad the band covers.")
    return "\n".join(L)


def blob_figure(run_dir, out_dir=None, band_width_mm=BAND_WIDTH_MM):
    """Per grasp: the real map with the MEASURED axis, beside the expected
    patch with the EXPECTED axis. Saves Stitched/blob_axis_<sensor>.png."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    out_dir = out_dir or os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for sensor in ("s1", "s2"):
        rows, meta = measure_run(run_dir, sensor, band_width_mm)
        if not rows:
            continue
        n = len(rows)
        fig = Figure(figsize=(2.6 * n + 1.0, 6.6))
        FigureCanvasAgg(fig)
        ext = (-PAD_W / 2, PAD_W / 2, -PAD_H / 2, PAD_H / 2)
        vmax = max(float(np.max(r["map"])) for r in rows) or 1.0

        def _axline(ax, ang, cen, **kw):
            if ang is None:
                return
            dx, dy = np.sin(np.radians(ang)), np.cos(np.radians(ang))
            h = PAD_H / 2.0
            ax.plot([cen[0] - dx * h, cen[0] + dx * h],
                    [cen[1] - dy * h, cen[1] + dy * h], **kw)

        for i, r in enumerate(rows):
            # top: what the geometry says the patch should be
            ax = fig.add_subplot(2, n, i + 1)
            xe = r["expected_info"]
            if xe is not None:
                ax.imshow(_upsample(xe["patch"]), cmap="jet", origin="lower",
                          extent=ext, aspect="equal")
                _axline(ax, r["expected_deg"],
                        xe["centroid_mm"] if xe["ok"] else (0, 0),
                        ls="--", lw=2.0, color="w")
                _axline(ax, r["line_deg"], (0, 0), ls=":", lw=1.2, color="k")
                ax.set_title(f"{r['grasp']}  EXPECTED\nline "
                             f"{r['line_deg']:+.1f}  patch "
                             + (f"{r['expected_deg']:+.1f}"
                                if r["expected_deg"] is not None else "--"),
                             fontsize=8)
            else:
                ax.set_title(f"{r['grasp']}  EXPECTED\n(no object geometry)",
                             fontsize=8)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])

            # bottom: what the sensor actually reported
            ax = fig.add_subplot(2, n, n + i + 1)
            ax.imshow(_upsample(r["map"]), cmap="jet", origin="lower",
                      extent=ext, aspect="equal", vmin=0, vmax=vmax)
            _axline(ax, r["measured_deg"],
                    r["info"]["centroid_mm"] if r["info"]["ok"] else (0, 0),
                    ls="-", lw=2.0, color="k")
            _axline(ax, r["expected_deg"],
                    r["info"]["centroid_mm"] if r["info"]["ok"] else (0, 0),
                    ls="--", lw=1.6, color="w")
            err = (None if (r["measured_deg"] is None
                            or r["expected_deg"] is None)
                   else (r["measured_deg"] - r["expected_deg"] + 90) % 180 - 90)
            ax.set_title("MEASURED " + (f"{r['measured_deg']:+.1f}"
                                        if r["measured_deg"] is not None
                                        else "--")
                         + (f"\nerror {err:+.1f}" if err is not None else "\n"),
                         fontsize=8)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"BLOB AXIS [{sensor}] — rod tilt "
                     f"{meta['tilt_deg']:+.1f} deg about {meta['tilt_axis']}, "
                     f"band {band_width_mm:.1f} mm.  TOP = expected patch "
                     f"(white dashed = its axis, dotted = infinite line);  "
                     f"BOTTOM = measured (black = its axis)", fontsize=10)
        p = os.path.join(out_dir, f"blob_axis_{sensor}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        made.append(p)
    return made


def blob_and_save(run_dir, want_figure=True, band_width_mm=BAND_WIDTH_MM,
                  fit_width=False):
    """Report + figures, saved beside the stitch outputs.
    fit_width=True measures the band width from this run first."""
    extra = ""
    if fit_width:
        w, tbl = fit_band_width(run_dir, "s1", verbose=False)
        if w:
            band_width_mm = w
            extra = "\n\n" + tbl
        else:
            extra = "\n\n[band width fit failed: " + str(tbl) + "]"
    txt = blob_report(run_dir, band_width_mm) + extra
    out_dir = os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "blob_axis_report.txt"), "w") as f:
        f.write(txt + "\n")
    pngs = blob_figure(run_dir, out_dir, band_width_mm) if want_figure else []
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
