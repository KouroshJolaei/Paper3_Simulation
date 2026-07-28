"""
validation.py — STITCH ROUND-TRIP FIDELITY CHECK (Block 2 self-consistency)

WHAT THIS MEASURES (read this before trusting the numbers)
-----------------------------------------------------------
For every grasp we already know:
  (a) its ORIGINAL hold-average 7x4 map (straight from the CSV), and
  (b) the exact world position its pad occupied (from pose_history).
We stitch all grasps onto the mm canvas, then SAMPLE THE CANVAS BACK at each
grasp's own 28 taxel positions to RECOVER a 7x4 map, and compare recovered
vs original with SSIM, Tactile-Centroid (TC) error, and (optionally) GSR
error.

So this validates the STITCHER AS A CONTAINER: how much do resampling +
overlap-averaging distort a single grasp? It is NOT a model-completion
metric. Expect high SSIM and small TC error, especially for the CENTER
grasp (whose INPUT is literally itself repainted). Grasps in heavy overlap
will differ the most, because averaging blends them with neighbours — i.e.
this is the per-grasp, metric-space view of the same thing `overlap sigma`
reports globally.

The REAL Paper-3 metric comparison (predicted extended map vs TARGET) lives
in Block 4, once the model exists. Keep these two uses separate when talking
to supervisors.

DEPENDS ON stitching.py — it reuses that module's own geometry
(_taxel_centers, load_offsets, _reanchor_to_gui, hold_average, CAL, pitches)
so the round-trip can never drift from the real paint step.

GSR: ported from get_gsr.py — model CNN_conv2d_norm3000_full.h5, input
resized to 9x9, divided by 3000. It is OPTIONAL: if TensorFlow or the model
file is missing, GSR columns read "n/a" and SSIM+TC still run.

Usage:
  python3 validation.py <run_dir> [res_mm]
or from the GUI Stitching tab button "Validate Stitch (SSIM/TC/GSR)".
"""

import os, sys, json
import numpy as np

# ---- import the stitcher's OWN geometry so we invert exactly what it paints
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import stitching as ST      # noqa: E402


# ------------------------------------------------------------------ GSR ----
# Optional. Uses the REAL Paper-2 loader/preprocessing by importing
# network_gsr.predict_grasp_success directly (no reimplementation -> cannot
# drift). GSR is a GRASP-LEVEL value: it needs s1 AND s2 together (56 values),
# so it is computed per grasp, not per sensor.
_GSR_FN = None
_GSR_TRIED = False
_NETWORK_GSR_DIR = "/home/kourosh/Pipeline_ws/ros2_ws/Python_Modules"


def _try_load_gsr():
    """Return the real predict_grasp_success callable, or None."""
    global _GSR_FN, _GSR_TRIED
    if _GSR_TRIED:
        return _GSR_FN
    _GSR_TRIED = True
    if os.path.isdir(_NETWORK_GSR_DIR) and _NETWORK_GSR_DIR not in sys.path:
        sys.path.insert(0, _NETWORK_GSR_DIR)
    try:
        from network_gsr import predict_grasp_success  # real Paper-2 module
        _GSR_FN = predict_grasp_success
        print("[validate] GSR enabled (network_gsr.predict_grasp_success)")
    except Exception as e:
        print(f"[validate] GSR disabled ({type(e).__name__}: {e})")
        _GSR_FN = None
    return _GSR_FN


def gsr_grasp(map7x4_s1, map7x4_s2, use_baseline=True):
    """Scalar GSR (%) for a grasp, from both sensors' 7x4 maps.
    Feeds a (1,56) DataFrame [s1 28 | s2 28] to the real Paper-2 pipeline."""
    fn = _try_load_gsr()
    if fn is None:
        return None
    try:
        import pandas as pd
        vec = np.concatenate([np.asarray(map7x4_s1, float).reshape(-1),
                              np.asarray(map7x4_s2, float).reshape(-1)])
        T = pd.DataFrame(vec.reshape(1, 56),
                         columns=[f"Column_{i+1}" for i in range(56)])
        out = fn(T, use_baseline=use_baseline)
        # predict_grasp_success returns e.g. ['87.12%']; be tolerant
        if isinstance(out, (list, tuple)):
            out = out[0]
        return float(str(out).replace("%", ""))
    except Exception as e:
        print(f"[validate] GSR predict failed ({e})")
        return None


# ------------------------------------------------------------- metrics ----
def tactile_centroid(map7x4):
    """Pressure-weighted centroid (row, col) in taxel units.
    Matches the Paper 1/2 'tactile centroid' definition:
    center of mass of the taxel grid weighted by (non-negative) pressure."""
    m = np.clip(np.asarray(map7x4, float), 0.0, None)
    tot = m.sum()
    if tot <= 1e-9:
        return None
    rr, cc = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    return (float((rr * m).sum() / tot), float((cc * m).sum() / tot))


def tc_error_mm(orig, recov):
    """Distance between the two maps' tactile centroids, in mm
    (using the real taxel pitch, so the number is physically meaningful)."""
    a = tactile_centroid(orig)
    b = tactile_centroid(recov)
    if a is None or b is None:
        return None
    dz = (a[0] - b[0]) * ST.PITCH_Z
    dy = (a[1] - b[1]) * ST.PITCH_Y
    return float(np.hypot(dy, dz))


def ssim(a, b):
    """SSIM between two small maps. Uses skimage if present, else a
    single-window Wang et al. fallback (fine for 7x4)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if rng <= 1e-9:
        return 1.0
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(a, b, data_range=rng,
                             win_size=min(7, a.shape[0]
                                          if a.shape[0] % 2 else a.shape[0]-1)))
    except Exception:
        # global single-window SSIM fallback
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        c1 = (0.01 * rng) ** 2; c2 = (0.03 * rng) ** 2
        return float(((2*mu_a*mu_b + c1) * (2*cov + c2)) /
                     ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2)))


# ------------------------------------------------ round-trip sampler ----
def _sample_canvas(canvas, extent, res_mm, oy, oz, tax):
    """Recover a 7x4 by reading the canvas back at each taxel's world
    position, using the SAME block footprint the stitcher painted with
    (_bw x _bh cells, round-to-cell) — so a no-overlap grasp round-trips
    to itself exactly."""
    y0, y1, z0, z1 = extent
    nz, ny = canvas.shape
    bw = max(1, int(round(ST.PITCH_Y / res_mm)))
    bh = max(1, int(round(ST.PITCH_Z / res_mm)))
    out = np.zeros((ST.N_ROWS, ST.N_COLS))
    for r in range(ST.N_ROWS):
        for c in range(ST.N_COLS):
            ty = oy + tax[r, c, 0]
            tz = oz + tax[r, c, 1]
            icy = int(round((ty - y0) / res_mm))
            icz = int(round((tz - z0) / res_mm))
            iy0 = max(icy - bw // 2, 0); iy1 = min(iy0 + bw, ny)
            iz0 = max(icz - bh // 2, 0); iz1 = min(iz0 + bh, nz)
            if iy1 > iy0 and iz1 > iz0:
                block = canvas[iz0:iz1, iy0:iy1]
                out[r, c] = float(block.mean())
    return out


def validate_run(run_dir, res_mm=1.0, want_gsr=True):
    """Round-trip every grasp of both sensors; return a results dict.
    SSIM and TC are per-sensor; GSR is per-grasp (needs s1+s2 together)."""
    results = {"run_dir": os.path.abspath(run_dir), "res_mm": res_mm,
               "sensors": {}}
    gsr_on = want_gsr and (_try_load_gsr() is not None)

    # ---- pass 1: per-sensor round-trip, keep recovered maps for GSR pairing
    recov_maps = {"s1": {}, "s2": {}}     # ptNN -> recovered 7x4
    orig_maps = {"s1": {}, "s2": {}}      # ptNN -> original  7x4
    for sensor in ("s1", "s2"):
        try:
            res = ST.build_canvas(run_dir, sensor, res_mm, verbose=False)
        except RuntimeError as e:
            results["sensors"][sensor] = {"error": str(e)}
            continue
        canvas, extent = res["canvas"], res["extent"]
        tax = ST._taxel_centers(ST.CAL[sensor])
        rows = []
        for i, (key, (oy, oz), orig) in enumerate(res["grasps"]):
            recov = _sample_canvas(canvas, extent, res_mm, oy, oz, tax)
            orig_maps[sensor][key] = orig
            recov_maps[sensor][key] = recov
            rows.append({"grasp": key,
                         "is_center": (i == res["center_index"]),
                         "ssim": ssim(orig, recov),
                         "tc_err_mm": tc_error_mm(orig, recov)})
        ssims = [r["ssim"] for r in rows if r["ssim"] is not None]
        tcs = [r["tc_err_mm"] for r in rows if r["tc_err_mm"] is not None]
        results["sensors"][sensor] = {
            "rows": rows,
            "summary": {"n": len(rows),
                        "ssim_mean": float(np.mean(ssims)) if ssims else None,
                        "tc_err_mean_mm": float(np.mean(tcs)) if tcs else None,
                        "overlap_sigma": res["overlap_std"]}}

    # ---- pass 2: grasp-level GSR (original s1+s2 vs recovered s1+s2)
    if gsr_on:
        common = sorted(set(orig_maps["s1"]) & set(orig_maps["s2"]))
        gsr_rows = []
        for key in common:
            g0 = gsr_grasp(orig_maps["s1"][key], orig_maps["s2"][key])
            g1 = gsr_grasp(recov_maps["s1"][key], recov_maps["s2"][key])
            gsr_rows.append({"grasp": key, "gsr_orig": g0, "gsr_recov": g1,
                             "gsr_err": (None if (g0 is None or g1 is None)
                                         else abs(g0 - g1))})
        gerrs = [r["gsr_err"] for r in gsr_rows if r["gsr_err"] is not None]
        results["gsr"] = {"rows": gsr_rows,
                          "gsr_err_mean": (float(np.mean(gerrs))
                                           if gerrs else None)}
    return results


def format_report(results):
    """Human-readable table (also what the GUI prints)."""
    L = []
    L.append(f"STITCH ROUND-TRIP VALIDATION  ({results['res_mm']} mm/cell)")
    L.append(f"run: {results['run_dir']}")
    L.append("(validates the STITCHER as a container — not model completion; "
             "high SSIM / low TC error is expected, esp. the center grasp)")
    for sensor, blk in results["sensors"].items():
        L.append("")
        if "error" in blk:
            L.append(f"  {sensor}: {blk['error']}")
            continue
        rows, s = blk["rows"], blk["summary"]
        L.append(f"  {sensor}:  grasp     SSIM    TC_err(mm)")
        for r in rows:
            tag = "*" if r["is_center"] else " "
            tc = r["tc_err_mm"] if r["tc_err_mm"] is not None else -1
            L.append(f"     {tag}{r['grasp']}    {r['ssim']:.3f}     {tc:5.2f}")
        L.append(f"     SUMMARY  SSIM={s['ssim_mean']:.3f}   "
                 f"TC_err={s['tc_err_mean_mm']:.2f} mm   "
                 f"(overlap sigma={s['overlap_sigma']:.0f})")

    gsr = results.get("gsr")
    if gsr and gsr["rows"]:
        L.append("")
        L.append("  GSR (grasp-level, s1+s2 together; original vs recovered):")
        L.append("     grasp   GSR_orig   GSR_recov   GSR_err")
        for r in gsr["rows"]:
            go, gr, ge = r["gsr_orig"], r["gsr_recov"], r["gsr_err"]
            L.append(f"     {r['grasp']}   "
                     f"{('%6.2f%%'%go) if go is not None else '   n/a':>8}   "
                     f"{('%6.2f%%'%gr) if gr is not None else '   n/a':>8}   "
                     f"{('%5.2f'%ge) if ge is not None else ' n/a':>6}")
        if gsr["gsr_err_mean"] is not None:
            L.append(f"     SUMMARY  mean GSR_err = {gsr['gsr_err_mean']:.2f} "
                     f"percentage points")

    L.append("")
    L.append("* = center grasp (its INPUT is itself; SSIM should be ~1.0)")
    return "\n".join(L)


def validate_and_save(run_dir, res_mm=1.0, want_gsr=True):
    results = validate_run(run_dir, res_mm, want_gsr=want_gsr)
    report = format_report(results)
    out_dir = os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, "validation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(report)
    print(f"\nsaved {os.path.join(out_dir, 'validation_report.txt')}")
    return results, report


if __name__ == "__main__":
    rd = os.path.expanduser("~/Paper3_Simulation/Data/gui_run")
    rr = 1.0
    import glob as _glob
    for a in sys.argv[1:]:
        try:
            rr = float(a)
        except ValueError:
            rd = a
    if not _glob.glob(os.path.join(rd, "*_s1_tactile_maps.csv")):
        runs = sorted(_glob.glob(os.path.join(rd, "run_*")))
        if runs:
            rd = runs[-1]
    validate_and_save(rd, rr)
