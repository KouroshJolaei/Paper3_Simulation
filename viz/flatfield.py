"""
flatfield.py — remove the hinge row-gradient, then MEASURE the tilt.

THE ARGUMENT
------------
The gripper's hinged fingers press hardest near the pad's upper rows (5-6)
on EVERY grasp, wherever the pad is. Stitching many pads at different
heights smears those horizontal bright bars into one vertical block, which
is brighter than the real signal — so a tilted cylinder's diagonal contact
line gets buried and the stitched map looks vertical.

On a STRAIGHT (tilt = 0) cylinder the true contact is uniform along Z: a
plain vertical stripe. Therefore ANY row-to-row structure in the average of
that run's pad-frame maps is instrument artifact, not object. That average
is our correction map — no flat plate and no contact-aware closure needed.

WHAT IS CORRECTED, AND WHAT IS NOT
----------------------------------
ROWS (Z, along the pad's long axis) -> corrected. Artifact.
COLS (Y, across the pad)            -> LEFT ALONE. On a round rod the
    middle columns genuinely touch harder than the edges; that curvature
    is real object information and must survive.

So the gain is a 7-vector, not a 7x4 map. Deliberately conservative.

MEASURING THE TILT
------------------
For each Z row of the stitched canvas, take the pressure-weighted Y
centroid. Those centroids should lie on a straight line whose slope is
    dY/dZ = -tan(tilt)
(the cylinder axis leans by exactly that much). Fitting the line gives a
NUMBER to compare against the commanded tilt, instead of squinting at a
colour map.

Usage:
  python3 flatfield.py <flat_tilt0_run> <tilted_run> <tilt_deg> [res_mm]
e.g.
  python3 flatfield.py .../run_20260729_132515 .../run_20260729_160824 20 0.75
"""

import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import stitching as ST      # noqa: E402


# --------------------------------------------------------------- gain ----
def build_row_gain(run_dir, sensor, verbose=True):
    """7-vector row gain from a STRAIGHT-cylinder run, normalised to mean 1.

    Averages every grasp's hold-average 7x4, then collapses across columns.
    Returns (gain7, info)."""
    res = ST.build_canvas(run_dir, sensor, 1.0, verbose=False)
    maps = np.stack([g[2] for g in res["grasps"]], axis=0)   # (n,7,4)
    mean_map = maps.mean(axis=0)
    prof = mean_map.mean(axis=1)                             # (7,) row means
    if not np.all(np.isfinite(prof)) or prof.mean() <= 0:
        raise RuntimeError(f"{sensor}: cannot build gain (empty maps?)")
    gain = prof / prof.mean()
    gain = np.clip(gain, 0.15, 6.0)        # never amplify a dead row wildly
    info = {"sensor": sensor, "n_grasps": int(maps.shape[0]),
            "row_profile": [float(x) for x in prof],
            "gain": [float(x) for x in gain],
            "spread_pct": float(100.0 * (prof.max() - prof.min()) / prof.mean())}
    if verbose:
        print(f"[flatfield] {sensor}: {maps.shape[0]} grasps, row profile "
              f"(row0=bottom) = " + " ".join(f"{x:6.0f}" for x in prof))
        print(f"[flatfield] {sensor}: gain            = "
              + " ".join(f"{x:6.3f}" for x in gain)
              + f"   spread {info['spread_pct']:.0f}% of mean")
    return gain, info


def save_gain(path, gains):
    with open(path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in gains.items()},
                  f, indent=2)
    print(f"[flatfield] saved {path}")


def load_gain(path):
    with open(path) as f:
        return {k: np.asarray(v, float) for k, v in json.load(f).items()}


# ------------------------------------------------- corrected stitching ----
class row_gain_applied:
    """Context manager: make ST.hold_average divide by the row gain, so
    build_canvas paints CORRECTED maps without editing stitching.py."""

    def __init__(self, gains):
        self.gains = gains
        self._orig = None

    def __enter__(self):
        self._orig = ST.hold_average
        gains = self.gains

        def patched(csv_path):
            m, nfr, peak = self._orig(csv_path)
            base = os.path.basename(csv_path)
            sensor = "s2" if "_s2_" in base else "s1"
            g = gains.get(sensor)
            if g is None or m.shape[0] != g.shape[0]:
                return m, nfr, peak
            return m / g[:, None], nfr, peak

        ST.hold_average = patched
        return self

    def __exit__(self, *exc):
        ST.hold_average = self._orig
        return False


# ------------------------------------------------------ tilt measurement ----
def measure_tilt_slope(canvas, extent, res_mm, frac=0.25):
    """Fit dY/dZ through the per-Z-row pressure-weighted Y centroids.

    Rows whose total pressure is below `frac` x the strongest row are
    skipped (they are noise, and their centroid is meaningless).
    Returns (slope, r2, n_rows_used, angle_deg)."""
    y0, y1, z0, z1 = extent
    nz, ny = canvas.shape
    yy = y0 + np.arange(ny) * res_mm
    zz = z0 + np.arange(nz) * res_mm
    w = np.clip(canvas, 0.0, None)
    tot = w.sum(axis=1)
    if tot.max() <= 0:
        return None, None, 0, None
    keep = tot >= frac * tot.max()
    if keep.sum() < 3:
        return None, None, int(keep.sum()), None
    cen = (w[keep] * yy[None, :]).sum(axis=1) / tot[keep]
    zs = zz[keep]
    A = np.vstack([zs, np.ones_like(zs)]).T
    coef, *_ = np.linalg.lstsq(A, cen, rcond=None)
    slope = float(coef[0])
    pred = A @ coef
    ss_res = float(((cen - pred) ** 2).sum())
    ss_tot = float(((cen - cen.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, r2, int(keep.sum()), float(np.degrees(np.arctan(slope)))


# ------------------------------------------------------------- driver ----
def compare(flat_run, tilt_run, tilt_deg, res_mm=0.75):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    expect = np.tan(np.radians(float(tilt_deg)))
    L = [f"FLAT-FIELD + TILT MEASUREMENT",
         f"  gain from : {os.path.basename(os.path.abspath(flat_run))} "
         f"(must be a STRAIGHT, tilt=0 run)",
         f"  tested on : {os.path.basename(os.path.abspath(tilt_run))} "
         f"(tilt = {tilt_deg} deg)",
         f"  expected |dY/dZ| = tan({tilt_deg} deg) = {expect:.3f}",
         ""]

    gains, panels = {}, {}
    for sensor in ("s1", "s2"):
        g, info = build_row_gain(flat_run, sensor)
        gains[sensor] = g
        L.append(f"  {sensor} row gain spread = {info['spread_pct']:.0f}% "
                 f"of mean over {info['n_grasps']} grasps")

    L.append("")
    for sensor in ("s1", "s2"):
        raw = ST.build_canvas(tilt_run, sensor, res_mm, verbose=False)
        with row_gain_applied(gains):
            cor = ST.build_canvas(tilt_run, sensor, res_mm, verbose=False)
        s_raw = measure_tilt_slope(raw["canvas"], raw["extent"], res_mm)
        s_cor = measure_tilt_slope(cor["canvas"], cor["extent"], res_mm)
        panels[sensor] = (raw, cor, s_raw, s_cor)
        L.append(f"  {sensor}:")
        for tag, s in (("raw      ", s_raw), ("corrected", s_cor)):
            if s[0] is None:
                L.append(f"     {tag}  no measurable band")
                continue
            L.append(f"     {tag}  dY/dZ = {s[0]:+.3f}  "
                     f"(|{abs(s[0]):.3f}| vs {expect:.3f} expected, "
                     f"{100*abs(s[0])/expect:5.0f}%)   "
                     f"angle {abs(s[3]):5.1f} deg   R2 = {s[1]:.3f}   "
                     f"rows {s[2]}")

    # ---- figure: raw | corrected, per sensor, with the fitted line
    fig = Figure(figsize=(9.0, 8.0))
    FigureCanvasAgg(fig)
    k = 0
    for sensor in ("s1", "s2"):
        raw, cor, s_raw, s_cor = panels[sensor]
        for name, r, s in (("raw", raw, s_raw), ("row-gain corrected", cor, s_cor)):
            k += 1
            ax = fig.add_subplot(2, 2, k)
            ext = r["extent"]
            ax.imshow(r["canvas"], cmap="jet", origin="lower",
                      extent=ext, aspect="equal")
            if s[0] is not None:
                zz = np.array([ext[2], ext[3]])
                y0_, y1_ = ext[0], ext[1]
                w = np.clip(r["canvas"], 0, None)
                yy = ext[0] + np.arange(r["canvas"].shape[1]) * res_mm
                zc = ext[2] + np.arange(r["canvas"].shape[0]) * res_mm
                tot = w.sum(1)
                keep = tot >= 0.25 * tot.max()
                cen = (w[keep] * yy[None, :]).sum(1) / tot[keep]
                ax.plot(cen, zc[keep], "w.", ms=2)
                mid = cen.mean() - s[0] * zc[keep].mean()
                ax.plot(s[0] * zz + mid, zz, "w-", lw=1.4)
                ax.set_xlim(y0_, y1_); ax.set_ylim(ext[2], ext[3])
                ax.set_title(f"{sensor} — {name}\n|dY/dZ| = {abs(s[0]):.3f} "
                             f"({abs(s[3]):.1f} deg), R2 {s[1]:.2f}",
                             fontsize=9)
            else:
                ax.set_title(f"{sensor} — {name}\n(no band)", fontsize=9)
            ax.set_xlabel("Y (mm)"); ax.set_ylabel("Z (mm)")
    fig.suptitle(f"Hinge row-gain removed, then tilt measured "
                 f"(expected {expect:.3f} = {tilt_deg} deg)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = os.path.join(tilt_run, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "flatfield_tilt.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    save_gain(os.path.join(out_dir, "row_gain.json"), gains)

    rpt = "\n".join(L)
    with open(os.path.join(out_dir, "flatfield_report.txt"), "w") as f:
        f.write(rpt + "\n")
    print(rpt)
    print(f"\nsaved {png}")
    return gains, rpt


if __name__ == "__main__":
    args, nums = [], []
    for a in sys.argv[1:]:
        try:
            nums.append(float(a))
        except ValueError:
            args.append(a)
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    tilt = nums[0] if nums else 20.0
    res = nums[1] if len(nums) > 1 else 0.75
    compare(args[0], args[1], tilt, res)
