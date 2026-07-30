"""
repeat_compare.py — How repeatable is one grid run?

Compares TWO runs of the IDENTICAL config at three levels:

  1. PER-GRASP  : each ptNN's 7x4 hold-average map, run A vs run B.
                  Isolates sim/sensor repeatability from stitching.
  2. POSE       : each ptNN's pad centre (y,z), run A vs run B.
                  Isolates robot placement repeatability.
  3. STITCH     : the full stitched canvas, aligned on pt00's pad centre
                  and cropped to the common region.
                  This is the number that matters for a training pair.

Reuses stitching.py's own geometry, so it can never drift from the real
paint step.

Usage:
  python3 repeat_compare.py <run_A> <run_B> [res_mm]
"""

import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import stitching as ST      # noqa: E402


def _rel(a, b, scale):
    """Difference stats between two arrays, as % of `scale`."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"rms": float(np.sqrt(np.mean(d ** 2))),
            "max": float(np.max(np.abs(d))),
            "rms_pct": 100.0 * float(np.sqrt(np.mean(d ** 2))) / scale,
            "max_pct": 100.0 * float(np.max(np.abs(d))) / scale}


def compare(run_a, run_b, res_mm=0.75):
    out = []
    out.append(f"REPEATABILITY  ({res_mm} mm/cell)")
    out.append(f"  A: {os.path.basename(os.path.abspath(run_a))}")
    out.append(f"  B: {os.path.basename(os.path.abspath(run_b))}")

    for sensor in ("s1", "s2"):
        try:
            A = ST.build_canvas(run_a, sensor, res_mm, verbose=False)
            B = ST.build_canvas(run_b, sensor, res_mm, verbose=False)
        except RuntimeError as e:
            out.append(f"\n{sensor}: {e}")
            continue

        ga = {k: (o, m) for k, (o, m) in
              ((g[0], (g[1], g[2])) for g in A["grasps"])}
        gb = {k: (o, m) for k, (o, m) in
              ((g[0], (g[1], g[2])) for g in B["grasps"])}
        common = sorted(set(ga) & set(gb))
        out.append(f"\n{sensor}:  {len(common)} grasps in both "
                   f"(A has {len(ga)}, B has {len(gb)})")

        # ---- 1. per-grasp maps -----------------------------------------
        peak = max(float(np.max(ga[k][1])) for k in common) or 1.0
        rows = []
        for k in common:
            rows.append((k, _rel(ga[k][1], gb[k][1], peak)))
        rms = [r[1]["rms_pct"] for r in rows]
        mx = [r[1]["max_pct"] for r in rows]
        out.append(f"  1. per-grasp 7x4 maps  (peak {peak:.0f} a.u.)")
        out.append(f"     mean RMS diff = {np.mean(rms):.2f} % of peak"
                   f"   worst grasp = {max(rms):.2f} %  ({rows[int(np.argmax(rms))][0]})")
        out.append(f"     max cell diff = {max(mx):.2f} % of peak")

        # ---- 2. poses ---------------------------------------------------
        dy = np.array([ga[k][0][0] - gb[k][0][0] for k in common])
        dz = np.array([ga[k][0][1] - gb[k][0][1] for k in common])
        d = np.hypot(dy, dz)
        out.append(f"  2. pad placement (offsets, re-anchored the same way)")
        out.append(f"     mean |dpos| = {d.mean():.3f} mm   max = {d.max():.3f} mm"
                   f"   (dy {dy.mean():+.3f}, dz {dz.mean():+.3f} mean)")

        # ---- 3. stitched canvas, aligned on pt00 ------------------------
        ia, ib = ST._initial_index(A), ST._initial_index(B)
        oya, oza = A["grasps"][ia][1]
        oyb, ozb = B["grasps"][ib][1]
        ya0, _, za0, _ = A["extent"]
        yb0, _, zb0, _ = B["extent"]
        # cell index of each run's pt00 centre
        cay = (oya - ya0) / res_mm; caz = (oza - za0) / res_mm
        cby = (oyb - yb0) / res_mm; cbz = (ozb - zb0) / res_mm
        sy = int(round(cay - cby)); sz = int(round(caz - cbz))
        CA, CB = A["canvas"], B["canvas"]
        # overlap window in A's index space
        y_lo = max(0, sy); y_hi = min(CA.shape[1], CB.shape[1] + sy)
        z_lo = max(0, sz); z_hi = min(CA.shape[0], CB.shape[0] + sz)
        sub_a = CA[z_lo:z_hi, y_lo:y_hi]
        sub_b = CB[z_lo - sz:z_hi - sz, y_lo - sy:y_hi - sy]
        pk = float(max(np.max(sub_a), np.max(sub_b))) or 1.0
        st = _rel(sub_a, sub_b, pk)
        out.append(f"  3. stitched canvas, aligned on pt00 "
                   f"({sub_a.shape[0]}x{sub_a.shape[1]} cells common)")
        out.append(f"     RMS diff = {st['rms']:.1f} a.u. = {st['rms_pct']:.2f} % of peak")
        out.append(f"     max diff = {st['max']:.1f} a.u. = {st['max_pct']:.2f} % of peak")
        out.append(f"     overlap sigma:  A {A['overlap_std']:.0f}   B {B['overlap_std']:.0f}")

    return "\n".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    res = 0.75
    dirs = []
    for a in args:
        try:
            res = float(a)
        except ValueError:
            dirs.append(a)
    if len(dirs) < 2:
        print(__doc__)
        sys.exit(1)
    rpt = compare(dirs[0], dirs[1], res)
    print(rpt)
    with open(os.path.join(dirs[1], "repeatability_report.txt"), "w") as f:
        f.write(rpt + "\n")
    print(f"\nsaved {os.path.join(dirs[1], 'repeatability_report.txt')}")
