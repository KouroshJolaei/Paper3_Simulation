"""
diag8_row_slope.py — flip_ud verdict from RAW CSVs, using the tilted rod
as the physical probe. Model-free: no stitcher, no CAL, no mirroring.

Physics of this run (+20 deg about X): the rod top leans toward world -Y,
and the CSV column index increases toward -Y (PROVEN by the diag7 Y-sweep).
Therefore the pad row that touches the rod HIGHER in the world must have
its pressure centroid at a HIGHER column index.

  If CSV row 0 is physically the TOP (+Z) row  -> flip_ud=False CORRECT:
      per-row column centroid DECREASES from row 0 -> row 6
      (slope ~ -0.35 cols/row for 20 deg: 5.286*tan20 / 5.5)
  If the centroid INCREASES down the rows -> row 0 is the BOTTOM row ->
      rows are FLIPPED -> mirror the zs line in _taxel_centers
      (stitching.py), NOT via CAL:
          zs = (np.arange(N_ROWS) - (N_ROWS - 1) / 2.0) * PITCH_Z

Usage (normal python, NOT isaac):
  python3 diag8_row_slope.py <run_dir>
"""
import sys, os, glob
import numpy as np

EXPECT = np.tan(np.deg2rad(20.0)) * 5.286 / 5.5   # ~0.35 cols/row


def hold_average(csv_path):
    """Mean 7x4 map over frames with taxel sum >= 0.5 * peak. NEVER transpose."""
    rows = []
    with open(csv_path) as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) < 30:
                continue
            try:
                rows.append([float(v) for v in p[2:30]])
            except ValueError:
                continue
    if not rows:
        return None
    a = np.array(rows)
    s = a.sum(1)
    return a[s >= 0.5 * s.max()].mean(0).reshape(7, 4)


def main(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "*_pt*_s1_tactile_maps.csv")))
    if not files:
        print(f"no *_pt*_s1_tactile_maps.csv in {run_dir}")
        return
    slopes = {"s1": [], "s2": []}
    for f1 in files:
        tag = os.path.basename(f1).split("_s1_")[0]
        print("=" * 60)
        print(tag)
        for name in ("s1", "s2"):
            path = f1 if name == "s1" else f1.replace("_s1_", "_s2_")
            if not os.path.exists(path):
                print(f"  {name}: FILE MISSING")
                continue
            m = hold_average(path)
            if m is None:
                print(f"  {name}: no data")
                continue
            print(f"  {name}: row -> pressure-weighted column centroid")
            cents, rs = [], []
            for r in range(7):
                w = m[r] - m[r].min()          # sharpen: per-row floor removed
                if w.sum() <= 1e-9:
                    continue
                c = float((w * np.arange(4)).sum() / w.sum())
                cents.append(c)
                rs.append(r)
                print(f"    row {r}: centroid {c:5.2f}   "
                      + " ".join(f"{v:7.1f}" for v in m[r]))
            if len(rs) >= 3:
                sl = float(np.polyfit(rs, cents, 1)[0])
                slopes[name].append(sl)
                print(f"    SLOPE d(centroid)/d(row) = {sl:+.3f} cols/row")
    print("\n" + "=" * 60)
    print(f"VERDICT   (expected |slope| ~ {EXPECT:.2f} cols/row at 20 deg)")
    for name in ("s1", "s2"):
        if not slopes[name]:
            continue
        ms = float(np.mean(slopes[name]))
        if ms < -0.05:
            v = "DECREASES -> row 0 IS the top -> flip_ud=False CORRECT"
        elif ms > +0.05:
            v = "INCREASES -> rows FLIPPED -> mirror the zs line in _taxel_centers"
        else:
            v = "flat -> contact too concentrated to judge; rely on calibrate"
        print(f"  {name}: mean slope {ms:+.3f}  -> {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
