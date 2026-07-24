"""
diag9_rim_mask.py — locate the PHYSICAL rim row in the census run by
cancelling the pad's intrinsic pressure dome.

Idea: full-contact grasps show the pad's own row profile (the "dome",
row 3 hottest even with uniform backing — see the tilted-run diag8 sums).
The census grasp profile = dome x contact mask. So

        mask[row] = census_profile[row] / reference_profile[row]

removes the dome and leaves the mask: ~1 where the rod backs the pad,
falling to ~0 in free air. The row where the mask crosses 0.5 is the rim.

NO calibration constant is involved anywhere in this measurement, so it
independently tests PAD_CENTER_ABOVE_CASE_M:
    rim ~ row 3.0   -> pad placed as commanded -> the row-4 signal is real
                       sensor bleed (constant 0.0223 vindicated)
    rim ~ row 4.5-5 -> pad sits LOWER than commanded by
                       (rim_row - 3) * 5.286 mm  (points at 0.0293)

Usage (normal python):
  python3 diag9_rim_mask.py <reference_full_contact_run> <census_run>
e.g.
  python3 diag9_rim_mask.py \
      ~/Paper3_Simulation/Data/gui_run/run_20260721_152151 \
      ~/Paper3_Simulation/Data/gui_run/<census_run_dir>
"""
import sys, os, glob
import numpy as np

PITCH_Z = 37.0 / 7.0          # 5.286 mm per row
RIM_ROW_IF_CORRECT = 3.0      # pad centre commanded exactly at the rod top


def hold_average(csv_path):
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


def run_row_profile(run_dir, sensor):
    """Mean per-row sum over all grasps of one sensor (negatives clipped)."""
    pats = sorted(glob.glob(os.path.join(run_dir, f"*_pt*_{sensor}_tactile_maps.csv")))
    profs = []
    for p in pats:
        m = hold_average(p)
        if m is None:
            continue
        profs.append(np.clip(m, 0.0, None).sum(axis=1))   # (7,)
    if not profs:
        return None
    return np.mean(profs, axis=0)


def rim_from_mask(mask):
    """First crossing of 0.5 walking UP the rows (linear interpolation)."""
    for r in range(6):
        a, b = mask[r], mask[r + 1]
        if a >= 0.5 > b:
            return r + (a - 0.5) / max(a - b, 1e-9)
    return None


def main(ref_dir, cen_dir):
    print(f"reference (full contact): {ref_dir}")
    print(f"census    (half overhang): {cen_dir}\n")
    for sensor in ("s1", "s2"):
        ref = run_row_profile(ref_dir, sensor)
        cen = run_row_profile(cen_dir, sensor)
        if ref is None or cen is None:
            print(f"{sensor}: missing CSVs in one of the runs")
            continue
        raw = cen / np.maximum(ref, 1e-9)
        # normalise so the definitely-backed bottom rows (0-2) read 1.0
        scale = np.mean(raw[0:3])
        mask = raw / max(scale, 1e-9)
        print(f"{sensor}:  row |   ref     census   mask")
        for r in range(7):
            print(f"      {r}   | {ref[r]:8.1f} {cen[r]:8.1f}   {mask[r]:5.2f}")
        rim = rim_from_mask(mask)
        if rim is None:
            print(f"  {sensor}: mask never crosses 0.5 cleanly — inspect by eye\n")
            continue
        dz = (rim - RIM_ROW_IF_CORRECT) * PITCH_Z
        print(f"  {sensor}: rim at row {rim:.2f}  ->  implied pad-height error "
              f"{dz:+.1f} mm (positive = pad sits LOWER than commanded)")
        if rim < 3.5:
            print(f"  {sensor}: VERDICT leaning: placement correct, row-4 signal "
                  f"is real bleed\n")
        elif rim > 4.0:
            print(f"  {sensor}: VERDICT leaning: pad LOW by ~{dz:.0f} mm — "
                  f"suspect PAD_CENTER_ABOVE_CASE_M (0.0293 ghost)\n")
        else:
            print(f"  {sensor}: borderline — run the rim ladder to split it\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
