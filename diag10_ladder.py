"""
diag10_ladder.py — assumption-free pad-placement verdict from a Z ladder.

Physics: total tactile signal is proportional to the BACKED area of the pad.
As the commanded pad centre climbs past the rod top, the backed area — and
therefore the hold-average total — falls LINEARLY with commanded z and hits
the air baseline exactly when the pad bottom clears the rim:

    zero-contact at   z_cmd = (rod_top - obj_centre) + PAD_H/2
                            = 70.0 + 18.5 = +88.5 mm      (26 mm rod scene)

No dome model, no reference run, no calibration constant anywhere in the
loop. The fitted x-intercept IS the verdict:

    intercept ~ +88.5 mm  -> calibration correct
    intercept ~ +88.5+e   -> pad sits e mm LOWER than commanded

Usage (normal python):
  python3 diag10_ladder.py <ladder_run_dir>
Expects the 1x6 ladder: y=0, z=+70..+90, step 4.
"""
import sys, os, glob, json, re
import numpy as np

PAD_H_HALF = 18.5           # mm
ZERO_IF_CORRECT = 88.5      # mm, commanded z where contact must vanish


def hold_total(csv_path):
    """Clipped hold-average total for one CSV (sum >= 0.5*peak frames)."""
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
    hold = a[s >= 0.5 * s.max()]
    return float(np.clip(hold.mean(0), 0.0, None).sum())


def _pt_key(text):
    m = re.search(r"pt(\d+)", str(text))
    return f"pt{int(m.group(1)):02d}" if m else None


def load_z_cmd(run_dir):
    """{ptNN: commanded pad z offset (mm)} from the run's config copy."""
    for name in ("gui_config_used.json", "gui_config.json"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            c = json.load(f)
        return {f"pt{int(q['index']):02d}": float(q["pad_offset_z_mm"])
                for q in c.get("points", [])}
    ph = os.path.join(run_dir, "pose_history.json")
    if os.path.exists(ph):
        with open(ph) as f:
            d = json.load(f)
        return {f"pt{int(q['index']):02d}": float(q["pad_offset_z_mm"])
                for q in d.get("config", {}).get("points", [])}
    return {}


def main(run_dir):
    zmap = load_z_cmd(run_dir)
    if not zmap:
        print("no config with points found in run dir")
        return
    for sensor in ("s1", "s2"):
        pts = []
        for fp in sorted(glob.glob(os.path.join(run_dir,
                                   f"*_pt*_{sensor}_tactile_maps.csv"))):
            key = _pt_key(os.path.basename(fp))
            if key is None or key not in zmap:
                continue
            t = hold_total(fp)
            if t is not None:
                pts.append((zmap[key], t, key))
        if len(pts) < 4:
            print(f"{sensor}: only {len(pts)} grasps found — need the 6-point ladder")
            continue
        pts.sort()
        zs = np.array([p[0] for p in pts])
        ts = np.array([p[1] for p in pts])
        floor = float(ts.min())               # deepest-in-air grasp = baseline
        span = float(ts.max()) - floor
        print(f"{sensor}:   z_cmd |  total   (baseline {floor:.0f})")
        for z, t, k in pts:
            tag = "  <- air?" if (t - floor) < 0.15 * span else ""
            print(f"      {z:6.1f} | {t:7.0f}{tag}")
        # fit the clearly-contacting points only
        m = (ts - floor) > 0.15 * span
        if m.sum() < 2:
            print(f"  {sensor}: not enough contacting points to fit\n")
            continue
        a, b = np.polyfit(zs[m], ts[m] - floor, 1)   # y = a z + b
        z0 = -b / a
        err = z0 - ZERO_IF_CORRECT
        print(f"  {sensor}: linear fit on {int(m.sum())} contact points -> "
              f"zero-contact intercept z = {z0:+.1f} mm")
        print(f"  {sensor}: implied pad-height error = {err:+.1f} mm "
              f"(positive = pad LOWER than commanded)")
        if abs(err) <= 1.5:
            print(f"  {sensor}: VERDICT: calibration CORRECT "
                  f"(intercept within +-1.5 mm of {ZERO_IF_CORRECT})\n")
        else:
            print(f"  {sensor}: VERDICT: adjust PAD_CENTER_ABOVE_CASE_M by "
                  f"{err/1000:+.4f} m and recalibrate\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
