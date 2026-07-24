"""
diag12_mesh_geometry.py — measure the sensor's SENSING GEOMETRY directly
from the extension's own mesh_state log.

The TSF extension logs the filtered sensing nodes (18 x 12 = 216 per
sensor_config.json) as positions IN THE CASE-LOCAL FRAME. Those node
positions ARE the sensing area's geometry, authored by the sensor model —
no contact, no physics, no force assumptions, no tipping.

From one frame we get, definitively:
  1. the sensing area's EXTENT per axis  -> is it really 22 x 37 mm?
     (if the long axis is ~19-21 mm instead, PITCH_Z in stitching.py is
      ~2x too large and the ladder's early vanishing point is explained)
  2. the node-cloud CENTROID offset from the CASE ORIGIN along the long
     axis -> that magnitude IS PAD_CENTER_ABOVE_CASE_M (currently 0.0329)
  3. the per-row centroids (18 rows) -> the row axis, the row pitch, and
     which end row 0 sits at (an independent check on flip_ud)

Only the header + first data row are read, so the 159 MB file is fine.

Usage (normal python):
  python3 diag12_mesh_geometry.py <run_dir> [sensor]
e.g.
  python3 diag12_mesh_geometry.py \
      ~/Paper3_Simulation/Data/gui_run/run_20260723_153318 s1
"""
import os, sys, re, json
import numpy as np

CURRENT_CONST_MM = 32.9        # PAD_CENTER_ABOVE_CASE_M currently in use
ASSUMED_PAD_H_MM = 37.0        # what stitching.py assumes (7 rows)
ASSUMED_PAD_W_MM = 22.0        # what stitching.py assumes (4 cols)
N_TAXEL_ROWS, N_TAXEL_COLS = 7, 4


def sniff(header):
    return "\t" if header.count("\t") > header.count(",") else ","


def read_head(path, n_rows=2):
    with open(path) as f:
        header = f.readline().rstrip("\n")
        rows = []
        for _ in range(n_rows):
            line = f.readline()
            if not line:
                break
            rows.append(line.rstrip("\n"))
    return header, rows


def load_grid_shape():
    for p in (os.path.expanduser(
                "~/Paper3_Simulation/TSF-85/TSF_85_Ext/data/sensor_config.json"),):
        if os.path.exists(p):
            with open(p) as f:
                c = json.load(f)
            g = c.get("grid", {})
            return int(g.get("rows", 18)), int(g.get("cols", 12))
    return 18, 12


def main(run_dir, sensor="s1"):
    path = os.path.join(run_dir, f"gui_{sensor}_mesh_state.csv")
    if not os.path.exists(path):
        cands = [f for f in os.listdir(run_dir) if "mesh_state" in f]
        if not cands:
            print(f"no mesh_state csv in {run_dir}")
            return
        path = os.path.join(run_dir, cands[0])
    print(f"file: {path}")
    header, rows = read_head(path)
    d = sniff(header)
    cols = header.split(d)
    print(f"delimiter: {'TAB' if d == chr(9) else 'COMMA'}   n_columns: {len(cols)}")
    print("first 12 columns :", cols[:12])
    print("last 6 columns   :", cols[-6:])
    if not rows:
        print("no data rows")
        return

    pat = re.compile(r"node(\d+)_([xyz])$")
    idx = {}
    for i, name in enumerate(cols):
        m = pat.match(name.strip())
        if m:
            idx.setdefault(int(m.group(1)), {})[m.group(2)] = i
    if not idx:
        print("\nNo node{i}_x columns found — this looks like the LONG format.")
        print("Paste the header line above and I'll adapt the parser.")
        return

    vals = rows[0].split(d)
    ids = sorted(k for k in idx if all(a in idx[k] for a in "xyz"))
    pts = []
    for k in ids:
        try:
            pts.append([float(vals[idx[k][a]]) for a in "xyz"])
        except (ValueError, IndexError):
            pts.append([np.nan] * 3)
    P = np.array(pts, dtype=float)
    good = ~np.isnan(P).any(axis=1)
    P = P[good]
    ids = [k for k, g in zip(ids, good) if g]
    print(f"\nparsed {len(P)} sensing nodes from frame 0 (case-local frame)")

    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = hi - lo
    ctr = 0.5 * (lo + hi)
    mean = P.mean(axis=0)
    unit = 1000.0 if ext.max() < 1.0 else 1.0     # metres vs mm autodetect
    tag = "mm (converted from m)" if unit == 1000.0 else "mm (already mm)"
    print(f"units: {tag}")
    print(f"  extent  : X={unit*ext[0]:7.2f}  Y={unit*ext[1]:7.2f}  Z={unit*ext[2]:7.2f}")
    print(f"  bbox ctr: X={unit*ctr[0]:+7.2f}  Y={unit*ctr[1]:+7.2f}  Z={unit*ctr[2]:+7.2f}")
    print(f"  mean pos: X={unit*mean[0]:+7.2f}  Y={unit*mean[1]:+7.2f}  Z={unit*mean[2]:+7.2f}")

    order = np.argsort(ext)[::-1]
    long_ax, wide_ax, norm_ax = int(order[0]), int(order[1]), int(order[2])
    L = unit * ext[long_ax]
    W = unit * ext[wide_ax]
    off = unit * ctr[long_ax]
    print(f"\n  long axis   = {'XYZ'[long_ax]}  extent {L:.2f} mm "
          f"(stitching assumes {ASSUMED_PAD_H_MM})")
    print(f"  wide axis   = {'XYZ'[wide_ax]}  extent {W:.2f} mm "
          f"(stitching assumes {ASSUMED_PAD_W_MM})")
    print(f"  normal axis = {'XYZ'[norm_ax]}  extent {unit*ext[norm_ax]:.2f} mm")
    print(f"\n  CASE ORIGIN -> array centre along long axis = {off:+.2f} mm")
    print(f"  |offset| = {abs(off):.2f} mm   vs current constant "
          f"{CURRENT_CONST_MM:.1f} mm   -> delta {abs(off)-CURRENT_CONST_MM:+.2f} mm")
    print(f"\n  implied taxel pitch: long {L/N_TAXEL_ROWS:.3f} mm/row "
          f"(stitching uses {ASSUMED_PAD_H_MM/N_TAXEL_ROWS:.3f})")
    print(f"                       wide {W/N_TAXEL_COLS:.3f} mm/col "
          f"(stitching uses {ASSUMED_PAD_W_MM/N_TAXEL_COLS:.3f})")

    # ---- per-row structure of the deformation grid ----
    g_rows, g_cols = load_grid_shape()
    if len(P) == g_rows * g_cols:
        print(f"\n  node count matches the {g_rows}x{g_cols} grid -> per-row centres "
              f"along the long axis ({'XYZ'[long_ax]}):")
        M = P[:, long_ax].reshape(g_rows, g_cols) * unit
        rc = M.mean(axis=1)
        for r in range(g_rows):
            print(f"      grid row {r:2d}: {rc[r]:+8.2f} mm")
        step = float(np.mean(np.diff(rc)))
        print(f"  mean spacing between grid rows = {step:+.3f} mm "
              f"(sign shows which way row index runs)")
        print(f"  grid row 0 is at the "
              f"{'MORE POSITIVE' if rc[0] > rc[-1] else 'MORE NEGATIVE'} end "
              f"of the long axis")
    else:
        print(f"\n  node count {len(P)} != {g_rows}x{g_cols} = {g_rows*g_cols}; "
              f"skipping per-row breakdown")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "s1")
