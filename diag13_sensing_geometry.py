"""
diag13_sensing_geometry.py — the sensing array's true geometry, read from
the extension's own mesh log (LONG format: one row per node per frame).

The mesh log contains EVERY node of the deformable pad. Only the 216 ids
listed in sensor_config.json (18 x 12) are the SENSING grid that feeds the
CNN, and the 7 x 4 tactile output represents that same physical area. So:

  * bbox of ALL nodes      -> the deformable pad body (the ~41 mm figure)
  * bbox of SENSING nodes  -> the real sensing area  <-- the number that
                              PITCH_Y / PITCH_Z in stitching.py must match
  * sensing centroid, expressed in the CASE frame -> PAD_CENTER_ABOVE_CASE_M
  * per-grid-row centres   -> row pitch, and which end grid row 0 sits at

Node positions are converted into the case frame using the Trans/Ori
columns logged on the same row, so the result is independent of where the
robot happened to be. Rest positions (_Rx/_Ry/_Rz) are used when present,
since undeformed geometry is what we want.

Usage (normal python):
  python3 diag13_sensing_geometry.py <run_dir> [s1|s2]
"""
import os, sys, json
import numpy as np

CURRENT_CONST_MM = 32.9
ASSUMED_PAD_H_MM, ASSUMED_PAD_W_MM = 37.0, 22.0
N_TAXEL_ROWS, N_TAXEL_COLS = 7, 4
SENSOR_CFG = os.path.expanduser(
    "~/Paper3_Simulation/TSF-85/TSF_85_Ext/data/sensor_config.json")


def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_cfg():
    with open(SENSOR_CFG) as f:
        c = json.load(f)
    g = c["grid"]
    return int(g["rows"]), int(g["cols"]), c["node_ids"]


def read_frame0(path, pfx):
    with open(path) as f:
        cols = f.readline().rstrip("\n").split("\t")
        ix = {n.strip(): i for i, n in enumerate(cols)}

        def grp(names):
            return [ix[n] for n in names] if all(n in ix for n in names) else None

        c_pos = grp([f"{pfx}_x", f"{pfx}_y", f"{pfx}_z"])
        c_rest = grp([f"{pfx}_Rx", f"{pfx}_Ry", f"{pfx}_Rz"])
        c_tr = grp([f"{pfx}_Trans_x", f"{pfx}_Trans_y", f"{pfx}_Trans_z"])
        c_or = grp([f"{pfx}_Ori_w", f"{pfx}_Ori_x", f"{pfx}_Ori_y", f"{pfx}_Ori_z"])
        i_fr, i_nd = ix.get("frame"), ix.get("node_id")
        if c_pos is None or i_fr is None or i_nd is None:
            print("unexpected columns:", cols)
            return None
        pos, rest, nid, T, Q = {}, {}, [], None, None
        f0 = None
        for line in f:
            v = line.rstrip("\n").split("\t")
            if len(v) < len(cols):
                continue
            try:
                fr = float(v[i_fr])
            except ValueError:
                continue
            if f0 is None:
                f0 = fr
            elif fr != f0:
                break
            try:
                k = int(float(v[i_nd]))
                pos[k] = [float(v[j]) for j in c_pos]
                if c_rest:
                    rest[k] = [float(v[j]) for j in c_rest]
                if T is None and c_tr:
                    T = np.array([float(v[j]) for j in c_tr])
                if Q is None and c_or:
                    Q = np.array([float(v[j]) for j in c_or])
            except (ValueError, IndexError):
                continue
        return pos, rest, T, Q


def to_case(P, T, Q):
    """World -> case frame, if the logged points look like world coords."""
    if T is None or Q is None:
        return P, "logged frame (no case pose columns)"
    if np.abs(P).max() < 0.15:            # already small => case-local
        return P, "case-local (as logged)"
    R = quat_to_R(*Q)
    return (P - T) @ R, "converted world -> case using Trans/Ori"


def describe(P, unit, label):
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext, ctr = (hi - lo) * unit, 0.5 * (lo + hi) * unit
    print(f"  {label}: n={len(P)}")
    print(f"      extent  X={ext[0]:7.2f}  Y={ext[1]:7.2f}  Z={ext[2]:7.2f}  mm")
    print(f"      centre  X={ctr[0]:+7.2f}  Y={ctr[1]:+7.2f}  Z={ctr[2]:+7.2f}  mm")
    return ext, ctr


def main(run_dir, sensor="s1"):
    path = os.path.join(run_dir, f"gui_{sensor}_mesh_state.csv")
    if not os.path.exists(path):
        print(f"not found: {path}")
        return
    print(f"file: {path}")
    got = read_frame0(path, sensor)
    if got is None:
        return
    pos, rest, T, Q = got
    src = rest if len(rest) == len(pos) and len(rest) else pos
    which = "REST positions" if src is rest else "current positions"
    print(f"frame 0: {len(pos)} mesh nodes   (using {which})")
    if T is not None:
        print(f"case pose on this row: T={np.round(T, 5).tolist()}  "
              f"Q={np.round(Q, 5).tolist() if Q is not None else None}")

    ids_all = sorted(src.keys())
    A = np.array([src[k] for k in ids_all], dtype=float)
    A, note = to_case(A, T, Q)
    print(f"frame handling: {note}")
    unit = 1000.0 if np.abs(A).max() < 1.0 else 1.0
    print(f"units: {'m -> mm' if unit == 1000.0 else 'already mm'}\n")

    describe(A, unit, "ALL mesh nodes (deformable pad body)")

    g_rows, g_cols, node_ids = load_cfg()
    flat = [int(v) for row in node_ids for v in row]
    missing = [k for k in flat if k not in src]
    sel = [k for k in flat if k in src]
    if missing:
        print(f"\n  NOTE: {len(missing)} of {len(flat)} sensing ids not in the log "
              f"(first few: {missing[:6]})")
    if not sel:
        print("  no sensing nodes matched — cannot continue")
        return
    S = np.array([src[k] for k in sel], dtype=float)
    S, _ = to_case(S, T, Q)
    print()
    ext, ctr = describe(S, unit, f"SENSING nodes ({g_rows}x{g_cols} grid)")

    order = np.argsort(ext)[::-1]
    la, wa = int(order[0]), int(order[1])
    L, W, off = ext[la], ext[wa], ctr[la]
    print(f"\n  long axis  = {'XYZ'[la]}  {L:.2f} mm   "
          f"(stitching assumes {ASSUMED_PAD_H_MM})")
    print(f"  wide axis  = {'XYZ'[wa]}  {W:.2f} mm   "
          f"(stitching assumes {ASSUMED_PAD_W_MM})")
    print(f"\n  >>> CASE ORIGIN -> sensing-array centre, long axis = {off:+.2f} mm")
    print(f"  >>> |offset| {abs(off):.2f} mm  vs current constant "
          f"{CURRENT_CONST_MM:.1f} mm   delta {abs(off)-CURRENT_CONST_MM:+.2f} mm")
    print(f"\n  implied taxel pitch: {L/N_TAXEL_ROWS:.3f} mm/row "
          f"(stitching uses {ASSUMED_PAD_H_MM/N_TAXEL_ROWS:.3f})")
    print(f"                       {W/N_TAXEL_COLS:.3f} mm/col "
          f"(stitching uses {ASSUMED_PAD_W_MM/N_TAXEL_COLS:.3f})")

    if len(sel) == g_rows * g_cols:
        M = (S[:, la] * unit).reshape(g_rows, g_cols)
        rc = M.mean(axis=1)
        print(f"\n  per grid-row centre along {'XYZ'[la]} (mm):")
        for r in range(g_rows):
            print(f"      row {r:2d}: {rc[r]:+8.2f}")
        print(f"  mean row spacing = {float(np.mean(np.diff(rc))):+.3f} mm")
        print(f"  grid row 0 sits at the "
              f"{'POSITIVE' if rc[0] > rc[-1] else 'NEGATIVE'} end of the long axis")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "s1")
