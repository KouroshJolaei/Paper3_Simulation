"""
stitching.py — BLOCK 2: stitch per-grasp tactile maps into ONE extended
contact map per sensor, and export the (INPUT -> TARGET) training pair.

WHAT IT DOES
  1. For every grasp in a run folder, take the HOLD-AVERAGE map (mean of
     frames with taxel sum >= 0.5 x that grasp's peak — project convention):
     one representative (7,4) map per sensor per grasp.
  2. Read each grasp's pad offset from pose_history.json (the recorded
     pad positions, mm); falls back to the commanded offsets in
     gui_config_used.json / gui_config.json.
  3. Splat every taxel as a pitch-sized rectangle onto a mm canvas
     (canvas horizontal axis = world Y, vertical = world Z).
     Overlapping cells are AVERAGED (running mean) -> natural de-noising.
  4. stitch_run() saves, per sensor, into <run>/Stitched/:
        stitched_s1.png   (map + coverage figure)
        stitched_s1.npy , stitched_s1_mask.npy      (same for s2)
  5. export_pair() writes <run>/Stitched/training_pair.npz:
        input_s1  = CENTER grasp only, on the same canvas (+ input_mask_s1)
        target_s1 = the full stitched canvas         (+ target_mask_s1)
        center_temporal_s1 = the center grasp's raw 4 temporal snapshots
                             (4,7,4), if temporal_snapshots.json exists
        (same keys for s2, plus a JSON meta string)

GEOMETRY (verify once on a real run — see the calibration check in chat):
  pad face = 22 mm (4 taxels, across = world Y) x 37 mm (7 taxels, up = Z)
  -> taxel pitch 5.50 mm in Y, 5.286 mm in Z.
  Row r of the (7,4) map runs along Z, column c along Y (never transpose).
  If a sensor's stitched band GHOSTS into displaced copies, flip that
  sensor's sign/flip constants in CAL below. For training, internal
  COHERENCE is what matters; absolute world orientation can be pinned
  later during real-robot validation.

Pure post-processing (PyCharm side). Standalone Figure + Agg canvas only —
never touches the global matplotlib backend (GUI safety rule).

Usage:
  python3 stitching.py <run_dir> [res_mm]
or from the GUI's "Stitching (Block 2)" tab.
"""

import os, sys, re, glob, json
import numpy as np
import pandas as pd

# ---- pad geometry (mm) ----
PAD_W, PAD_H = 22.0, 37.0          # 4-taxel side (Y) , 7-taxel side (Z)
N_ROWS, N_COLS = 7, 4
PITCH_Y = PAD_W / N_COLS           # 5.50 mm
PITCH_Z = PAD_H / N_ROWS           # 5.286 mm
HOLD_FRAC = 0.5                    # hold-average window (project convention)
OUTLIER_MM = 8.0                   # drop a grasp if recorded pose is >8mm
                                   # off its commanded pose (bad pose record)
MIRROR_S2_IN_OVERLAY = False        # column 3: show s2 mirrored L-R, since
                                   # the two pads face each other (display only)
BASE_FRAC = 0.05                   # frames with sum <= 5% of peak = baseline
SUBTRACT_BASELINE = False          # OFF -> single-grasp stitch == raw heatmap
                                   #        (flipped to world Z-up), easy to verify.
                                   # ON  -> remove the sensor's fixed background
                                   #        (better for multi-grasp training data).

# ---- per-sensor orientation calibration ----
# Do NOT hand-tune: run   python3 stitching.py <run_dir> 1.0 calibrate
# and paste the winning block it prints.
CAL = {
    # s1 faces the object directly -> keep its raw orientation.
    # s2 faces the OPPOSITE way (pads squeeze toward each other) -> mirror L-R.
    # No up-down flip: the array's row order already matches world Z once drawn.
    # (Verify with the 1xN line test; flip here only if contact lands off-object.)
    "s1": {"sign_dy": +1.0, "sign_dz": +1.0, "flip_lr": False, "flip_ud": False},
    "s2": {"sign_dy": +1.0, "sign_dz": +1.0, "flip_lr": False,  "flip_ud": False},
}


def hold_average(csv_path):
    """(map_7x4, n_hold_frames, peak_sum) for one tactile CSV.

    BASELINE-SUBTRACTED: each grasp's own pre/post-contact frames
    (taxel sum <= 5% of that grasp's peak) define a per-taxel baseline —
    the fixed sensor pattern visible even with the gripper open. It is
    locked to the pad, so without subtraction stitching smears a copy of
    it across the whole grid and buries the real contact structure."""
    df = pd.read_csv(csv_path)
    pred = [c for c in df.columns if c.startswith("pred_")]
    v = df[pred].to_numpy()
    s = v.sum(1)
    if not len(s):
        return np.zeros((N_ROWS, N_COLS)), 0, 0.0
    peak, smin = float(s.max()), float(s.min())
    rng = peak - smin
    if rng <= 0:
        return np.zeros((N_ROWS, N_COLS)), 0, 0.0
    # thresholds referenced to the MINIMUM: a fixed sensor pattern raises the
    # floor, so peak-relative thresholds would never find baseline frames
    base_mask = s <= smin + BASE_FRAC * rng
    hold_mask = s >= smin + HOLD_FRAC * rng
    if SUBTRACT_BASELINE:
        baseline = v[base_mask].mean(0) if base_mask.sum() >= 3 else 0.0
        hold = np.clip(v[hold_mask].mean(0) - baseline, 0.0, None)
    else:
        hold = v[hold_mask].mean(0)
    return hold.reshape(N_ROWS, N_COLS), int(hold_mask.sum()), peak


def _pt_key(text):
    """'gui_pt03_s1_...' or 'pt3' -> 'pt03' (None if no ptNN found)."""
    m = re.search(r"pt(\d+)", str(text))
    return f"pt{int(m.group(1)):02d}" if m else None


def _span_mm(offs):
    """Largest extent of the offsets (mm). ~0 means all grasps at one spot."""
    ys = [v[0] for v in offs.values()]
    zs = [v[1] for v in offs.values()]
    return float(np.hypot(max(ys) - min(ys), max(zs) - min(zs)))


DEGENERATE_SPAN_MM = 1.0     # >1 grasp but everything within 1 mm = broken source


def load_offsets(run_dir):
    """Return ({ptNN: (y_mm, z_mm)}, source_string) in a common frame.

    Tries several sources IN ORDER and skips any whose offsets are
    DEGENERATE (all grasps within ~1 mm of one spot — a pose-recording
    bug we have actually seen, where pad_actual stayed stuck at the
    startup pose):
      1. pose_history.json  pad_actual_pos_m   (real recorded pad positions)
      2. pose_history.json  ee_world_m         (EE differences == pad differences)
      3. pose_history.json  pad_desired_pos_m  (initial + commanded offset)
      4. run config points                     (commanded offsets)
    The chosen source is printed and shown in the figure title."""
    candidates = []
    ph = os.path.join(run_dir, "pose_history.json")
    if os.path.exists(ph):
        with open(ph) as f:
            data = json.load(f)
        pts = data.get("points", [])
        for field in ("pad_actual_pos_m", "ee_world_m", "pad_desired_pos_m"):
            offs = {}
            for p in pts:
                key = _pt_key(p.get("tag", ""))
                pos = p.get(field)
                if key and pos:
                    offs[key] = (float(pos[1]) * 1000.0, float(pos[2]) * 1000.0)
            if offs:
                candidates.append((offs, f"pose_history.json [{field}]"))
    for cfg_name in ("gui_config_used.json", "gui_config.json"):
        cfg = os.path.join(run_dir, cfg_name)
        if os.path.exists(cfg):
            with open(cfg) as f:
                c = json.load(f)
            offs = {}
            for p in c.get("points", []):
                key = f"pt{int(p['index']):02d}"
                offs[key] = (float(p["pad_offset_y_mm"]),
                             float(p["pad_offset_z_mm"]))
            if offs:
                candidates.append((offs, cfg_name + " (commanded offsets)"))

    for offs, src in candidates:
        if len(offs) < 2 or _span_mm(offs) > DEGENERATE_SPAN_MM:
            return offs, src
        print(f"[stitch] SKIPPING {src}: all {len(offs)} grasps within "
              f"{_span_mm(offs):.3f} mm of one spot (degenerate / recording bug)")
    if candidates:
        offs, src = candidates[0]
        return offs, src + "  (WARNING: degenerate offsets!)"
    return {}, "none"


def _reanchor_to_gui(run_dir, offs, verbose=True):
    """Remove the constant FK offset (wrist->pad points at the pad prim
    origin, not the face) so the map axes match the GUI/world frame.
    Returns (offs, applied). NEVER silent: prints what it did or why not."""
    import statistics
    reasons = []
    for cfg_name in ("gui_config_used.json", "gui_config.json",
                     os.path.join("..", "..", "gui_config.json")):
        cfg = os.path.join(run_dir, cfg_name)
        if not os.path.exists(cfg):
            reasons.append(f"{cfg_name}: not found")
            continue
        try:
            with open(cfg) as f:
                c = json.load(f)
            ctr = c["object"]["center_world_mm"]
            cmd = {f"pt{int(p['index']):02d}":
                   (float(ctr[1]) + float(p["pad_offset_y_mm"]),
                    float(ctr[2]) + float(p["pad_offset_z_mm"]))
                   for p in c.get("points", [])}
        except Exception as e:
            reasons.append(f"{cfg_name}: parse error ({e})")
            continue
        keys = [k for k in offs if k in cmd]
        if len(keys) < max(2, len(offs) // 2):
            reasons.append(f"{cfg_name}: only {len(keys)}/{len(offs)} points match")
            continue
        dys = [offs[k][0] - cmd[k][0] for k in keys]
        dzs = [offs[k][1] - cmd[k][1] for k in keys]
        sy = statistics.pstdev(dys) if len(keys) > 2 else 0.0
        sz = statistics.pstdev(dzs) if len(keys) > 2 else 0.0
        # Accept up to ~8 mm spread: small per-row IK/gravity droop makes the
        # actual-vs-commanded gap vary slightly. We remove the MEDIAN shift,
        # which aligns the overlay to the GUI frame within a couple mm — enough
        # to check shape-vs-object. Larger spread => genuinely wrong config.
        REANCHOR_MAX_SPREAD = 8.0
        if sy > REANCHOR_MAX_SPREAD or sz > REANCHOR_MAX_SPREAD:
            reasons.append(f"{cfg_name}: delta not constant "
                           f"(spread y={sy:.1f}, z={sz:.1f} mm > "
                           f"{REANCHOR_MAX_SPREAD:.0f})")
            continue
        dy, dz = statistics.median(dys), statistics.median(dzs)
        if verbose:
            print(f"[stitch] axes re-anchored to GUI frame via {cfg_name} "
                  f"(removed MEDIAN FK offset dy={dy:+.1f}, dz={dz:+.1f} mm; "
                  f"residual spread y={sy:.1f}, z={sz:.1f} mm)")
        return {k: (v[0] - dy, v[1] - dz) for k, v in offs.items()}, True
    if verbose:
        print("[stitch] axes NOT re-anchored (raw pad_actual frame): "
              + " | ".join(reasons))
    return offs, False


def _taxel_centers(cal):
    """(7,4,2) array of taxel (y,z) positions in pad-local mm (center = 0).
    Row 0 is drawn at the TOP (+Z), matching imshow of the (7,4) map."""
    # ys = (np.arange(N_COLS) - (N_COLS - 1) / 2.0) * PITCH_Y
    ys = ((N_COLS - 1) / 2.0 - np.arange(N_COLS)) * PITCH_Y
    # zs = ((N_ROWS - 1) / 2.0 - np.arange(N_ROWS)) * PITCH_Z
    zs = (np.arange(N_ROWS) - (N_ROWS - 1) / 2.0) * PITCH_Z  # row 0 = BOTTOM (proven 2026-07-22)
    if cal["flip_lr"]:
        ys = ys[::-1]
    if cal["flip_ud"]:
        zs = zs[::-1]
    Y, Z = np.meshgrid(ys, zs)
    return np.stack([Y, Z], axis=-1)


def build_canvas(run_dir, sensor, res_mm=1.0, cal=None, verbose=True):
    """Stitch all grasps of one sensor onto a mm canvas.
    Returns a dict with the canvas, coverage count, extent, the per-grasp
    list, the center-grasp index/key, an overlap-disagreement score, and a
    paint(indices) helper re-rendering any subset on the SAME grid."""
    offs, src = load_offsets(run_dir)
    if not offs:
        raise RuntimeError("no pad offsets found "
                           "(need pose_history.json or the run's config copy)")
    gui_frame = "pose_history" not in src        # config offsets are GUI-frame
    if "pose_history" in src:
        offs, gui_frame = _reanchor_to_gui(run_dir, offs, verbose=verbose)
    files = sorted(glob.glob(os.path.join(run_dir, f"*_{sensor}_tactile_maps.csv")))
    cal = cal if cal is not None else CAL[sensor]
    grasps = []                                    # (key, (y_mm,z_mm), map7x4)
    for fp in files:
        key = _pt_key(os.path.basename(fp))
        if key is None or key not in offs:
            continue
        m, nfr, peak = hold_average(fp)
        if peak <= 0:
            continue
        oy = offs[key][0] * cal["sign_dy"]
        oz = offs[key][1] * cal["sign_dz"]
        grasps.append((key, (oy, oz), m))
    # ---- outlier rejection: compare recorded offset to COMMANDED offset ----
    gui = _load_gui_grid(run_dir)
    if gui is not None and len(grasps) >= 4:
        # convert this sensor's stored (world) offsets back to (Y,Z) and compare
        # to commanded world positions; median-centre so a constant frame shift
        # (the FK offset) does not count as an outlier.
        resid = {}
        for key, (oy, oz), _m in grasps:
            if key in gui["cmd"]:
                resid[key] = (oy - gui["cmd"][key][0], oz - gui["cmd"][key][1])
        if resid:
            mdy = float(np.median([r[0] for r in resid.values()]))
            mdz = float(np.median([r[1] for r in resid.values()]))
            kept, dropped = [], []
            for g in grasps:
                key = g[0]
                if key in resid:
                    ey = resid[key][0] - mdy; ez = resid[key][1] - mdz
                    if np.hypot(ey, ez) > OUTLIER_MM:
                        dropped.append((key, ey, ez)); continue
                kept.append(g)
            if dropped:
                print(f"[stitch {sensor}] DROPPED {len(dropped)} outlier grasp(s) "
                      f"(pose off by >{OUTLIER_MM:.0f} mm from commanded):")
                for key, ey, ez in dropped:
                    print(f"      {key}: off by (dY={ey:+.1f}, dZ={ez:+.1f}) mm")
                grasps = kept
    if not grasps:
        raise RuntimeError(f"no usable {sensor} grasps matched to offsets")
    if verbose:
        print(f"[stitch {sensor}] {len(grasps)} grasps kept, offsets from {src}")

    tax = _taxel_centers(cal)
    ys = [g[1][0] for g in grasps]
    zs = [g[1][1] for g in grasps]
    cy, cz = float(np.mean(ys)), float(np.mean(zs))          # canvas center

    # Canvas spans the pad-CENTRE sweep + half a pad on each side (so full pads
    # fit), plus a small FIXED margin in mm. (Previously the margin was
    # 2*res_mm, which ballooned to 10 mm/side at res=5 and inflated the frame.)
    # Canvas = pad-CENTRE sweep + exactly half a pad on each side = the full
    # pad footprint over the sweep, with NO extra margin. This makes the frame
    # hug the swept region exactly (== the grid outer bound + pad), no dead blue
    # border. All three columns use this same extent.
    MARGIN = 0.0
    half_y = (max(ys) - min(ys)) / 2.0 + PAD_W / 2.0 + MARGIN
    half_z = (max(zs) - min(zs)) / 2.0 + PAD_H / 2.0 + MARGIN
    ny = int(np.ceil(2 * half_y / res_mm))
    nz = int(np.ceil(2 * half_z / res_mm))
    y0, z0 = cy - half_y, cz - half_z                        # origin (mm)

    # each taxel paints a FIXED-SIZE block (its true footprint in cells),
    # anchored at its centre. This preserves the 7x4 pattern exactly at any
    # resolution — neighbours never merge or split (the old edge-rounding did,
    # which mangled the map at coarse res like 5 mm/cell).
    _bw = max(1, int(round(PITCH_Y / res_mm)))     # taxel block width  (cells)
    _bh = max(1, int(round(PITCH_Z / res_mm)))     # taxel block height (cells)

    def _splat_one(oy, oz, m, acc, cnt, sq):
        for r in range(N_ROWS):
            for c in range(N_COLS):
                ty = oy + tax[r, c, 0]
                tz = oz + tax[r, c, 1]
                # cell containing the taxel centre, then a fixed block around it
                icy = int(round((ty - y0) / res_mm))
                icz = int(round((tz - z0) / res_mm))
                iy0 = icy - _bw // 2; iy1 = iy0 + _bw
                iz0 = icz - _bh // 2; iz1 = iz0 + _bh
                iy0, iy1 = max(iy0, 0), min(iy1, ny)
                iz0, iz1 = max(iz0, 0), min(iz1, nz)
                if iy1 > iy0 and iz1 > iz0:
                    acc[iz0:iz1, iy0:iy1] += m[r, c]
                    cnt[iz0:iz1, iy0:iy1] += 1.0
                    sq[iz0:iz1, iy0:iy1] += m[r, c] * m[r, c]

    def paint(indices):
        """(canvas, count, sumsq) for the given grasps, on the same grid."""
        acc = np.zeros((nz, ny))
        cnt = np.zeros((nz, ny))
        sq = np.zeros((nz, ny))
        for i in indices:
            key, (oy, oz), m = grasps[i]
            _splat_one(oy, oz, m, acc, cnt, sq)
        return np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0), cnt, sq

    canvas, cnt, sq = paint(range(len(grasps)))
    # overlap disagreement: mean per-cell std where >=2 grasps contribute.
    # Poses are ground truth, so the CORRECT array orientation minimises it.
    ov = cnt >= 2
    if ov.any():
        var = np.clip(sq[ov] / cnt[ov] - (canvas[ov]) ** 2, 0, None)
        overlap_std = float(np.mean(np.sqrt(var)))
        overlap_mean = float(np.mean(np.abs(canvas[ov]))) or 1e-9
    else:
        overlap_std, overlap_mean = 0.0, 1e-9
    d = [np.hypot(y - cy, z - cz) for y, z in zip(ys, zs)]
    center_i = int(np.argmin(d))                    # grasp closest to centroid

    return {"canvas": canvas, "count": cnt,
            "extent": (y0, y0 + ny * res_mm, z0, z0 + nz * res_mm),
            "grasps": grasps, "center_index": center_i,
            "center_key": grasps[center_i][0],
            "gui_frame": gui_frame,
            "overlap_std": overlap_std,
            "overlap_rel": overlap_std / overlap_mean,
            "res_mm": res_mm, "offset_source": src, "paint": paint}


def _load_gui_grid(run_dir):
    """Commanded grid (the SAME numbers the GUI preview uses) + object center.
    Returns dict: {cy,cz, cmd:{ptNN:(Yworld,Zworld)}, gy0,gy1,gz0,gz1} or None.
    This is what column 3 uses so it matches the GUI exactly — independent of
    any pad_actual recording noise or outliers."""
    for p in (os.path.join(run_dir, "gui_config_used.json"),
              os.path.join(run_dir, "gui_config.json"),
              os.path.join(run_dir, "..", "..", "gui_config.json"),
              os.path.expanduser("~/Paper3_Simulation/Data/gui_config.json")):
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                c = json.load(f)
            ctr = c["object"]["center_world_mm"]
            cmd = {f"pt{int(q['index']):02d}":
                   (float(ctr[1]) + float(q["pad_offset_y_mm"]),
                    float(ctr[2]) + float(q["pad_offset_z_mm"]))
                   for q in c["points"]}
        except Exception:
            continue
        if not cmd:
            continue
        gy = [v[0] for v in cmd.values()]; gz = [v[1] for v in cmd.values()]
        return {"cy": float(ctr[1]), "cz": float(ctr[2]), "cmd": cmd,
                "gy0": min(gy), "gy1": max(gy), "gz0": min(gz), "gz1": max(gz)}
    return None


def _load_scene(run_dir, verbose=True):
    """Object geometry from the run's config (for the overlay panel).
    Searches the run folder AND the standard project config locations,
    and prints exactly which file it used or why each was skipped."""
    cands = [
        os.path.join(run_dir, "gui_config_used.json"),
        os.path.join(run_dir, "gui_config.json"),
        os.path.join(run_dir, "..", "..", "gui_config.json"),
        os.path.join(run_dir, "..", "..", "..", "Data", "gui_config.json"),
        os.path.expanduser("~/Paper3_Simulation/Data/gui_config.json"),
    ]
    reasons = []
    for p in cands:
        if not os.path.exists(p):
            reasons.append(f"{p}: not found")
            continue
        try:
            with open(p) as f:
                cfg = json.load(f)
        except Exception as e:
            reasons.append(f"{p}: parse error ({e})")
            continue
        o = cfg.get("object")
        if not o or "center_world_mm" not in o:
            reasons.append(f"{os.path.basename(p)}: no 'object' block")
            continue
        if verbose:
            print(f"[stitch] overlay scene from {p} "
                  f"(tilt {o.get('tilt_deg', 0.0)} deg about "
                  f"{o.get('tilt_axis','X')})")
        return {"cy": float(o["center_world_mm"][1]),
                "cz": float(o["center_world_mm"][2]),
                "tilt": float(o.get("tilt_deg", 0.0)),
                "axis": str(o.get("tilt_axis", "X")).upper(),
                "d": float(o.get("diameter_mm", 26.0)),
                "L": float(o.get("length_mm", 140.0))}
    if verbose:
        print("[stitch] NO overlay column (object geometry not found): "
              + " | ".join(reasons))
    return None


def stitch_run(run_dir, res_mm=1.0):
    """Build + save the stitched maps for s1 and s2.
    Returns the list of saved PNG paths (may be empty)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt

    out_dir = os.path.join(run_dir, "Stitched")
    made = []
    for sensor in ("s1", "s2"):
        try:
            res = build_canvas(run_dir, sensor, res_mm)
        except RuntimeError as e:
            print(f"{sensor}: {e}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        canvas, cnt, ext = res["canvas"], res["count"], res["extent"]
        np.save(os.path.join(out_dir, f"stitched_{sensor}.npy"), canvas)
        np.save(os.path.join(out_dir, f"stitched_{sensor}_mask.npy"), cnt > 0)

        # Column 3 uses the GUI's OWN frame: commanded grid + object centre,
        # with the stitched pressure RE-PAINTED at the commanded pad positions.
        # -> the heatmap lands on the cylinder exactly as the grid box does in
        # the GUI. Immune to pad_actual noise/outliers.
        gui = _load_gui_grid(run_dir)
        scene = _load_scene(run_dir, verbose=False)
        show_overlay = gui is not None
        ncols = 3 if show_overlay else 2

        fig = Figure(figsize=(5.2 * ncols + 0.3, 4.6))
        FigureCanvasAgg(fig)
        ax1 = fig.add_subplot(1, ncols, 1)
        ax2 = fig.add_subplot(1, ncols, 2)
        im1 = ax1.imshow(canvas, cmap="jet", origin="lower",
                         extent=ext, aspect="equal")
        ax1.set_title(f"{sensor} — stitched extended map "
                      f"({len(res['grasps'])} grasps, hold-avg)", fontsize=10)
        ax1.set_xlabel("Y (mm)"); ax1.set_ylabel("Z (mm)")
        fig.colorbar(im1, ax=ax1, shrink=0.85).set_label("pressure (a.u.)",
                                                         fontsize=8)
        im2 = ax2.imshow(cnt, cmap="viridis", origin="lower",
                         extent=ext, aspect="equal")
        ax2.set_title("coverage (grasps per cell)", fontsize=10)
        ax2.set_xlabel("Y (mm)")
        fig.colorbar(im2, ax=ax2, shrink=0.85)

        if show_overlay:
            print(f"[stitch] column 3 in GUI frame "
                  f"(object centre Y={gui['cy']:.1f}, Z={gui['cz']:.1f}; "
                  f"{len(gui['cmd'])} commanded grid points)")
            # Re-paint stitched pressure at COMMANDED pad positions.
            cal = CAL[sensor]
            tax = _taxel_centers(cal)
            gmap = {g[0]: g[2] for g in res["grasps"]}         # ptNN -> 7x4 map
            pts = [(gui["cmd"][k][0], gui["cmd"][k][1], gmap[k])
                   for k in gmap if k in gui["cmd"]]
            gy = [p[0] for p in pts]; gz = [p[1] for p in pts]
            hy = (max(gy) - min(gy)) / 2 + PAD_W / 2
            hz = (max(gz) - min(gz)) / 2 + PAD_H / 2
            ccy = (max(gy) + min(gy)) / 2; ccz = (max(gz) + min(gz)) / 2
            gny = int(np.ceil(2 * hy / res_mm)); gnz = int(np.ceil(2 * hz / res_mm))
            gy0 = ccy - hy; gz0 = ccz - hz
            acc = np.zeros((gnz, gny)); gcnt = np.zeros((gnz, gny))
            for oy, oz, m in pts:
                for r in range(N_ROWS):
                    for c in range(N_COLS):
                        ty = oy + tax[r, c, 0]; tz = oz + tax[r, c, 1]
                        iy0 = int(round((ty - PITCH_Y/2 - gy0)/res_mm))
                        iy1 = int(round((ty + PITCH_Y/2 - gy0)/res_mm))
                        iz0 = int(round((tz - PITCH_Z/2 - gz0)/res_mm))
                        iz1 = int(round((tz + PITCH_Z/2 - gz0)/res_mm))
                        iy0, iy1 = max(iy0,0), min(iy1,gny)
                        iz0, iz1 = max(iz0,0), min(iz1,gnz)
                        if iy1>iy0 and iz1>iz0:
                            acc[iz0:iz1, iy0:iy1] += m[r,c]
                            gcnt[iz0:iz1, iy0:iy1] += 1.0
            gcanvas = np.where(gcnt>0, acc/np.maximum(gcnt,1.0), 0.0)
            gext = (gy0, gy0+gny*res_mm, gz0, gz0+gnz*res_mm)

            ax3 = fig.add_subplot(1, ncols, 3)
            clipped = np.ma.masked_where(~(gcnt>0), gcanvas)
            cmap = plt.cm.jet.copy(); cmap.set_bad(alpha=0.0)
            ax3.imshow(clipped, cmap=cmap, origin="lower", extent=gext,
                       aspect="equal")
            if scene:
                th = np.deg2rad(scene["tilt"]) if scene["axis"]=="X" else 0.0
                cth, sth = np.cos(th), np.sin(th)
                yl = np.array([-1,1,1,-1,-1]) * scene["d"]/2.0
                zl = np.array([-1,-1,1,1,-1]) * scene["L"]/2.0
                ax3.plot(scene["cy"] + cth*yl - sth*zl,
                         scene["cz"] + sth*yl + cth*zl,
                         color="royalblue", lw=2,
                         label=f"cylinder (tilt {scene['tilt']:g} deg about {scene['axis']})")
            ax3.plot([gui["gy0"], gui["gy1"], gui["gy1"], gui["gy0"], gui["gy0"]],
                     [gui["gz0"], gui["gz0"], gui["gz1"], gui["gz1"], gui["gz0"]],
                     "k--", lw=1.0, alpha=0.6, label="grid outer bound (pad centres)")
            ax3.scatter([p[0] for p in pts], [p[1] for p in pts],
                        s=3, color="k", alpha=0.35, zorder=5)
            # Frame the whole object like the GUI (not just the grid): include
            # the full cylinder outline plus a margin, so nothing is cut off.
            xs = [gext[0], gext[1]]; ys_ = [gext[2], gext[3]]
            if scene:
                th = np.deg2rad(scene["tilt"]) if scene["axis"] == "X" else 0.0
                cth, sth = np.cos(th), np.sin(th)
                yl = np.array([-1, 1, 1, -1]) * scene["d"]/2.0
                zl = np.array([-1, -1, 1, 1]) * scene["L"]/2.0
                cyy = scene["cy"] + cth*yl - sth*zl
                cyz = scene["cz"] + sth*yl + cth*zl
                xs += [cyy.min(), cyy.max()]; ys_ += [cyz.min(), cyz.max()]
            mx = 8
            x_lo, x_hi = min(xs)-mx, max(xs)+mx
            ax3.set_ylim(min(ys_)-mx, max(ys_)+mx)
            if sensor == "s2" and MIRROR_S2_IN_OVERLAY:
                ax3.set_xlim(x_hi, x_lo)          # mirror L-R (facing pad view)
                ax3.text(0.02, 0.02, "mirrored (facing view)", fontsize=6,
                         color="gray", transform=ax3.transAxes)
            else:
                ax3.set_xlim(x_lo, x_hi)
            ax3.set_aspect("equal")
            ax3.set_title("swept shape on object (GUI frame)", fontsize=10)
            ax3.set_xlabel("Y (mm)")
            ax3.legend(fontsize=6, loc="upper right")
        else:
            print("[stitch] NO overlay column: could not read commanded grid "
                  "from config (need gui_config with object + points).")

        fig.suptitle(f"BLOCK 2 — stitched contact map [{sensor}]   "
                     f"(offsets: {res['offset_source']}, {res_mm} mm/cell, "
                     f"overlap sigma={res['overlap_std']:.0f})",
                     fontsize=11)
        print(f"[stitch {sensor}] overlap sigma = {res['overlap_std']:.1f} "
              f"({100*res['overlap_rel']:.0f}% of overlap signal)")
        png = os.path.join(out_dir, f"stitched_{sensor}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        made.append(png)
        print(f"saved {png}")
    return made


def export_pair(run_dir, res_mm=1.0):
    """Write Stitched/training_pair.npz:
    INPUT = center grasp only on the canvas, TARGET = full stitched canvas
    (+ masks, per sensor), plus the center grasp's raw temporal snapshots
    (4,7,4) per sensor when temporal_snapshots.json exists."""
    out = {}
    meta = {"run_dir": os.path.abspath(run_dir), "res_mm": res_mm}
    for sensor in ("s1", "s2"):
        res = build_canvas(run_dir, sensor, res_mm)
        inp, icnt, _sq = res["paint"]([res["center_index"]])
        out[f"input_{sensor}"] = inp
        out[f"input_mask_{sensor}"] = icnt > 0
        out[f"target_{sensor}"] = res["canvas"]
        out[f"target_mask_{sensor}"] = res["count"] > 0
        meta[f"center_{sensor}"] = res["center_key"]
        meta[f"extent_{sensor}"] = list(res["extent"])
        meta[f"n_grasps_{sensor}"] = len(res["grasps"])
        meta[f"offset_source_{sensor}"] = res["offset_source"]

    ts = os.path.join(run_dir, "temporal_snapshots.json")
    if os.path.exists(ts):
        with open(ts) as f:
            T = json.load(f)
        for sensor in ("s1", "s2"):
            ck = meta.get(f"center_{sensor}")
            for tag, entry in T.items():
                if _pt_key(tag) == ck and sensor in entry:
                    snaps = entry[sensor]["snapshots"]
                    out[f"center_temporal_{sensor}"] = np.stack(
                        [np.array(snaps[k])
                         for k in ("p05", "p50", "p95", "post3s")])
                    meta[f"temporal_{sensor}"] = "included (4,7,4)"

    out_dir = os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "training_pair.npz")
    np.savez_compressed(npz, meta=json.dumps(meta), **out)
    print(f"saved {npz}")
    return npz


def calibrate(run_dir, res_mm=1.0):
    """Try all four array orientations per sensor on this run and score
    each by overlap disagreement (lower = overlapping grasps agree more).
    Prints a table and a ready-to-paste CAL block for the winners."""
    print(f"[calibrate] run: {run_dir}")
    best = {}
    for sensor in ("s1", "s2"):
        rows = []
        for flip_lr in (False, True):
            for flip_ud in (False, True):
                cal = {"sign_dy": +1.0, "sign_dz": +1.0,
                       "flip_lr": flip_lr, "flip_ud": flip_ud}
                try:
                    r = build_canvas(run_dir, sensor, res_mm,
                                     cal=cal, verbose=False)
                except RuntimeError as e:
                    print(f"  {sensor}: {e}")
                    rows = []
                    break
                rows.append((flip_lr, flip_ud,
                             r["overlap_std"], r["overlap_rel"]))
        if not rows:
            continue
        rows.sort(key=lambda t: t[2])
        print(f"\n  {sensor}:  flip_lr  flip_ud   overlap sigma   (rel)")
        for lr, ud, s, rel in rows:
            mark = "  <-- BEST" if (lr, ud) == rows[0][:2] else ""
            print(f"        {str(lr):5}    {str(ud):5}      {s:8.1f}     "
                  f"({100*rel:4.0f}%){mark}")
        best[sensor] = {"sign_dy": +1.0, "sign_dz": +1.0,
                        "flip_lr": rows[0][0], "flip_ud": rows[0][1]}
    if len(best) == 2:
        print("\n[calibrate] paste this into stitching.py:\n")
        print("CAL = {")
        for s in ("s1", "s2"):
            b = best[s]
            print(f'    "{s}": {{"sign_dy": +1.0, "sign_dz": +1.0, '
                  f'"flip_lr": {b["flip_lr"]}, "flip_ud": {b["flip_ud"]}}},')
        print("}")
    return best


if __name__ == "__main__":
    rd = os.path.expanduser("~/Paper3_Simulation/Data/gui_run")
    rr, mode = 1.0, "stitch"
    for a in sys.argv[1:]:
        if a.lower() in ("calibrate", "--calibrate"):
            mode = "calibrate"
        else:
            try:
                rr = float(a)
            except ValueError:
                rd = a
    # if given the gui_run parent, pick the newest run_* inside
    if not glob.glob(os.path.join(rd, "*_s1_tactile_maps.csv")):
        runs = sorted(glob.glob(os.path.join(rd, "run_*")))
        if runs:
            rd = runs[-1]
    if mode == "calibrate":
        calibrate(rd, rr)
    elif stitch_run(rd, rr):
        export_pair(rd, rr)
    else:
        print(f"nothing stitched in {rd}")
