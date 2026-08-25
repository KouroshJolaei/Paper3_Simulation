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
        input_s1  = the INITIAL grasp only, on the same canvas
                    (+ input_mask_s1). "Initial" = INITIAL_GRASP below,
                    default "pt00" = the pad pose the GUI designed.
        target_s1 = the full stitched canvas         (+ target_mask_s1)
        center_temporal_s1 = that same grasp's raw 4 temporal snapshots
                             (4,7,4), if temporal_snapshots.json exists
     A run whose designed initial grasp never executed writes NO pair.
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
HOLD_FRAC = 0.9                    # hold-average window, as a fraction of
                                   # (peak - min). Raised from 0.5 on
                                   # 2026-07-29: at 0.5 the window reached
                                   # down to 62.6% of peak, i.e. into the
                                   # closing ramp. Paper 2 averaged a fixed
                                   # 1 s window taken entirely inside the
                                   # steady grasp (network_gsr /
                                   # tactile_DataReadSave3.run_average), so
                                   # 0.9 (91.8-100% of peak) matches that
                                   # convention. Costs 6 of 216 frames.
OUTLIER_MM = 8.0                   # drop a grasp if recorded pose is >8mm
                                   # off its commanded pose (bad pose record)
MIRROR_S2_IN_OVERLAY = False        # column 3: show s2 mirrored L-R, since
                                   # the two pads face each other (display only)

# ------------------------------------------------------------ pad roll ----
# PAD ROLL GEOMETRY (added 2026-08-05) — ONE definition of "where does the
# 22x37 footprint actually lie", used by the splat (build_canvas._splat_one),
# by every drawing that outlines a pad, and by validation.py's round-trip
# sampler. Before this, the splat rotated (2026-08-04) but the DRAWINGS and
# the SAMPLER did not, so a rolled run was painted correctly and then
# outlined, sized and re-sampled as though it were upright.
#
# A "basis" is ((ay, az), (uy, uz)) — the pad's own ACROSS (4 columns,
# PITCH_Y) and UP (7 rows, PITCH_Z) unit vectors in the world Y-Z plane, as
# returned by load_pad_bases() from the MEASURED pad_actual_R. basis=None
# means upright, and every helper below then reduces exactly to the old
# axis-aligned arithmetic, so upright runs are bit-for-bit unchanged.
FLAT_BASIS = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
ROLL_DEADBAND_DEG = 0.05     # FK noise on a truly upright pad is ~0.0005 deg


def pad_roll_deg(basis):
    """Signed roll of the pad in the Y-Z plane, degrees, from its across-axis.
    0 = upright. Sign follows the measured basis, NOT the commanded angle."""
    if basis is None:
        return 0.0
    (ay, az), _ = basis
    return float(np.degrees(np.arctan2(float(az), float(ay))))


def is_flat(basis):
    """True if this pad should be treated as upright. Without the deadband,
    ordinary FK noise (8e-06 deg on the 2026-08-04 flat run) would push every
    upright run onto the rotated code path."""
    return basis is None or abs(pad_roll_deg(basis)) < ROLL_DEADBAND_DEG


def pad_half_extents(basis=None):
    """(half_y, half_z) of the pad footprint's AXIS-ALIGNED bounding box, mm.

    A rolled pad needs more room than a flat one: its bounding box grows to
    (W|cos| + H|sin|) x (W|sin| + H|cos|). Sizing a canvas for the flat 22x37
    clips the rotated footprint and turns the coverage map into an octagon
    (seen 2026-08-04 on the -25 deg run). At 0 deg this returns exactly
    (PAD_W/2, PAD_H/2)."""
    if is_flat(basis):
        return PAD_W / 2.0, PAD_H / 2.0
    (ay, az), _ = basis
    c, s = abs(float(ay)), abs(float(az))
    return (PAD_W * c + PAD_H * s) / 2.0, (PAD_W * s + PAD_H * c) / 2.0


def pad_corners(oy, oz, basis=None):
    """Closed 5-point outline (Y, Z) of the pad footprint centred at (oy, oz).
    Corners walk the pad's OWN axes, so at roll the outline is the true
    rotated rectangle instead of an axis-aligned box that no longer contains
    the painted taxels."""
    a, u = (FLAT_BASIS if is_flat(basis)
            else (np.asarray(basis[0], float), np.asarray(basis[1], float)))
    hw, hh = PAD_W / 2.0, PAD_H / 2.0
    sgn = [(-1, -1), (+1, -1), (+1, +1), (-1, +1), (-1, -1)]
    Y = np.array([oy + sa * hw * a[0] + su * hh * u[0] for sa, su in sgn])
    Z = np.array([oz + sa * hw * a[1] + su * hh * u[1] for sa, su in sgn])
    return Y, Z


def rotated_footprint_index(cell_y, cell_z, oy, oz, tax, basis):
    """Which canvas cells lie under a ROLLED pad, and which taxel owns each.

    Returns (inside, ri, ci):
      inside : (nz, ny) bool canvas mask of the rotated 22x37 footprint
      ri, ci : 1-D arrays, one entry per True cell of `inside` in C order,
               giving that cell's NEAREST taxel row and column.

    Nearest-taxel (Voronoi) on the pad's own lattice, identical to the flat
    rule but measured along the pad's axes. This is the SINGLE definition of
    the rotated paint rule: build_canvas._splat_one paints with it and
    validation._sample_canvas inverts it, so the two cannot drift apart."""
    (ay, az), (uy, uz) = basis
    LY, LZ = np.meshgrid(np.asarray(cell_y, float) - oy,
                         np.asarray(cell_z, float) - oz)      # (nz, ny)
    a = LY * ay + LZ * az                  # coordinate ACROSS the pad, mm
    b = LY * uy + LZ * uz                  # coordinate UP the pad,     mm
    inside = (np.abs(a) <= PAD_W / 2.0) & (np.abs(b) <= PAD_H / 2.0)
    if not inside.any():
        return inside, np.empty(0, int), np.empty(0, int)
    _uy = tax[0, :, 0]          # 4 column centres, pad-local (CAL applied)
    _uz = tax[:, 0, 1]          # 7 row centres,    pad-local
    ci = np.argmin(np.abs(a[inside][:, None] - _uy[None, :]), axis=1)
    ri = np.argmin(np.abs(b[inside][:, None] - _uz[None, :]), axis=1)
    return inside, ri, ci

# ---------------------------------------------------------------- colour ----
# COLOUR-SCALE POLICY (added 2026-08-04) — one source of truth, like
# HOLD_FRAC and SUBTRACT_BASELINE above, so heatmaps.py, temporal_snapshots.py
# and this module can never silently disagree about how a figure is scaled.
# Until now heatmaps.py auto-scaled per grasp while temporal_snapshots.py
# shared one scale across the run, which is exactly why panels from the two
# were not comparable by eye.
#
# Read from Data/plot_scale.json so the GUI checkbox and a standalone
# `python3 heatmaps.py <run>` behave identically. Three modes:
#   shared=False, fixed=None -> per-figure autoscale        (old heatmaps)
#   shared=True,  fixed=None -> one scale across the run    (recommended)
#   fixed=<number>           -> that scale in EVERY test    (cross-run compare)
# 2400 matches Paper 2's tactile-count figures, so sim panels can sit beside
# the published ones.
PLOT_SCALE_JSON = os.path.expanduser(
    "~/Paper3_Simulation/Data/plot_scale.json")


def load_plot_scale():
    """Return (shared: bool, fixed_vmax: float|None). Never raises."""
    try:
        with open(PLOT_SCALE_JSON) as f:
            d = json.load(f)
        fx = d.get("fixed_vmax", None)
        fx = float(fx) if fx not in (None, "", False) else None
        if fx is not None and fx <= 0:
            fx = None
        return bool(d.get("shared", False)), fx
    except Exception:
        return False, None


def resolve_vmax(local_max, run_max=None):
    """Pick the colour ceiling for one panel/figure.

    local_max : max of the data actually being drawn
    run_max   : max across the whole run (pass None if not computed)
    Returns (vmax, label) — the label goes on the colorbar so a reader can
    always tell WHICH scale they are looking at."""
    shared, fixed = load_plot_scale()
    if fixed is not None:
        return fixed, f"fixed {fixed:g}"
    if shared and run_max and run_max > 0:
        return float(run_max), f"shared across run, vmax={run_max:.0f}"
    lm = float(local_max) if (local_max and local_max > 0) else 1.0
    return lm, f"auto, vmax={lm:.0f}"


def run_hold_peak(run_dir):
    """Largest SINGLE-TAXEL value across every grasp and both sensors in a
    run, using the same hold-average the figures draw. This is the ceiling
    for 'shared across run' mode. Returns 0.0 if nothing readable."""
    peak = 0.0
    for f in sorted(glob.glob(os.path.join(run_dir,
                                           "*_pt*_s?_tactile_maps.csv"))):
        try:
            m, _, _ = hold_average(f)
            peak = max(peak, float(np.max(m)))
        except Exception:
            continue
    return peak


BASE_FRAC = 0.05                   # frames with sum <= 5% of peak = baseline
SUBTRACT_BASELINE = True           # ON 2026-07-29: removes the pad-locked
                                   # sensor floor (~1.3% of a map) and the
                                   # unphysical negatives at the fade edge.
                                   # Does NOT fix the row-gain spread (46->48%).

# Which grasp is the INITIAL (input) frame for column 4 / the training pair.
#   "pt00"   = the point the GUI DESIGNED as the initial pad pose (default)
#   "first"  = lowest ptNN PRESENT = where the sweep actually started
#   "center" = grasp nearest the sweep centroid
#   "ptNN"   = pin any other point explicitly
#
# WHY THE DEFAULT CHANGED (2026-08-08). It was "first", which resolves to the
# lowest ptNN that EXISTS, not the one that was asked for. On
# run_20260808_131104_obj0_pad45, pt00 failed its free move to UP and was never
# collected (execution_ledger.json: exec_stage "free_move_to_up"), so "first"
# silently became pt01 — a grid CORNER. The exported pair then carried 32.3 mm
# of extension right and up against 0.0 left and down, and nothing in the file
# said the designed point was missing.
INITIAL_GRASP = os.environ.get("STITCH_INITIAL_GRASP", "pt00")

# A run that LOST its designed initial grasp does not produce a training pair.
# The stitch itself is still valid data, so the FIGURE is still drawn (and
# names the point it fell back to); only export_pair refuses. Set "1" to
# export anyway — the substitution is then recorded in meta as
# initial_status_<sensor> rather than hidden.
ALLOW_INITIAL_FALLBACK = os.environ.get("STITCH_ALLOW_FALLBACK", "0") == "1"

# ---- THE TRAINING PAIR HAS ONE FIXED CANVAS (2026-08-09) -------------------
# build_canvas cuts its canvas to fit whatever the sweep happened to cover, in
# world Y/Z. That is right for the FIGURES — they are pictures of a particular
# run — and wrong for the training pair, which has to be one tensor shape:
# four runs so far gave 74x74, 87x48, 99x99 and 116x116, none of which can be
# batched with any other.
#
# So the pair gets its own canvas, pinned on three choices:
#
#   CELL 1.0 mm      still ~5x finer than the 5.5 x 5.29 mm taxel pitch, so
#                    nothing real is lost; it only sets how finely a rotated
#                    footprint's edge can be traced.
#   SIZE 96 x 96 mm  the pad (22 x 37) plus ~30 mm of margin all round, which
#                    lands on Paper 2's 30 mm haptic safe zone — the distance
#                    at which extrapolated contact was measured to stop
#                    tracking reality. The canvas extent IS the model's
#                    prediction extent, so this is a claim, not a container.
#   PAD FRAME        axes along the pad's own edges, centred on the designed
#                    initial grasp. The model's output must arrive in the
#                    frame the regrasp is commanded in (Paper 2's CDT maps
#                    dx, dy onto tool0.Y / tool0.Z). A world-aligned canvas
#                    would return the answer in a frame that depends on how
#                    the pad happened to be rolled, and the caller would have
#                    to un-rotate by an angle the model was never given.
#
# The FIGURES are untouched: they keep the world-aligned, fit-to-sweep canvas.
PAIR_RES_MM = float(os.environ.get("STITCH_PAIR_RES_MM", "1.0"))
PAIR_SIZE_MM = float(os.environ.get("STITCH_PAIR_SIZE_MM", "96.0"))
PAIR_PAD_FRAME = os.environ.get("STITCH_PAIR_PAD_FRAME", "1") == "1"
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


def _read_tactile_csv(csv_path):
    """Read a tactile CSV, TOLERATING corrupt lines.

    Berith's per-frame writer occasionally collides two writes into one
    line (e.g. "101.933339,6116102.816672,6169,..." — frame N's number
    fused with frame N+1's timestamp), giving a row with 31 fields instead
    of 30. Seen ~once per run, in s1 both times so far, and it lands in
    BOTH the per-grasp and the cumulative file.

    One bad frame out of ~325 is irrelevant to a hold-average, so skip it
    and say so, rather than aborting the whole stitch."""
    n_bad = 0
    try:
        with open(csv_path) as f:
            n_lines = sum(1 for _ in f) - 1          # minus header
    except Exception:
        n_lines = None
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except TypeError:                                # pandas < 1.3
        df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=False)
    if n_lines is not None:
        n_bad = max(0, n_lines - len(df))
    if n_bad:
        name = os.path.basename(csv_path)
        pct = 100.0 * n_bad / max(1, n_lines)
        print(f"[stitch] WARNING {name}: skipped {n_bad} corrupt line(s) "
              f"of {n_lines} ({pct:.1f}%) — writer race, see "
              f"_read_tactile_csv")
        if pct > 5.0:
            print(f"[stitch] WARNING {name}: that is a LOT of bad lines; "
                  f"this grasp's hold-average may not be trustworthy")
    return df


def hold_average(csv_path):
    """(map_7x4, n_hold_frames, peak_sum) for one tactile CSV.

    BASELINE-SUBTRACTED: each grasp's own pre/post-contact frames
    (taxel sum <= 5% of that grasp's peak) define a per-taxel baseline —
    the fixed sensor pattern visible even with the gripper open. It is
    locked to the pad, so without subtraction stitching smears a copy of
    it across the whole grid and buries the real contact structure."""
    df = _read_tactile_csv(csv_path)
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


def save_hold_averages(run_dir, verbose=True):
    """Write every grasp's 28-number hold-average to ONE tidy CSV.

    <run_dir>/hold_average_maps.csv — one row per grasp per sensor:
        grasp, sensor, n_hold_frames, peak_sum, map_sum, hold_frac,
        baseline_subtracted, t_r0c0 ... t_r6c3
    Taxel columns are row-major with r0 = pad BOTTOM, matching the 7x4
    array and the origin="lower" plots.

    This is the exact map that gets heat-mapped and stitched — same
    hold_average() call — so the file is the numeric ground truth behind
    both figures, not a re-derivation."""
    files = sorted(glob.glob(os.path.join(run_dir, "*_pt*_s1_tactile_maps.csv")))
    if not files:
        return None
    cols = [f"t_r{r}c{c}" for r in range(N_ROWS) for c in range(N_COLS)]
    rows = []
    for f1 in files:
        key = _pt_key(os.path.basename(f1)) or os.path.basename(f1)
        for sensor in ("s1", "s2"):
            path = f1 if sensor == "s1" else f1.replace("_s1_", "_s2_")
            if not os.path.exists(path):
                continue
            m, nfr, peak = hold_average(path)
            rec = {"grasp": key, "sensor": sensor, "n_hold_frames": nfr,
                   "peak_sum": round(float(peak), 2),
                   "map_sum": round(float(m.sum()), 2),
                   "hold_frac": HOLD_FRAC,
                   "baseline_subtracted": bool(SUBTRACT_BASELINE)}
            rec.update({c: round(float(v), 4)
                        for c, v in zip(cols, m.reshape(-1))})
            rows.append(rec)
    if not rows:
        return None
    out = os.path.join(run_dir, "hold_average_maps.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    if verbose:
        print(f"[maps] saved {out} ({len(rows)} rows = "
              f"{len(files)} grasps x 2 sensors)")
    return out


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


def _tool_offset_z(run_dir):
    """Flange->pad distance used by the run, in metres. Every run records it;
    fall back to the Ø26 value only as a last resort."""
    for name, key in (("reachability_report.json", "TOOL_OFFSET_Z"),
                      ("pad_truth_probe.json", "TOOL_OFFSET_Z_used_m")):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    v = float(json.load(f)[key])
                if 0.05 < v < 0.30:
                    return v
            except Exception:
                pass
    return 0.15657


def _pad_from_ee(pts, tool_z):
    """TRUE pad-face centre per grasp, from the recorded flange pose.

    WHY THIS EXISTS (2026-08-04)
    ----------------------------
    pad_actual_pos_m is NOT the pad centre. Measured against the same run's
    pad_truth_probe.json it is the sensor CASE prim at OPEN grip, so it is
    wrong by two independent amounts:
        open->closed finger swing   13.04 mm
        pad centre above the case   22.10 mm   (PAD_CENTER_ABOVE_CASE_M)
        --------------------------------------
        total                       35.14 mm in Z  (+14.2 mm in Y)
    Multi-grasp sweeps hid this: the error is constant, so _reanchor_to_gui
    removed it. But that helper needs >= 2 matching points, so SINGLE-point
    runs kept the raw offset and plotted ~35 mm too high.

    The flange pose has no such problem:
        pad_centre = ee_world + R_pad[:, 2] * TOOL_OFFSET_Z
    Because it goes through the pad's own rotation it stays exact when the
    pad is ROLLED. Checked against both runs of 2026-08-04: agrees with the
    commanded pose to 0.03 mm flat AND at 20 deg roll."""
    offs = {}
    for p in pts:
        key = _pt_key(p.get("tag", ""))
        ee, R = p.get("ee_world_m"), p.get("pad_actual_R")
        if not (key and ee and R):
            continue
        try:
            R = np.asarray(R, float)
            pad = np.asarray(ee, float) + R[:, 2] * float(tool_z)
            offs[key] = (float(pad[1]) * 1000.0, float(pad[2]) * 1000.0)
        except Exception:
            continue
    return offs


def load_offsets(run_dir):
    """Return ({ptNN: (y_mm, z_mm)}, source_string) in a common frame.

    Tries several sources IN ORDER and skips any whose offsets are
    DEGENERATE (all grasps within ~1 mm of one spot — a pose-recording
    bug we have actually seen, where pad_actual stayed stuck at the
    startup pose):
      0. ee_world_m + R*TOOL_OFFSET_Z          (TRUE pad centre — preferred)
      1. pose_history.json  pad_actual_pos_m   (open-grip CASE prim: biased!)
      2. pose_history.json  ee_world_m         (EE differences == pad differences)
      3. pose_history.json  pad_desired_pos_m  (initial + commanded offset)
      4. run config points                     (commanded offsets)
    Source 0 was added 2026-08-04 and supersedes 1 — see _pad_from_ee for the
    35 mm bias it removes. 1-3 are kept as fallbacks for older runs whose
    pose_history lacks pad_actual_R.
    The chosen source is printed and shown in the figure title."""
    candidates = []
    ph = os.path.join(run_dir, "pose_history.json")
    if os.path.exists(ph):
        with open(ph) as f:
            data = json.load(f)
        pts = data.get("points", [])
        true_offs = _pad_from_ee(pts, _tool_offset_z(run_dir))
        if true_offs:
            candidates.append((true_offs, "pose_history.json [pad centre "
                                          "from EE+FK]"))
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


def load_pad_bases(run_dir):
    """{ptNN: ((ay, az), (uy, uz))} — the pad's OWN axes in the world Y-Z
    plane, as unit vectors, MEASURED from pose_history's pad_actual_R.

    WHY (2026-08-04): the splat paints the 7x4 map on an axis-aligned
    lattice, which is only right while the pad is vertical. Once the pad is
    ROLLED (GRASP_ROT_DEG) the 22x37 footprint must land rotated, or the
    stitched map shows a tilted grasp as though it were upright.

    Taken from the RECORDED pose, not the commanded angle, so if the arm
    does not reach the requested roll the figure still tells the truth.
      across (4 columns, PITCH_Y) = R[:, 0] projected to (Y, Z)
      up     (7 rows,    PITCH_Z) = -R[:, 2] projected to (Y, Z)
    Verified on 2026-08-04: flat run gives (1,0)/(0,1); the -20 deg run
    gives (0.9397, 0.3420) = (cos20, sin20)."""
    out = {}
    ph = os.path.join(run_dir, "pose_history.json")
    if not os.path.exists(ph):
        return out
    try:
        with open(ph) as f:
            pts = json.load(f).get("points", [])
    except Exception:
        return out
    for p in pts:
        key = _pt_key(p.get("tag", ""))
        R = p.get("pad_actual_R")
        if not (key and R):
            continue
        try:
            R = np.asarray(R, float)
            a = np.array([R[1, 0], R[2, 0]])       # across the pad
            u = np.array([-R[1, 2], -R[2, 2]])     # up the pad
            na, nu = np.linalg.norm(a), np.linalg.norm(u)
            if na < 1e-6 or nu < 1e-6:
                continue
            out[key] = (a / na, u / nu)
        except Exception:
            continue
    return out


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
    Row 0 is the physical BOTTOM (-Z) of the pad — proven 2026-07-22 by four
    independent judges (handoff 6.2 3.4). Displays use origin="lower"."""
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


def resolve_initial(res, which=None):
    """Which grasp is the INITIAL (input) frame — and whether it is the one
    that was ASKED for.

    Returns {"index", "key", "requested", "status", "note"} with status:
        "designed" — the requested grasp is present and was used
        "explicit" — "first"/"center" were requested BY NAME, so whatever they
                     resolve to IS the answer, not a substitution
        "fallback" — the requested grasp is NOT in this run; a substitute was
                     used and the caller must decide what to do about it

    The status is the whole point. The old `_initial_index` returned a bare
    index, so "the designed initial grasp is missing" and "the run is healthy"
    came back looking identical, and a training pair built on a corner grasp
    was written with no complaint."""
    which = which if which is not None else INITIAL_GRASP
    keys = [g[0] for g in res["grasps"]]
    if not keys:
        raise ValueError("resolve_initial: this run has no grasps")

    def _first_index():
        nums = []
        for i, (key, _o, _m) in enumerate(res["grasps"]):
            try:
                nums.append((int(str(key)[2:]), i))
            except ValueError:
                nums.append((10**6 + i, i))
        return min(nums)[1]

    if which == "center":
        i = res["center_index"]
        return {"index": i, "key": keys[i], "requested": which,
                "status": "explicit",
                "note": "grasp nearest the sweep centroid, as requested"}
    if which == "first":
        i = _first_index()
        return {"index": i, "key": keys[i], "requested": which,
                "status": "explicit",
                "note": "lowest ptNN present, as requested"}
    if which in keys:
        i = keys.index(which)
        return {"index": i, "key": which, "requested": which,
                "status": "designed", "note": ""}

    i = _first_index()
    return {"index": i, "key": keys[i], "requested": which,
            "status": "fallback",
            "note": (f"designed initial grasp {which!r} is NOT in this run "
                     f"({len(keys)} present: {keys[0]}..{keys[-1]}); "
                     f"fell back to {keys[i]}")}


def _composite_extended(res, which=None, init=None):
    """INITIAL-frame composite: the chosen grasp's own map inside its pad
    footprint, the stitched extension everywhere outside it.

    Reuses res["paint"] so it lands on the SAME grid as res["canvas"] — no
    new geometry, so it cannot drift from the real paint step.

    Returns (composite, (oy, oz), key, ext_mm) where ext_mm says how far the
    canvas reaches BEYOND each edge of the initial pad frame:
        {"up":, "down":, "left":, "right":}  in mm.
    Measured PER SIDE, never assumed symmetric: a sweep that only climbed in
    Z extends up and not down, and the figure must show exactly that."""
    i = (init if init is not None else resolve_initial(res, which))["index"]
    inp, icnt, _sq = res["paint"]([i])
    composite = np.where(icnt > 0, inp, res["canvas"])
    key, (oy, oz), _m = res["grasps"][i]
    # SELF-CHECK: outside the initial pad footprint the composite must BE the
    # stitched canvas (column 1), cell for cell. If this ever prints False the
    # two columns really have diverged and nothing below should be trusted.
    _out = icnt <= 0
    print(f"[stitch] composite {key}: interior {int((icnt > 0).sum())} cells "
          f"(raw, max {float(inp.max()):.0f}) | exterior {int(_out.sum())} "
          f"cells (stitch, max {float(res['canvas'].max()):.0f}) | exterior "
          f"identical to column 1: "
          f"{bool(np.array_equal(composite[_out], res['canvas'][_out]))}")
    y0, y1, z0, z1 = res["extent"]
    # Measured from the pad's REAL bounding box. At roll the upright 22x37
    # box no longer contains the painted taxels, so the old form overstated
    # every side (at -25 deg by 6.8 mm in Y and 4.7 mm in Z).
    _b = res.get("bases", {}).get(key)
    _hy, _hz = pad_half_extents(_b)
    ext_mm = {"left":  (oy - _hy) - y0,
              "right": y1 - (oy + _hy),
              "down":  (oz - _hz) - z0,
              "up":    z1 - (oz + _hz)}
    return composite, (float(oy), float(oz)), key, ext_mm


def _rot2(vy, vz, deg):
    """Rotate a (Y, Z) vector by deg in the Y-Z plane."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return (float(vy) * c - float(vz) * s, float(vy) * s + float(vz) * c)


def to_pad_frame(grasps, bases, anchor_key):
    """Re-express a run in the ANCHOR pad's own frame.

    Rather than rotating the canvas — which would make the cell-centre axes
    2-D and break the fast separable splat — this rotates the DATA the other
    way: every pad centre turns about the anchor's centre by -roll, and every
    basis turns by -roll too. Painting the result on an ordinary axis-aligned
    grid then produces exactly the pad-frame canvas, and for the usual case
    where every pad carries the same roll the rotated bases come out FLAT, so
    the splat takes its fast path and the anchor's imprint lands upright.

    Returns (grasps', bases', roll_deg_removed).
    """
    keys = [g[0] for g in grasps]
    if anchor_key not in keys:
        raise RuntimeError(
            f"to_pad_frame: anchor {anchor_key!r} not among {len(keys)} grasps")
    a_oy, a_oz = grasps[keys.index(anchor_key)][1]
    th = pad_roll_deg(bases.get(anchor_key))
    if abs(th) < ROLL_DEADBAND_DEG:
        return list(grasps), dict(bases), 0.0
    g2 = []
    for key, (oy, oz), m in grasps:
        dy, dz = _rot2(oy - a_oy, oz - a_oz, -th)
        g2.append((key, (a_oy + dy, a_oz + dz), m))
    b2 = {}
    for key, (a, u) in bases.items():
        b2[key] = (np.array(_rot2(a[0], a[1], -th)),
                   np.array(_rot2(u[0], u[1], -th)))
    return g2, b2, float(th)


def pad_tip_edge(oy, oz, basis=None):
    """The pad's OUTER TIP edge — the free end of the finger.

    Returns ((Y0, Y1), (Z0, Z1)), the two corners of the short edge furthest
    from the palm, plus the outward direction as a third element.

    WHICH EDGE, AND HOW WE KNOW. The palm sits ABOVE the pad and the fingers
    hang down from it: pad_truth_probe.json measures
    measured_palm_above_pad_mm = 146.4, and the GUI draws the palm line above
    the pad for the same reason. So the tip is the pad's -u edge, where u is
    the pad's own long axis. At roll the edge turns with the pad, which is why
    this is derived from the basis rather than being "the bottom two corners".

    Why it is worth drawing: that edge is the part of the sensor that leaves
    the object first, so contact dying there is geometry, not a sensor fault —
    and it is the end that would strike anything below the grasp."""
    a, u = (FLAT_BASIS if is_flat(basis)
            else (np.asarray(basis[0], float), np.asarray(basis[1], float)))
    hw, hh = PAD_W / 2.0, PAD_H / 2.0
    Y = (oy - hw * a[0] - hh * u[0], oy + hw * a[0] - hh * u[0])
    Z = (oz - hw * a[1] - hh * u[1], oz + hw * a[1] - hh * u[1])
    return Y, Z, (-u[0], -u[1])


def _draw_pad_tip(ax, oy, oz, basis, label=None, z=10, arrow_mm=5.0, lw=4.0):
    """Mark the pad's tip edge: a thick bar plus a small outward arrow.

    arrow_mm and lw are exposed because the same marker is drawn on panels of
    very different scale: on the stitched map the view is ~90 mm across and a
    5 mm arrow reads clearly, while the heatmap pose panel spans a whole
    140 mm rod and the identical marker all but vanishes."""
    (Y0, Y1), (Z0, Z1), (uy, uz) = pad_tip_edge(oy, oz, basis)
    ax.plot([Y0, Y1], [Z0, Z1], "-", color="#7b2fbe", lw=lw,
            solid_capstyle="butt", zorder=z, label=label)
    my, mz = (Y0 + Y1) / 2.0, (Z0 + Z1) / 2.0
    ax.annotate("", xy=(my + arrow_mm * uy, mz + arrow_mm * uz),
                xytext=(my, mz),
                arrowprops=dict(arrowstyle="-|>", color="#7b2fbe",
                                lw=max(1.2, lw * 0.35)),
                zorder=z)
    return (my, mz), (uy, uz)


def build_canvas(run_dir, sensor, res_mm=1.0, cal=None, verbose=True,
                 frame="world", fixed_size_mm=None, anchor_key=None):
    """Stitch all grasps of one sensor onto a mm canvas.

    frame="world"        canvas axes are world Y/Z, extent cut to fit the
                         sweep. The figures use this and are unchanged.
    frame="pad"          canvas axes follow the ANCHOR pad's own edges and
                         the canvas is centred on it. With fixed_size_mm the
                         extent is pinned, so every run yields one tensor
                         shape regardless of roll, step or grid size.
    fixed_size_mm        square canvas of this side length, mm. A run whose
                         data would not fit raises rather than being cropped.
    anchor_key           which grasp is the centre; defaults to INITIAL_GRASP.

    Returns a dict with the canvas, coverage count, extent, the per-grasp
    list, the center-grasp index/key, an overlap-disagreement score, and a
    paint(indices) helper re-rendering any subset on the SAME grid."""
    offs, src = load_offsets(run_dir)
    if not offs:
        raise RuntimeError("no pad offsets found "
                           "(need pose_history.json or the run's config copy)")
    # The EE+FK source is ALREADY the true pad centre in world/GUI coordinates,
    # so re-anchoring it would subtract a shift that is not there. Only the
    # legacy biased sources still need _reanchor_to_gui — and that helper needs
    # >= 2 matching points, which is why single-point runs used to come out
    # ~35 mm high before the EE+FK source existed.
    if "from EE+FK" in src:
        gui_frame = True
        if verbose:
            print("[stitch] pad centre taken from EE+FK — already in the GUI "
                  "frame, no re-anchor needed (exact for 1 grasp and for "
                  "rolled pads)")
    else:
        gui_frame = "pose_history" not in src    # config offsets are GUI-frame
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
    _bases0 = load_pad_bases(run_dir)

    # ---- PAD FRAME: rotate the DATA, not the canvas (see to_pad_frame) -----
    frame_roll = 0.0
    _anchor = anchor_key or INITIAL_GRASP
    # If the requested anchor is not in this run, fall back to the first
    # grasp rather than raising. to_pad_frame used to raise here, which meant
    # a run whose pt00 never executed died with a traceback out of
    # export_pair instead of the clean "NO training_pair.npz written"
    # refusal — the refusal exists precisely for that case and never got the
    # chance to run. resolve_initial still sees the substitution downstream
    # and export_pair still refuses; it just does so in words.
    _keys = [g[0] for g in grasps]
    if frame == "pad" and _anchor not in _keys and _keys:
        if verbose:
            print(f"[stitch {sensor}] anchor {_anchor!r} is not in this run "
                  f"({len(_keys)} grasps); centring the canvas on {_keys[0]} "
                  f"so the export can report the substitution properly")
        _anchor = _keys[0]
    if frame == "pad":
        grasps, _bases0, frame_roll = to_pad_frame(grasps, _bases0, _anchor)
        if verbose and abs(frame_roll) > ROLL_DEADBAND_DEG:
            print(f"[stitch {sensor}] canvas in the PAD frame: run rotated by "
                  f"{-frame_roll:+.2f} deg about {_anchor}, so its imprint "
                  f"lands upright and the axes are the pad's own")

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
    # A ROLLED pad needs a bigger canvas than a flat one (see
    # pad_half_extents). Flat runs are untouched: at 0 deg this reduces
    # exactly to PAD_W / PAD_H.
    _fy, _fz = PAD_W / 2.0, PAD_H / 2.0
    for _b in _bases0.values():
        _hy, _hz = pad_half_extents(_b)
        _fy, _fz = max(_fy, _hy), max(_fz, _hz)
    half_y = (max(ys) - min(ys)) / 2.0 + _fy + MARGIN
    half_z = (max(zs) - min(zs)) / 2.0 + _fz + MARGIN

    # ---- FIXED EXTENT: one tensor shape, or a hard error ------------------
    # Silently cropping would hand the trainer a target with its edges cut off
    # and nothing in the file to say so, which is the same class of fault as
    # the initial-grasp substitution. So this raises instead.
    if fixed_size_mm:
        need_y, need_z = 2 * half_y, 2 * half_z
        if need_y > fixed_size_mm + 1e-6 or need_z > fixed_size_mm + 1e-6:
            raise RuntimeError(
                f"this run needs {need_y:.1f} x {need_z:.1f} mm but the pair "
                f"canvas is fixed at {fixed_size_mm:.1f} mm square. Reduce the "
                f"grid (a centred grid spans PAD + 2*n*step: "
                f"{PAD_W:.0f}+2*nx*step across, {PAD_H:.0f}+2*ny*step along), "
                f"or raise STITCH_PAIR_SIZE_MM — but then every earlier pair "
                f"has to be re-exported at the new size.")
        half_y = half_z = float(fixed_size_mm) / 2.0
        if frame == "pad":                 # centre on the anchor, not the mean
            _ak = [g[0] for g in grasps].index(_anchor) \
                if _anchor in [g[0] for g in grasps] else None
            if _ak is not None:
                cy, cz = grasps[_ak][1]

    ny = int(np.ceil(2 * half_y / res_mm))
    nz = int(np.ceil(2 * half_z / res_mm))
    y0, z0 = cy - half_y, cz - half_z                        # origin (mm)

    # Every canvas cell is assigned to the taxel whose centre is NEAREST.
    # On a regular lattice that is the taxel's true footprint (a Voronoi
    # cell), so the pad tiles EXACTLY: no gap between taxels, no overlap,
    # and the painted region is exactly PAD_W x PAD_H.
    #
    # The previous fixed-block splat used round(PITCH/res) cells. At
    # 0.75 mm/cell that is round(5.5/0.75) = 7 cells = 5.25 mm against a
    # 5.5 mm pitch, which left an unpainted gap column between taxels c2
    # and c3 and made the painted pad 21.0 mm wide instead of 22.0 — so
    # the drawn pad rectangle no longer matched the raw interior.
    _cell_y = y0 + np.arange(ny) * res_mm          # cell centres (mm, world)
    _cell_z = z0 + np.arange(nz) * res_mm
    _uy = tax[0, :, 0]          # 4 column centres, pad-local (CAL already applied)
    _uz = tax[:, 0, 1]          # 7 row centres,    pad-local

    # Pad roll, measured per grasp. Empty dict -> every grasp paints flat.
    _bases = _bases0
    if verbose and _bases:
        _rolls = [abs(pad_roll_deg(b)) for b in _bases.values()]
        if max(_rolls) > ROLL_DEADBAND_DEG:
            print(f"[stitch] pad ROLL applied to the splat, the drawn pad "
                  f"outlines and the canvas size: "
                  f"{min(_rolls):.1f}..{max(_rolls):.1f} deg "
                  f"(measured from pad_actual_R)")

    def _splat_one(oy, oz, m, acc, cnt, sq, basis=None):
        """Paint one grasp, still nearest-taxel (Voronoi) so the pad tiles
        exactly. When the pad is ROLLED the canvas offsets are projected onto
        the pad's OWN axes first, so the 22x37 footprint lands rotated
        (2026-08-04). The fast separable path is kept for the flat case, so
        nothing about existing upright runs changes by even one cell."""
        if is_flat(basis):
            ly = _cell_y - oy                      # pad-local mm
            lz = _cell_z - oz
            sy = np.where(np.abs(ly) <= PAD_W / 2.0)[0]
            sz = np.where(np.abs(lz) <= PAD_H / 2.0)[0]
            if sy.size == 0 or sz.size == 0:
                return
            ci = np.argmin(np.abs(ly[sy][None, :] - _uy[:, None]), axis=0)
            ri = np.argmin(np.abs(lz[sz][None, :] - _uz[:, None]), axis=0)
            sub = m[np.ix_(ri, ci)]
            acc[np.ix_(sz, sy)] += sub
            cnt[np.ix_(sz, sy)] += 1.0
            sq[np.ix_(sz, sy)] += sub * sub
            return

        # rotated: full 2-D pass over the canvas, via the ONE shared rule
        inside, ri, ci = rotated_footprint_index(
            _cell_y, _cell_z, oy, oz, tax, basis)
        if not inside.any():
            return
        vals = m[ri, ci]
        acc[inside] += vals
        cnt[inside] += 1.0
        sq[inside] += vals * vals

    def paint(indices):
        """(canvas, count, sumsq) for the given grasps, on the same grid."""
        acc = np.zeros((nz, ny))
        cnt = np.zeros((nz, ny))
        sq = np.zeros((nz, ny))
        for i in indices:
            key, (oy, oz), m = grasps[i]
            _splat_one(oy, oz, m, acc, cnt, sq, _bases.get(key))
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
            "frame": frame, "frame_roll_deg": frame_roll,
            "anchor_key": (_anchor if frame == "pad" else None),
            "grasps": grasps, "center_index": center_i,
            "center_key": grasps[center_i][0],
            "gui_frame": gui_frame,
            # measured pad roll per grasp ({} when upright / no pose_history).
            # Published so the plots and validation.py outline and re-sample
            # the SAME footprint the splat painted.
            "bases": _bases,
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


class _Tee:
    """Mirror stdout so a run's diagnostics reach BOTH the terminal and disk.
    Needed because the GUI launches stitching in a thread whose stdout the
    user usually never sees (and never at all when running headless)."""

    def __init__(self, stream):
        self._s = stream
        self.buf = []

    def write(self, txt):
        try:
            self._s.write(txt)
        except Exception:
            pass
        self.buf.append(txt)
        return len(txt)

    def flush(self):
        try:
            self._s.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def stitch_run(run_dir, res_mm=1.0):
    """Build + save the stitched maps for s1 and s2, and write every
    diagnostic line to <run_dir>/Stitched/stitch_report.txt.

    Returns the list of saved PNG paths (PNGs ONLY — the GUI imread()s
    whatever this returns, so the .txt must never appear in it)."""
    import time
    tee = _Tee(sys.stdout)
    old = sys.stdout
    sys.stdout = tee
    try:
        made = _stitch_run_body(run_dir, res_mm)
    finally:
        sys.stdout = old            # restore even if the body raised

    try:
        save_hold_averages(run_dir)
    except Exception as e:
        print(f"[maps] could not save hold_average_maps.csv ({e})")

    hdr = ["STITCH REPORT",
           f"written   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
           f"run_dir   : {os.path.abspath(run_dir)}",
           f"res_mm    : {res_mm}",
           f"settings  : INITIAL_GRASP={INITIAL_GRASP!r}  "
           f"SUBTRACT_BASELINE={SUBTRACT_BASELINE}  "
           f"MIRROR_S2_IN_OVERLAY={MIRROR_S2_IN_OVERLAY}  "
           f"OUTLIER_MM={OUTLIER_MM}",
           f"pad/pitch : PAD_W={PAD_W} PAD_H={PAD_H} "
           f"PITCH_Y={PITCH_Y} PITCH_Z={PITCH_Z}",
           "-" * 68, ""]
    body = "".join(tee.buf).rstrip()
    out_dir = os.path.join(run_dir, "Stitched")
    try:
        os.makedirs(out_dir, exist_ok=True)
        rpt = os.path.join(out_dir, "stitch_report.txt")
        with open(rpt, "w") as f:
            f.write("\n".join(hdr) + body + "\n")
        print(f"saved {rpt}")
    except Exception as e:
        print(f"[stitch] could not write stitch_report.txt ({e})")
    return made


def _stitch_run_body(run_dir, res_mm=1.0):
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
        ncols = 4 if show_overlay else 3   # last column = initial + extended

        fig = Figure(figsize=(5.2 * ncols + 0.3, 4.6))
        FigureCanvasAgg(fig)
        ax1 = fig.add_subplot(1, ncols, 1)
        ax2 = fig.add_subplot(1, ncols, 2)
        # ONE shared colour range for column 1 and the last column, so the
        # region outside the initial frame is directly comparable between them.
        v_lo, v_hi = float(np.min(canvas)), float(np.max(canvas))
        if not np.isfinite(v_hi) or v_hi <= v_lo:
            v_lo, v_hi = 0.0, 1.0
        # Colour policy (see load_plot_scale). The TRUE canvas max stays in
        # the title, so a fixed/shared ceiling can never hide that this run
        # peaked far below — or above — what the colours suggest.
        _canvas_max = v_hi
        v_hi, _scale_lbl = resolve_vmax(v_hi, run_hold_peak(run_dir))
        if v_hi <= v_lo:
            v_lo = 0.0
        im1 = ax1.imshow(canvas, cmap="jet", origin="lower",
                         extent=ext, aspect="equal", vmin=v_lo, vmax=v_hi)
        ax1.set_title(f"{sensor} — stitched extended map "
                      f"({len(res['grasps'])} grasps, hold-avg)\n"
                      f"canvas max {_canvas_max:.0f}", fontsize=10)
        ax1.set_xlabel("Y (mm)"); ax1.set_ylabel("Z (mm)")
        fig.colorbar(im1, ax=ax1, shrink=0.85).set_label(
            f"pressure (a.u.) — {_scale_lbl}", fontsize=8)
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
            gbase = res.get("bases", {})
            # POSITIONS stay COMMANDED (that is the point of this column —
            # immune to per-grasp pose noise). ORIENTATION is taken from the
            # MEASURED pad_actual_R, because roll is one constant per run, not
            # per-point noise, and because the sign convention of the
            # commanded angle is not independently confirmed. Flip this line
            # to a commanded angle if you ever want the pure-command view.
            pts = [(gui["cmd"][k][0], gui["cmd"][k][1], gmap[k], gbase.get(k))
                   for k in gmap if k in gui["cmd"]]
            gy = [p[0] for p in pts]; gz = [p[1] for p in pts]
            _phy = max(pad_half_extents(p[3])[0] for p in pts)
            _phz = max(pad_half_extents(p[3])[1] for p in pts)
            hy = (max(gy) - min(gy)) / 2 + _phy
            hz = (max(gz) - min(gz)) / 2 + _phz
            ccy = (max(gy) + min(gy)) / 2; ccz = (max(gz) + min(gz)) / 2
            gny = int(np.ceil(2 * hy / res_mm)); gnz = int(np.ceil(2 * hz / res_mm))
            gy0 = ccy - hy; gz0 = ccz - hz
            _gcell_y = gy0 + np.arange(gny) * res_mm
            _gcell_z = gz0 + np.arange(gnz) * res_mm
            acc = np.zeros((gnz, gny)); gcnt = np.zeros((gnz, gny))
            for oy, oz, m, gb in pts:
                if is_flat(gb):
                    # unchanged fixed-block paint: upright runs render exactly
                    # as they always have, cell for cell
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
                else:
                    # rolled: same rule the stitcher painted with, so this
                    # column and column 1 show the same footprint shape
                    inside, ri, ci = rotated_footprint_index(
                        _gcell_y, _gcell_z, oy, oz, tax, gb)
                    if not inside.any():
                        continue
                    acc[inside] += m[ri, ci]
                    gcnt[inside] += 1.0
            gcanvas = np.where(gcnt>0, acc/np.maximum(gcnt,1.0), 0.0)
            gext = (gy0, gy0+gny*res_mm, gz0, gz0+gnz*res_mm)
            _r3 = [pad_roll_deg(p[3]) for p in pts]
            if max(abs(r) for r in _r3) > ROLL_DEADBAND_DEG:
                print(f"[stitch] column 3 pads drawn ROLLED "
                      f"{min(_r3):+.1f}..{max(_r3):+.1f} deg (measured)")

            ax3 = fig.add_subplot(1, ncols, 3)
            clipped = np.ma.masked_where(~(gcnt>0), gcanvas)
            cmap = plt.cm.jet.copy(); cmap.set_bad(alpha=0.0)
            # SAME colour scale as columns 1 and 4 (2026-08-11). This imshow
            # had no vmin/vmax, so matplotlib autoscaled it to its own data —
            # which made an identical pressure render as a different colour in
            # column 3 than in column 1, and invited exactly the wrong
            # comparison between panels of the same figure.
            ax3.imshow(clipped, cmap=cmap, origin="lower", extent=gext,
                       aspect="equal", vmin=v_lo, vmax=v_hi)
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
            # THE INITIAL PAD FRAME (2026-08-11). Column 3 showed only pad
            # CENTRES as dots, so the one pose the whole run is anchored to —
            # and the one the GUI draws in purple on its preview — was
            # invisible here. Same outline, same colour and same source as
            # column 4, so the two panels cannot disagree about where pt00 sat.
            try:
                _ik = resolve_initial(res)["key"]
                if _ik in gui["cmd"]:
                    _iy, _iz = gui["cmd"][_ik][0], gui["cmd"][_ik][1]
                    _IY, _IZ = pad_corners(_iy, _iz, gbase.get(_ik))
                    ax3.plot(_IY, _IZ, "k-", lw=2.6, alpha=0.85, zorder=7)
                    ax3.plot(_IY, _IZ, "w--", lw=1.4, zorder=8,
                             label=f"initial pad frame ({_ik})")
                    ax3.plot([_iy], [_iz], "o", mfc="w", mec="k", ms=4,
                             mew=0.8, zorder=9)
                    _draw_pad_tip(ax3, _iy, _iz, gbase.get(_ik),
                                  label="pad tip (finger free end)")
            except Exception as _ie:
                print(f"[stitch {sensor}] note: initial pad frame not drawn "
                      f"on column 3 ({_ie})")
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
            # Legend BELOW the axes, not on top of them: at "upper right" it
            # sat over the swept region and hid the very cells the panel
            # exists to show.
            ax3.legend(fontsize=6, loc="upper center",
                       bbox_to_anchor=(0.5, -0.13), ncol=2,
                       frameon=True, borderaxespad=0.0)
        else:
            print("[stitch] NO overlay column: could not read commanded grid "
                  "from config (need gui_config with object + points).")

        # ---- LAST column: INITIAL contact frame + stitched extension -------
        # Resolved ONCE here and passed down, so the panel, its title and the
        # console line cannot end up describing different grasps.
        _init = resolve_initial(res)
        comp, (c_oy, c_oz), c_key, exd = _composite_extended(res, init=_init)
        ax4 = fig.add_subplot(1, ncols, ncols)
        im4 = ax4.imshow(comp, cmap="jet", origin="lower",
                         extent=ext, aspect="equal", vmin=v_lo, vmax=v_hi)
        # The outline walks the pad's OWN axes, so at roll it is the true
        # rotated rectangle around the painted taxels instead of an upright
        # box that cuts two corners off and includes two empty ones.
        c_basis = res.get("bases", {}).get(c_key)
        c_roll = pad_roll_deg(c_basis)
        rY, rZ = pad_corners(c_oy, c_oz, c_basis)
        ax4.plot(rY, rZ, color="k", lw=3.2, alpha=0.75)        # contrast underlay
        ax4.plot(rY, rZ, "w--", lw=1.6,
                 label=(f"initial pad frame ({c_key})" if is_flat(c_basis)
                        else f"initial pad frame ({c_key}, roll {c_roll:+.1f} deg)"))
        ax4.scatter([c_oy], [c_oz], s=18, color="w",
                    edgecolor="k", linewidth=0.6, zorder=6)
        _draw_pad_tip(ax4, c_oy, c_oz, c_basis,
                      label="pad tip (finger free end)")
        # At roll, "up/down/left/right" are world directions measured from the
        # pad's BOUNDING BOX, not from its own edges — say so rather than let
        # the reader assume pad-frame numbers. Nothing is added when upright,
        # so existing figures are unchanged.
        _frm = ("" if is_flat(c_basis)
                else f" [from pad bbox, roll {c_roll:+.1f} deg]")
        # A substituted initial frame is stated ON the figure, in red. The
        # extension numbers below are then one-sided for a reason that has
        # nothing to do with the sweep, and a reader must not have to open
        # execution_ledger.json to find that out.
        _bad = _init["status"] == "fallback"
        ax4.set_title(
            (f"SUBSTITUTED initial frame — {_init['requested']} MISSING\n"
             if _bad else "")
            + f"initial contact ({c_key}) + extended map\n"
            f"extends  up {exd['up']:.1f} / down {exd['down']:.1f} / "
            f"left {exd['left']:.1f} / right {exd['right']:.1f}  mm{_frm}\n"
            f"colour scale shared with col 1 (raw interior may saturate)",
            fontsize=9, color=("crimson" if _bad else "black"))
        ax4.set_xlabel("Y (mm)")
        ax4.legend(fontsize=6, loc="upper right")
        fig.colorbar(im4, ax=ax4, shrink=0.85).set_label("pressure (a.u.)",
                                                         fontsize=8)
        # The taxel conversion divides world mm by a PAD-FRAME pitch, which
        # only means something while the pad is upright. At roll it is
        # withheld rather than printed as a number that looks meaningful.
        _tax_txt = (f"  |  taxels  "
                    f"up={exd['up']/PITCH_Z:.2f} down={exd['down']/PITCH_Z:.2f} "
                    f"left={exd['left']/PITCH_Y:.2f} "
                    f"right={exd['right']/PITCH_Y:.2f}"
                    if is_flat(c_basis) else
                    f"  |  taxels n/a (pad rolled {c_roll:+.1f} deg; world mm "
                    f"do not map onto pad pitches)")
        print(f"[stitch {sensor}] initial frame = {c_key} (INITIAL_GRASP="
              f"{INITIAL_GRASP!r}, {_init['status']}); extension mm  "
              f"up={exd['up']:.1f} down={exd['down']:.1f} "
              f"left={exd['left']:.1f} right={exd['right']:.1f}{_tax_txt}")
        if _bad:
            print(f"[stitch {sensor}] !! {_init['note']}")
            print(f"[stitch {sensor}] !! the figure is still valid data, but "
                  f"export_pair will REFUSE to write a training pair "
                  f"(override: STITCH_ALLOW_FALLBACK=1)")

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


def export_pair(run_dir, res_mm=None, anchor=None,
                anchor_kind=None, out_path=None):
    """Write Stitched/training_pair.npz on the PINNED pair canvas.

    INPUT = the INITIAL grasp alone, TARGET = the full stitch, both on a
    canvas of PAIR_RES_MM cells, PAIR_SIZE_MM square, in the initial pad's own
    frame and centred on it — so every run of every roll, step and grid size
    exports the identical tensor shape. res_mm overrides PAIR_RES_MM only if
    you pass it; the default is the pinned value, NOT the figures' 0.75.

    THREE THINGS CHANGED, all visible in an old npz.

    1. The INPUT used to be painted from res["center_index"] — the grasp
       nearest the sweep CENTROID — while initial_/frame_rect_/frame_poly_/
       extension_mm_ and target_composite_ all described _initial_index's
       grasp. On run_20260808_131104_obj0_pad45 those were pt13 and pt01: one
       file, two different grasps, and frame_rect_s1 pointed 42 mm away from
       the input it claimed to bound. Both now come from resolve_initial.

    2. A run whose DESIGNED initial grasp never executed no longer produces a
       pair at all. Set STITCH_ALLOW_FALLBACK=1 to export anyway; the
       substitution is then recorded in initial_status_<sensor>.

    3. The canvas is pinned (2026-08-09). Runs exported before this are
       74x74, 87x48, 99x99 and 116x116 world-aligned cells and cannot be
       batched with each other or with anything written from now on."""
    res_mm = float(PAIR_RES_MM if res_mm is None else res_mm)
    _frame = "pad" if PAIR_PAD_FRAME else "world"
    out = {}
    meta = {"run_dir": os.path.abspath(run_dir), "res_mm": res_mm,
            "canvas_frame": _frame, "canvas_size_mm": PAIR_SIZE_MM}
    if anchor is not None:
        meta["anchor"] = str(anchor)
        meta["anchor_kind"] = str(anchor_kind or "?")
    refused = []
    for sensor in ("s1", "s2"):
        res = build_canvas(run_dir, sensor, res_mm,
                           frame=_frame, fixed_size_mm=PAIR_SIZE_MM,
                           anchor_key=anchor)
        init = resolve_initial(res, which=anchor)
        if init["status"] == "fallback":
            print(f"[pair {sensor}] !! {init['note']}")
            if not ALLOW_INITIAL_FALLBACK:
                refused.append(f"{sensor}: {init['note']}")
                continue
            print(f"[pair {sensor}] !! exporting anyway "
                  f"(STITCH_ALLOW_FALLBACK=1)")
        inp, icnt, _sq = res["paint"]([init["index"]])
        out[f"input_{sensor}"] = inp
        out[f"input_mask_{sensor}"] = icnt > 0
        out[f"target_{sensor}"] = res["canvas"]
        out[f"target_mask_{sensor}"] = res["count"] > 0
        # second TARGET variant: raw initial grasp inside its own pad frame,
        # stitch outside. Kept ALONGSIDE the full-stitch target so the choice
        # can be made at training time (Block 3), not now.
        comp, (c_oy, c_oz), c_key, exd = _composite_extended(res, init=init)
        out[f"target_composite_{sensor}"] = comp
        meta[f"initial_{sensor}"] = c_key
        meta[f"initial_mode_{sensor}"] = INITIAL_GRASP
        meta[f"initial_status_{sensor}"] = init["status"]
        if init["note"]:
            meta[f"initial_note_{sensor}"] = init["note"]
        # frame_rect stays the AXIS-ALIGNED box for backward compatibility,
        # but is now the rolled pad's true bounding box (identical at 0 deg).
        # frame_poly is the exact footprint; frame_roll_deg says which applies.
        _cb = res.get("bases", {}).get(c_key)
        _chy, _chz = pad_half_extents(_cb)
        meta[f"frame_rect_{sensor}"] = [c_oy - _chy, c_oy + _chy,
                                        c_oz - _chz, c_oz + _chz]
        _pY, _pZ = pad_corners(c_oy, c_oz, _cb)
        meta[f"frame_poly_{sensor}"] = [[float(y), float(z)]
                                        for y, z in zip(_pY, _pZ)]
        meta[f"frame_roll_deg_{sensor}"] = pad_roll_deg(_cb)
        meta[f"extension_mm_{sensor}"] = exd
        meta[f"center_{sensor}"] = res["center_key"]
        meta[f"extent_{sensor}"] = list(res["extent"])
        meta[f"n_grasps_{sensor}"] = len(res["grasps"])
        meta[f"offset_source_{sensor}"] = res["offset_source"]

    ts = os.path.join(run_dir, "temporal_snapshots.json")
    if os.path.exists(ts):
        with open(ts) as f:
            T = json.load(f)
        for sensor in ("s1", "s2"):
            # keyed to the INITIAL grasp, i.e. the one input_<sensor> was
            # painted from. It used to be keyed to center_<sensor>, so on any
            # run where those differed the snapshots belonged to a grasp that
            # appears nowhere else in the file. The array keeps its old name
            # so nothing downstream has to change.
            ck = meta.get(f"initial_{sensor}")
            for tag, entry in T.items():
                if _pt_key(tag) == ck and sensor in entry:
                    snaps = entry[sensor]["snapshots"]
                    out[f"center_temporal_{sensor}"] = np.stack(
                        [np.array(snaps[k])
                         for k in ("p05", "p50", "p95", "post3s")])
                    meta[f"temporal_{sensor}"] = f"included (4,7,4) from {ck}"

    if refused:
        print("[pair] NO training_pair.npz written — "
              + "; ".join(refused))
        print("[pair] the stitched maps and figures are unaffected and still "
              "valid. Re-collect the designed initial point, or export with "
              "STITCH_ALLOW_FALLBACK=1 to accept the substitution knowingly.")
        return None

    npz = out_path or os.path.join(run_dir, "Stitched", "training_pair.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(npz, meta=json.dumps(meta), **out)
    print(f"saved {npz}")
    return npz


def anchor_kinds(run_dir, sensor="s1", res_mm=None, tol_mm=1.0):
    """Classify every grasp as INTERIOR or EDGE of its own sweep.

    An anchored pair re-centres the 96 mm canvas on the chosen grasp. For a
    grasp in the middle of the sweep the box is filled with measured target on
    every side. For one on the rim, half the box hangs over ground the robot
    never visited, so that half of the target is blank -- not wrong, but a
    different and easier question ("extend inward") than the centred one
    ("extend outward in all directions").

    INTERIOR means: at least one other grasp lies further out in EACH of the
    four directions. On a 5x5 grid that is exactly the middle 3x3.

    The test is done on pad-centre positions rather than on the mask, because
    positions are exact while a mask is a rasterised approximation of them.
    """
    res = build_canvas(run_dir, sensor, res_mm or PAIR_RES_MM,
                       frame="pad" if PAIR_PAD_FRAME else "world",
                       fixed_size_mm=None, verbose=False)
    g = [(k, o[0], o[1]) for k, o, _m in res["grasps"]]
    out = {}
    for k, y, z in g:
        left = any(oy < y - tol_mm for _k, oy, _oz in g)
        right = any(oy > y + tol_mm for _k, oy, _oz in g)
        down = any(oz < z - tol_mm for _k, _oy, oz in g)
        up = any(oz > z + tol_mm for _k, _oy, oz in g)
        out[k] = ("interior" if (left and right and down and up) else "edge")
    return out


def export_anchor_pairs(run_dir, res_mm=None, include_edge=False,
                        verbose=True):
    """Write one pair per grasp into <run>/Stitched/pairs/.

    WHY THIS IS FREE. The pairs are DERIVED, not collected: a run folder holds
    the per-grasp tactile CSVs and pose_history, and training_pair.npz is only
    one view of them. Re-centring on a different grasp costs no robot time and
    needs no change to how data is gathered.

    WHAT IT IS NOT. These are not independent samples. Every anchor from one
    sweep shares an object, a pose and the same underlying measurements, so
    this is closer to augmentation than to new data. dataset.split_by_run
    keeps all of a run's anchors on one side of the train/val split, which is
    what stops that from turning into a flattering validation score.

    training_pair.npz is left exactly where it is -- untouched, still the
    pt00 pair -- so every existing scan and every already-exported run stays
    valid. The extra pairs live in a subfolder and can be included or ignored
    with one flag at training time.
    """
    res_mm = float(PAIR_RES_MM if res_mm is None else res_mm)
    kinds = anchor_kinds(run_dir, "s1", res_mm)
    out_dir = os.path.join(run_dir, "Stitched", "pairs")
    os.makedirs(out_dir, exist_ok=True)

    made, skipped = [], []
    order = sorted(kinds, key=lambda k: (0 if k == INITIAL_GRASP else 1, k))
    for tag in order:
        kind = kinds[tag]
        if kind == "edge" and not include_edge:
            skipped.append((tag, "edge (use include_edge=True)"))
            continue
        p = os.path.join(out_dir, f"pair_{tag}_{kind}.npz")
        try:
            got = export_pair(run_dir, res_mm=res_mm, anchor=tag,
                              anchor_kind=kind, out_path=p)
            if got:
                made.append((tag, kind))
            else:
                skipped.append((tag, "export refused"))
        except RuntimeError as e:
            # A rim anchor can push the fixed canvas past the data. That is a
            # real limit of that anchor, not a failure of the run, so it is
            # recorded and the rest continue.
            skipped.append((tag, str(e).split(".")[0][:80]))
    if verbose:
        n_i = sum(1 for _t, k in made if k == "interior")
        n_e = len(made) - n_i
        print(f"[anchors] {len(made)} pairs written to {out_dir}")
        print(f"[anchors]   {n_i} interior, {n_e} edge "
              f"(of {len(kinds)} grasps)")
        if skipped:
            print(f"[anchors] {len(skipped)} not written:")
            for t, why in skipped[:6]:
                print(f"[anchors]     {t}: {why}")
            if len(skipped) > 6:
                print(f"[anchors]     ... and {len(skipped)-6} more")
    return made, skipped


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
        # NOT export_pair(rd, rr): rr is the FIGURE resolution. The pair has
        # its own pinned canvas (PAIR_RES_MM / PAIR_SIZE_MM) and must not
        # inherit whatever the figures were drawn at.
        export_pair(rd)
    else:
        print(f"nothing stitched in {rd}")
