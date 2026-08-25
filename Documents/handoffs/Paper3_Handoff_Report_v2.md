# PAPER 3 — PROJECT STATE & HANDOFF REPORT (v2)
**Kourosh — PhD, CoRo Lab (ÉTS Montréal), supervisors: Vincent Duchaine, Jean-Philippe Roberge**
**Date: July 7, 2026. This document is the COMPLETE state of the project across two long Claude chats. Read it fully before helping. It supersedes v1 (July 3); v1 content is folded in here.**

---

## 0. HOW TO USE THIS DOCUMENT
- **§1–§4** = project goal, hard environment constraints, module inventory (stable, mostly unchanged from v1).
- **§5** = the full arc of CHAT 2 (stitching + pose work): every fix, diagnosis, decision, and dead-end, in order.
- **§6** = current honest state of the stitching pipeline (what is trusted, what is not).
- **§7** = OPEN ISSUES ledger (the live one — read this closely).
- **§8** = the CURRENT BLOCKER and immediate next action.
- **§9** = working-style notes (how Kourosh wants to be helped — important).

---

## 1. PROJECT GOAL (Paper 3)

Build a **synthetic tactile data factory** in NVIDIA **Isaac Sim 5.1**: a UR5e + Robotiq gripper with **two TSF-85 tactile pads** grasps a cylinder at many grid positions, collecting tactile pressure maps. The data trains a model that **predicts the EXTENDED contact map from an INITIAL partial contact** (contact-completion), replacing Paper 1/2's hand-crafted shape rules with a learned model.

**4 blocks:**
1. **Simulation data collection** ("the factory") — BUILT, working end-to-end.
2. **Data prep / stitching** — combine per-grasp maps into one extended contact map (input = center grasp, target = extended map). **IN PROGRESS (this is where chat 2 lived).**
3. **Training** — two U-Nets: (A) sim-only, (B) sim + real. NOT started.
4. **Result** — A vs B = the sim-to-real benefit (Vincent's headline). NOT started.

**Motivating paper (Vincent's):** Roberge, L'Écuyer-Lapierre, Kwiatkowski, Nadeau, Duchaine, *"Tactile-Based Object Recognition Using a Grasp-Centric Exploration"*, IEEE CASE 2021. Strongest modality (97.7%) = **temporal deformation**: 4 static maps at taxel-SUM = **5% / 50% / 95% of max + one 3 s after squeeze start**. Paper 3 replicates this (implemented). **Division of labor decided this chat:** stitching works in SPACE (across grid positions) to build the TARGET; the 4 temporal snapshots are TRAINING INPUT CHANNELS (shape/stiffness cue), NOT used to build the stitched map.

---

## 2. HARDWARE / ENVIRONMENT FACTS (hard constraints — do not violate)

- **Machine:** Ubuntu 22.04.5, RTX 2060 6GB, driver 550.144.03, i7-9750H.
- **Isaac Sim 5.1**, launched ONLY via `~/isaacsim/python.sh <script>` in a terminal. NEVER from PyCharm.
- **THE TWO-PYTHON WALL (architectural rule):** Isaac scripts run only in terminal Python (`~/isaacsim/python.sh`). All pure-Python tools (GUI, plotters, stitching, diagnostics) run in PyCharm's normal Python. They communicate ONLY through the filesystem (JSON config in → CSV/JSON out) + a copy-paste terminal command. GUI auto-launch of Isaac was tried and FAILED — the copy-paste bridge is the chosen, working design.
- **Isaac Kit SWALLOWS `print()`** in script mode → all Isaac-side diagnostics must be written to FILES. File size is ground truth for whether data landed.
- **Never let an imported module call `matplotlib.use()` globally** — it hijacks the GUI's TkAgg backend. Modules that save figures use standalone `Figure` + `FigureCanvasAgg`.
- **Tkinter `IntVar`/`DoubleVar` `.get()` throws on partial input** → all GUI fields are `StringVar` with safe parsing.
- **Always call `canvas.draw()`** after matplotlib updates in Tkinter.
- **`ask_user_input`/option buttons often DON'T render for Kourosh** — always ask in plain text.

### Key paths (verbatim)
- Project root: `~/Paper3_Simulation/` (`sim/`, `viz/`, `factory/`, `train/`(empty), `objects_stl/`, `objects_usd/`, `scenes/`, `Data/`, `TSF-85/`, `curobo-stable/`)
- cuRobo editable source: `/home/kourosh/Paper3_Simulation/curobo-stable/src`
- Scene USD: `~/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd` (+ `ur5e.yml`)
- **Must `cd ~/Paper3_Simulation/TSF-85/examples` before running Isaac grasp scripts.**
- Data output: `~/Paper3_Simulation/Data/` ; GUI runs land in `Data/gui_run/run_<YYYYMMDD_HHMMSS>/`
- GUI config: `~/Paper3_Simulation/Data/gui_config.json`
- Experiment recipes (NEW this chat): `~/Paper3_Simulation/Data/experiments/*.json`

### Scene geometry (probed & verified)
- Robot prim: `/World/robot_gripper_adapter_sensor` (all prims at a DOUBLED path `.../robot_gripper_adapter_sensor/...`).
- Sensors: right = `.../TSF_85_right/TSF_85` (**s1**), left = `.../TSF_85_left/TSF_85` (**s2**).
- Cylinder: `Object_02/Cylinder`; **Diameter 26 mm, length 140 mm**, standing (long axis world Z), center world **(−0.26806, 0.199, 1.0522) m**.
- Robot base_link world: (0.02093, −0.3375, 0.99275).
- Proven grasp EE target = (−0.26806, 0.199, 1.24244) m → `TOOL_OFFSET_Z = 0.19024 m`. gripper close = **0.55 rad**.
- **Pad normal = wrist-local X.** Pad offset from `wrist_3_link` (wrist-local): n −14.2, u 62.4, v 121.4 mm.
- **TSF-85 pad face: 22 mm (4-taxel short side, across = world Y) × 37 mm (7-taxel long side, up = world Z).** Taxel pitch: **Y = 5.5 mm, Z = 5.286 mm.**

### Tactile data format (verified)
- `*_tactile_maps.csv`: `time_sec, frame, pred_0..pred_27` (28 taxels). ~175 rows/grasp @60 Hz (full close→hold→open). Failed file ≈ 230 bytes; good ≈ 92 KB.
- **Reshape to (7,4). NEVER transpose.**
- Representative map per grasp = **hold-average** (mean of frames where sum ≥ 0.5×peak).
- The TSF extension APPENDS every grasp to the same `BASENAME_s1/s2` file; clean per-grasp files come from row-slicing → `gui_ptNN_s1/s2_tactile_maps.csv`. **Note: a base append-file `gui_s1_tactile_maps.csv` (no ptNN) also exists — all tools now glob `*_pt*` to exclude it.**
- Object stabilization: fixed joint (kinematic freeze KILLS the deformable sensor → 230-byte files; never use it).

---

## 3. CURRENT WORKING PIPELINE (verified end-to-end)

GUI (plan grid) → save `gui_config.json` → copy command → terminal Isaac run → per-point tactile CSVs + `pose_history.json` in a fresh timestamped folder → back in GUI: heatmaps, pose history, verification plots, temporal snapshots, **stitching (Block 2 tab)**.

### Run command (proven):
```bash
cd ~/Paper3_Simulation/TSF-85/examples && \
GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \
GRASP_BASENAME="gui" \
GRASP_HEADLESS="0" \
~/isaacsim/python.sh ~/Paper3_Simulation/sim/collect_from_config.py \
  --config ~/Paper3_Simulation/Data/gui_config.json
```
Setting no `OBJ_ORIENT`/`OBJ_TILT_DEG` env vars → collector reads tilt from the config (see §5).

---

## 4. MODULE INVENTORY (current, with chat-2 changes marked ★)

### 4.1 `main_gui.py` → `~/Paper3_Simulation/` (PyCharm)
Tkinter cockpit, now a **Notebook with two tabs**: "Collection" and "Stitching (Block 2)".
- Inputs (all mm, StringVar): object x/y/z (defaults −268.06, 199.0, 1052.2); **tilt (deg) + tilt axis**; pad offset from object center (Y,Z only; X greyed); **★ signed anchored grid** — see below.
- **★ Signed/anchored grid (`grid_2d`):** the grid is ANCHORED at the entered pad offset (that base = pt00, the first grasp). **|n| = number of points; the SIGN of "n steps X/Y" picks the direction.** `nx=+3`→ 0,+step,+2·step; `nx=-3`→ 0,−step,−2·step. GUI convention: **n steps X → moves along world Z (up/down the cylinder); n steps Y → moves along world Y (across).** (So a vertical line = n steps X large, n steps Y = 1.) Config still stores absolute offsets, so the collector is unchanged.
- **★ Visit-path overlay** on the FRONT (Y-Z) preview: dashed polyline in execution order, green dot = pt00 start, red X = last point, index labels for grids ≤24 pts.
- **★ Save/Load Experiment** buttons: "Save Experiment As…" writes the full recipe (all GUI fields + built config) to `Data/experiments/<name>.json`; "Load Experiment…" refills every field so an experiment can be reproduced exactly.
- **★ "Plot from folder…" / "Use newest"**: all four readback buttons (Heatmaps, Pose History, Verification, Temporal) plus the Stitch tab read from — and save into — the chosen folder. Default = newest run (auto). Routes through one method `_run_dir()` with `_forced_run_dir` override.
- **★ Stitching (Block 2) tab:** pick run (newest or Browse), canvas resolution (mm/cell), "Build Stitched Maps (s1+s2)", "Export Training Pair". Calls `viz/stitching.py`.
- Readback buttons otherwise as v1.

### 4.2 `collect_from_config.py` → `~/Paper3_Simulation/sim/` (terminal Isaac)
Config-driven collector.
- Builds each grasp EE target: X=obj_x (centered), Y=obj_y+dy, Z=obj_z+dz+TOOL_OFFSET_Z.
- Fresh timestamped output folder per run.
- Per-point proven motion (free move to UP → descent → settle → RECORD ON → close(0.55) → hold → open → RECORD OFF → ascent). Row-slicing → clean per-grasp files.
- **★ FIXED this chat — pad pose recording:** the old `pad_pose_world()` read the pad prim via `UsdGeom.XformCache` on the USD stage. In script mode PhysX moves links in FABRIC only; the USD stage keeps the AUTHORED (startup) transform → every grasp recorded the SAME frozen pose. Replaced with **`pad_pose_from_joints(q)`**: joints → cuRobo FK → EE world → fixed wrist→pad offset (`PAD_OFF_WRIST = [-0.0142, 0.0624, 0.1214]` m). Used at BOTH the per-grasp record and the startup initial-pose read.
- **★ FIXED this chat — object tilt was never applied to the grasp:** the config tilt was read but the object bolting always used `orient="keep"`, and the grasp targets never rotated. Now: orientation comes from the config (`tilt_x`/`tilt_y` by axis) unless an env var overrides; tilt about Y supported; **rotation pivots about the object CENTER** (not the prim origin at the cylinder base — a naive rotation swung the center ~24 mm at 20°). Verified in math: center stays fixed, top leans −Y at +20°.
  - **⚠ IMPORTANT UNRESOLVED SUBTLETY (see §7):** the tilt is now applied to the OBJECT, but the GRID/pad targets are still built as `object_center + (dy,dz)` — an **upright** rectangle. The grid does NOT rotate with the object. This is the A-vs-B decision below and is the current open design question.
- `pose_history.json` per grasp: `tag`, `ee_world_m`, `joints_rad`, `pad_actual_pos_m`, `pad_actual_R`, `pad_desired_pos_m`, `pad_initial_pos_m`. Copies config → `gui_config_used.json`.

### 4.3 `viz/heatmaps.py` ★ (NEW this chat) → `~/Paper3_Simulation/viz/`
Per-grasp **hold-average** heatmaps, s1 | s2 side by side, saved to `<run>/Heatmaps/heatmap_<tag>.png`, one window per grasp. Standalone Figure+Agg. **★ s2 shown MIRRORED left-right** (`MIRROR_S2=True`) since the pads face each other; display only. Globs `*_pt*` to skip the base append-file. Replaced the old broken `show_heatmaps` (which had been overwritten by a temporal plotter calling a nonexistent `SNAP_LABELS` and died silently in Tkinter).

### 4.4 `viz/stitching.py` ★ (NEW this chat, heavily iterated) → `~/Paper3_Simulation/viz/`
**Block 2 core.** For each sensor: takes each grasp's hold-average map, places each taxel at its recorded pad world position, averages overlaps. Outputs to `<run>/Stitched/`: `stitched_s1/s2.png` (3 columns: extended map | coverage | GUI-frame overlay), `.npy` + `_mask.npy`, and `training_pair.npz` (input = center grasp, target = full map, + masks, + `center_temporal_s1/s2` (4,7,4) if `temporal_snapshots.json` exists). Key internals / decisions (all this chat):
- **Offset source with degeneracy fallback:** tries `pose_history.json [pad_actual_pos_m]` → `[ee_world_m]` → `[pad_desired_pos_m]` → config offsets, skipping any source where all grasps sit within 1 mm (catches the frozen-pose bug).
- **Axis re-anchor to GUI frame:** removes the constant FK offset (wrist→pad points at the pad prim origin, not the face) by comparing recorded vs commanded poses; uses the MEDIAN shift; tolerance raised to **8 mm** spread (small per-row IK/gravity droop, slope ≈0.04 mm/mm, makes the raw actual-vs-commanded gap vary ~5 mm — real robot behavior, not a bug).
- **Orientation calibration `CAL` per sensor** — CURRENT SETTING (Kourosh's logic, provably removes a redundant double-flip): `s1 = {flip_lr:False, flip_ud:False}`, `s2 = {flip_lr:True, flip_ud:False}`. Rationale: s1 faces the object directly (keep raw orientation); s2 faces opposite (mirror L-R only). The earlier "both True" came from a calibrator run on a standing cylinder, which is symmetric top-bottom/left-right so it COULD NOT distinguish the vertical flip — it was under-determined. **Orientation is still NOT independently confirmed on tilted data (see §7).**
- **Splatting = FIXED-SIZE block per taxel** (each taxel paints a `round(pitch/res)`-cell block anchored at its center). Replaced edge-rounding, which merged/split neighbors and mangled the 7×4 pattern at coarse resolution (e.g. res=5).
- **Canvas margin = 0** → frame hugs exactly the pad footprint over the sweep (grid outer bound + pad), no dead border. (Margin had drifted between 2·res, 3 mm, etc., causing framing complaints; now zero and consistent across all 3 columns.)
- **`SUBTRACT_BASELINE` toggle (default False):** when False, single-grasp stitched map == raw heatmap flipped to world-Z-up (proven identical, peak 32=32). When True, subtracts the sensor's fixed background (better for multi-grasp training, but makes stitched ≠ heatmap). This was the source of long confusion ("why isn't stitched just the raw flipped?"). Default OFF so the identity check holds; consider ON for final training data.
- **Column 3 (GUI-frame overlay):** re-paints the stitched pressure at the COMMANDED grid positions (immune to pad_actual noise/outliers), clipped to touched cells (untouched = transparent), with the tilted-cylinder outline + grid-bound box. `MIRROR_S2_IN_OVERLAY=True` mirrors s2's panels.
- **Outlier rejection:** drops any grasp whose recorded pose is >8 mm off its commanded pose (median-robust); prints which. (Caught a real pt57 with +50 mm bad Z in one run.)
- **`calibrate(run,res)` mode** (`python3 viz/stitching.py <run> <res> calibrate`): tries all 4 flips per sensor, scores by overlap-disagreement; prints a paste-ready CAL block. **Caveat: it only distinguishes flips when the object is NOT symmetric under them — useless on a standing cylinder for the vertical flip.**
- Overlap-σ printed per sensor as a seam-quality number.

### 4.5 `viz/temporal_snapshots.py` ★ (updated) → `~/Paper3_Simulation/viz/`
CASE-2021 4-snapshot extractor (5/50/95% of max sum + post-3s). `extract_snapshots`, `process_run`→`temporal_snapshots.json`, `plot_run`→ overview PNGs (rows=grasps). **★ Added `plot_per_grasp` (2×4: s1 top / s2 bottom).** **★ s2 mirrored L-R** in both plot funcs (`MIRROR_S2=True`). Globs `*_pt*`. **Limitation unchanged:** `post3s_valid=False` until the collector hold is lengthened to ~3 s (§7).

### 4.6 `viz/individual_verifications.py` → `~/Paper3_Simulation/viz/` (unchanged from v1)
Per-grasp desired-vs-actual pad plot; error banner. Now that `pad_actual` is live (§4.2 fix), the desired-vs-actual errors are finally meaningful — a fresh confirming run is still worth doing (§7, old §6.1).

### 4.7 Diagnostics written this chat (all in `~/Paper3_Simulation/`, run in PyCharm python)
- `diag_poses.py` — per-point commanded-Z vs actual-Z, droop slope.
- `diag2_outliers.py` — lists grasps whose pose is >Nmm off commanded.
- `diag4_raw_tilt.py` — hot-COLUMN vs grid-height slope (a tilt probe; **flawed — see §5**).
- `diag5_tilt_both_axes.py` — hot row AND column vs grid position + a crest-based tilt-angle estimate.
- `diag6_single_grasp_flip.py` — per-sensor flip test on a single grasp vs a crest-line model. **Flawed for thin tilted rods (crest model too crude; gave impossible 30 mm errors at 40° offset). Do not trust its verdict.**
- `diag7_show_contact.py` — **model-free**: prints the raw 7×4 hold-average + hot row/col per sensor. This is the trustworthy one — reason from the numbers, not a model.

---

## 5. CHAT-2 ARC (what happened, in order — the reasoning trail)

1. **Fixed the dead "Show Heatmaps" button.** Root cause: it had been overwritten with a temporal plotter that indexed `extract_snapshots` results with ints and used a nonexistent `SNAP_LABELS`; the KeyError escaped the Tkinter callback silently. Rebuilt as `viz/heatmaps.py`.
2. **Decided stitching approach:** global mm canvas, place each taxel by recorded pose, average overlaps. **Skip ICP** (poses are ~ground truth in sim). Temporal = training input, not for building the map. s1/s2 = separate canvases.
3. **Built `viz/stitching.py` + the Stitching tab.** First real run stitched to ONE pad-sized blob → discovered the **frozen `pad_actual` bug** (USD stage vs Fabric). Added the multi-source offset fallback; then FIXED it at the source in `collect_from_config.py` (`pad_pose_from_joints`).
4. **Confirmed the fix:** stitched map then spanned the true swept area with a correct coverage pyramid; offsets read from `pad_actual_pos_m` cleanly.
5. **Signed/anchored grid + visit-path overlay** added (Kourosh's request: start grid at initial pose, sign = direction, one-sided edge scans).
6. **Tilt saga (the long one):**
   - Tilted the cylinder 20° → stitched map showed **no tilt** (vertical blob).
   - Found the collector **never applied the config tilt** to the object (`orient="keep"`); fixed it (tilt from config, pivot about center, tilt-about-Y added).
   - Re-ran 20° and 40° tilts → **still** a vertical-ish blob.
   - `diag4` (hot-COLUMN vs height) reported slope≈0 → **wrongly** concluded "tilt not in data." **This was a flawed diagnostic** (measured only the Y axis; the contact moves mostly in ROWS/Z).
   - Kourosh looked at the RAW heatmaps directly and saw the contact **does** move grasp-to-grasp → correctly pushed back. (Pattern holds: **when Kourosh's eyes disagree with a diagnostic, the diagnostic has been wrong every time.**)
   - Root finding: **the GRID does not rotate with the object.** Grasp targets are an upright rectangle even when the cylinder tilts (`collect_from_config.py` builds `center + (dy,dz)`, no rotation). So an upright sampling pattern on a tilted rod gives mostly-vertical contact + noise, not a clean diagonal — UNLESS many grasps along the rod are stitched. This is the **A-vs-B decision** (below), still unresolved.
7. **Orientation/flip investigation:** `calibrate` said "both flips True" but that was under-determined on a symmetric standing cylinder. Kourosh reasoned (correctly) that (a) the up-down flip was applied twice and cancels, and (b) only s2 (facing opposite) should be L-R mirrored, s1 kept raw. Set `CAL` accordingly. **Still needs the clean 1×N tilted-line test to independently confirm.**
8. **Single-grasp identity proof:** established that a 1-grasp stitched map must equal the raw heatmap flipped to world-Z-up. It didn't — traced to **baseline subtraction** in stitching (heatmap doesn't subtract). Added `SUBTRACT_BASELINE` (default False) → proven identical (peak 32=32). Also fixed **coarse-resolution mangling** (fixed-block splat) and **framing/margin** (margin=0).
9. **GUI features added:** Save/Load Experiment; Plot-from-folder for all readback buttons; s2 mirroring in heatmaps + temporal + stitching overlay.
10. **THE NEW BLOCKER (end of chat 2):** Kourosh suspects the **pad pose actually reached in Isaac does not match the pad pose designed in the GUI** — i.e. the collected tactile data may be from the wrong poses. He was about to send GUI-vs-Isaac screenshots to prove the pad lands in the wrong place relative to the cylinder. **This is the thing to resolve FIRST in the new chat — before any more stitching work.** (See §8.)

---

## 6. STITCHING PIPELINE — WHAT IS TRUSTED vs NOT (honest ledger)

**Trusted (verified):**
- Pad POSITIONS are recorded correctly now (frozen-pose bug fixed; coverage pyramids match commanded grids to ~mm).
- Placement + averaging is faithful (single-grasp stitched == raw heatmap flipped, with baseline off; peak matches exactly).
- Fixed-block splat preserves the 7×4 pattern at any resolution.
- Framing hugs the pad footprint (margin 0), consistent across all 3 columns.
- Multi-grasp overlap works; coverage + overlap-σ behave sensibly. A 5-point vertical line at 20° began to show the diagonal (σ dropped 85→63).

**NOT trusted / unresolved:**
- **Whether the GUI-designed pad pose matches the Isaac-reached pad pose** (the new blocker, §8). If poses are wrong, everything downstream is suspect.
- **Sensor orientation (flips)** — current `CAL` is from Kourosh's (sound) reasoning, not an independent tilted-data confirmation. The single-grasp flip diagnostics (diag6) were model-flawed. Needs the clean 1×N tilted-line test read via `diag7`.
- **A-vs-B grid-tilt decision** (does the grid rotate with the object?) — see §7.
- **post3s temporal snapshot** — invalid until the hold is lengthened.

---

## 7. OPEN ISSUES / DECISIONS (live ledger)

### 7.1 ★ BLOCKER — GUI pad pose vs Isaac-reached pad pose (verify FIRST)
Kourosh's observation: the pad's final pose in Isaac (after robot motion) is NOT where the GUI grid placed it relative to the cylinder → data possibly collected from wrong poses. **Action:** compare, for one grasp, three numbers: GUI-intended pad pose (object_center + offset) vs `pad_desired_pos_m` vs `pad_actual_pos_m` in `pose_history.json`. This isolates the break: GUI→config, config→command (tool offset/swing), or command→reality (IK/collision/tilt). Build a single diagnostic that prints all three side-by-side in mm. **Trust Kourosh's visual report.**

### 7.2 ★ A-vs-B: does the grasp GRID rotate with the object tilt?
- **A (current):** grid stays world-upright; object tilts. Stitched map will NOT look like a tilted cylinder — it's whatever an upright scan of a tilted rod feels. Valid for training variety; never "draws the tilt."
- **B (probably what Kourosh wants):** grid rotates with the object so pads scan ALONG the tilted surface; stitched map shows the 20° lean. Fix is small and lives in `collect_from_config.py`: rotate each `(dy,dz)` by the tilt before adding to center. **DECISION PENDING — ask Kourosh A, B, or both-behind-a-flag.** (He leans toward "the stitch must resemble the object," which = B.)

### 7.3 Orientation (flip) final confirmation
Run a clean **1×N vertical line at 20° tilt** (n steps X = 5, n steps Y = 1, step 8–10), then `diag7_show_contact.py`. The hot spot should walk smoothly along the rod; that pattern confirms the flips unambiguously (a single grasp cannot). Set `CAL` from that if it disagrees with the current s1-raw/s2-mirror setting.

### 7.4 Snapshot #4 (+3 s) needs a ~3 s closed hold
Lengthen `WAIT_HOLD_SECONDS` in `collect_from_config.py`; verify `post3s_valid:true` and files grow ~+180 rows.

### 7.5 SUBTRACT_BASELINE final choice
Default False (so stitched==heatmap for verification). Decide whether the TRAINING export should use True (removes sensor background). Likely True for final data; add it to the export path deliberately.

### 7.6 Motion efficiency: home detour + occasional wild swing (DEFERRED, real)
Per-point collector lifts to approach height and free-plans between points; occasional jerky wide swings; Kourosh has seen a home-return between grasps (logging never captured it — path_length=0 bug). For ~1000 points this wastes major time. Target motion model: pad translates+rotates within its own plane ("peg sliding/spinning in a slot"). Ideas: constrain IK to nearest config, joint-space lines for short hops, integrate Paper-2 incremental executor.

### 7.7 Pad-normal rotation (SET ASIDE)
Pad doesn't yet spin about its own normal (pivot geometry unresolved). Needed later for rotated regrasps.

### 7.8 GUI shape/STL selector (LATER)
Tab to pick object STL from `objects_stl/` for multi-shape collection. Only the cylinder exists now.

---

## 8. IMMEDIATE NEXT ACTION (what the new chat should do FIRST)

**Do NOT resume stitching polish.** Start with §7.1:
1. Have Kourosh send the two snapshots (GUI grid design vs Isaac pad-on-cylinder) AND, for one grasp (e.g. pt00): the GUI-intended pad pose, `pad_desired_pos_m`, `pad_actual_pos_m` from `pose_history.json`.
2. Build one diagnostic printing GUI-intended vs desired vs actual (mm) per grasp, to localize the divergence (GUI→config, config→command, or command→reality).
3. Fix whichever stage is broken in `collect_from_config.py` (most likely candidate: the tool-offset/swing math that turns a desired PAD pose into a commanded EE pose, or the tilt-not-applied-to-grid issue in §7.2).
4. Only once poses are trustworthy, return to: §7.3 flip confirmation → §7.2 A/B decision → finish stitching → Block 2 training-pair export at scale → Block 3.

---

## 9. WORKING-STYLE NOTES (how Kourosh wants help — important)
- **One step at a time.** Build the minimal verifiable piece, confirm with him, then expand. No big multi-part systems in one shot.
- **Modular code** — small focused files called by `main_gui.py`, never one giant script.
- **He wants root causes, explained simply and SHORTLY.** Use short, clear, concrete language; bullet points; minimal jargon. He has explicitly asked for concise, easy-to-follow answers.
- **When his visual observation contradicts logs/plots, HE HAS BEEN RIGHT EVERY TIME** (home detour, pad normal, W=0, desired-pose bug, frozen pad_actual, tilt-not-applied, the flawed diag4, baseline-subtraction confusion). Never dismiss his observation; instrument until the data shows what he saw.
- **Measure, don't guess.** Several confident guesses this chat were wrong (pad-rotation theory, diag4 slope, diag6 crest model). The winning move each time was a model-free look at the raw numbers (diag7 style) or Kourosh's eyes. Prefer direct measurement over clever models.
- **Plain-text questions** (option buttons often don't render).
- Distances in **mm** in UI; meters internally in Isaac.
- Keep every run reproducible: config + pose history + copies saved in the run folder (preserved).
- Isaac swallows prints → write diagnostics to files. If a GUI plot "does nothing," check for a swallowed exception + `canvas.draw()`. 230-byte tactile file = sensor died. 2× size file = row-slicing desync.

---

## 10. FILE-DELIVERY STATUS (what Kourosh already has installed from chat 2)
Latest versions delivered this chat (Kourosh copied them in): `main_gui.py`, `sim/collect_from_config.py`, `viz/heatmaps.py`, `viz/stitching.py`, `viz/temporal_snapshots.py`, and diagnostics `diag_poses.py`, `diag2_outliers.py`, `diag4_raw_tilt.py`, `diag5_tilt_both_axes.py`, `diag6_single_grasp_flip.py`, `diag7_show_contact.py`. **Current `CAL` in stitching.py: s1 {False,False}, s2 {lr True, ud False}. `SUBTRACT_BASELINE=False`. Canvas margin=0.**

*End of handoff v2. Treat §2 + §9 as hard constraints, §6–§7 as the honest open ledger, §8 as the starting point. The immediate priority is verifying that the pad pose reached in Isaac matches the GUI design — everything downstream depends on it.*
