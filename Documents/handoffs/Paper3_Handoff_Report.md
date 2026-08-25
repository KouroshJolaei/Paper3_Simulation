# PAPER 3 — PROJECT STATE & HANDOFF REPORT
**Kourosh — PhD, CoRo Lab (ÉTS Montréal), supervisor: Vincent Duchaine**
**Date: July 3, 2026. This document is the complete state of the project, written to hand off to a new Claude chat. Read it fully before helping.**

---

## 1. PROJECT GOAL (Paper 3)

Build a **synthetic tactile data factory** in NVIDIA **Isaac Sim 5.1**: a UR5e + Robotiq gripper with **two TSF-85 tactile sensor pads** grasps objects (currently one cylinder) at many positions, collecting tactile pressure maps. The data trains a model that **predicts the EXTENDED contact map from an INITIAL partial contact** (contact-completion).

**Paper 3 plan = 4 blocks:**
1. **Simulation data collection** (the "factory") — mostly built, working end-to-end.
2. **Data prep / stitching** — combine per-grasp maps into one big extended contact map (input = center contact, target = extended map). NOT started.
3. **Training** — two U-Net models: (A) sim-only, (B) sim + a dose of real data. NOT started (train/ folder empty).
4. **Result** — A vs B comparison = the sim-to-real benefit (Vincent's headline). NOT started.

**Key motivating paper (Vincent's):** Roberge, L'Écuyer-Lapierre, Kwiatkowski, Nadeau, **Duchaine**, *"Tactile-Based Object Recognition Using a Grasp-Centric Exploration"*, IEEE CASE 2021. Their strongest modality (97.7% alone) = **temporal deformation**: during each squeeze they keep **4 static pressure maps** at the moments the taxel SUM reaches **5% / 50% / 95% of max, plus one 3 seconds after squeeze start** (post-squeeze creep, rigid vs soft). They stack both sensors into 7×8 images at the 4 instants → 3D spatiotemporal tensor → 3D CNN. **Paper 3 must replicate this temporal sampling** (already implemented, see §5.4).

---

## 2. HARDWARE / ENVIRONMENT FACTS (critical constraints)

- **Machine:** Ubuntu 22.04.5, RTX 2060 6GB, driver 550.144.03, i7-9750H.
- **Isaac Sim 5.1**, launched ONLY via `~/isaacsim/python.sh <script>` **in a terminal**. NEVER from PyCharm.
- **THE TWO-PYTHON WALL (hard architectural rule):** Isaac scripts run only in the terminal Python (`~/isaacsim/python.sh`). All pure-Python tools (GUI, plotters, extractors) run in PyCharm's normal Python. They communicate **only through the filesystem** (JSON config in → CSV/JSON data out) plus a **copy-paste terminal command**. GUI auto-launch of Isaac via subprocess was tried and FAILED (PyCharm subprocess env breaks Isaac scene loading) → the copy-paste bridge is the chosen, working design.
- **Isaac Kit SWALLOWS `print()`** in script mode. All Isaac-side diagnostics must be **written to files**. File size is the ground truth for whether data landed.
- **`ask_user_input` style option buttons often do NOT render for Kourosh** — always also ask questions in plain text.
- **Never let an imported module call `matplotlib.use()` globally** — it hijacked the GUI's TkAgg backend once and broke all window display. Modules that save figures must use standalone `Figure` + `FigureCanvasAgg`.
- **Tkinter `IntVar`/`DoubleVar` `.get()` throws on partial input** → all GUI fields are `StringVar` with safe parsing.
- **Always call `canvas.draw()` (+ `flush_events()`)** after matplotlib updates in Tkinter — an accidental deletion of this once made the preview silently freeze.

### Key paths (verbatim)
- Project root: `~/Paper3_Simulation/` (subfolders: `sim/`, `viz/`, `factory/`, `train/` (empty), `objects_stl/`, `objects_usd/`, `scenes/`, `Data/`, `TSF-85/`, `curobo-stable/`)
- cuRobo editable source: `/home/kourosh/Paper3_Simulation/curobo-stable/src` (each Isaac script does `sys.path.insert(0, ...)` at top)
- Scene USD: `~/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd` (+ `ur5e.yml`)
- **Must `cd ~/Paper3_Simulation/TSF-85/examples` before running Isaac grasp scripts** (relative scene paths).
- Data output: `~/Paper3_Simulation/Data/` ; GUI runs land in `Data/gui_run/run_<YYYYMMDD_HHMMSS>/`
- GUI config: `~/Paper3_Simulation/Data/gui_config.json`
- Paper-2 real-robot code (reference): `/home/kourosh/Pipeline_ws/ros2_ws/Python_Modules/` (`main6.py` `_incremental_execute`, `regrasp30.py`, `get_jacobian.py`, `robot_adaptor.py`, `get_plot.py` incl. `plot_frame_move` line 632, `gui_regrasp69.py` — Tkinter, 4659 lines)
- Berith's reference example: `touch_cylinder.py` — uses **cuRobo MotionGen** (same as us), single grasp only; continuous multi-grasp is beyond it.

### Scene geometry (probed & verified)
- Robot prim: `/World/robot_gripper_adapter_sensor`; **all prims live at a DOUBLED path** `/World/robot_gripper_adapter_sensor/robot_gripper_adapter_sensor/...`
- Sensors: right = `.../TSF_85_right/TSF_85` (**s1**), left = `.../TSF_85_left/TSF_85` (**s2**).
- Cylinder: `/World/robot_gripper_adapter_sensor/Object_02/Cylinder`; parent `Object_02` (Xform) is the rigid body. **Diameter 26 mm, length 140 mm**, standing (long axis world Z), center world **(−0.26806, 0.199, 1.0522) m**, table top ≈ Z 0.982.
- Robot base_link world: (0.02093, −0.3375, 0.99275).
- **Proven grasp EE target = (−0.26806, 0.199, 1.24244) m** → `TOOL_OFFSET_Z = 0.19024 m` above the object center. `APPROACH_H = 0.10 m`, gripper close = **0.55 rad**.
- **Pad normal = wrist-local X (column 0)** — dot 0.985 with pad→cylinder direction; CONFIRMED by `probe_pad_normal.py`.
- Pad offset from `wrist_3_link` in wrist-local frame (user's convention n=col0/x, u=col1/y, v=col2/z): n −14.2 mm, u 62.4 mm, v 121.4 mm (distance 137.3 mm). **User's insight (correct): the real robot's W=0 because the pad is centered on the flange; the Isaac 62 mm is an artifact of measuring from `wrist_3_link` rather than the flange face.** Paper-2 real L=171 mm from tool0.
- TSF-85 pad face size: **22 mm (4-taxel short side) × 37 mm (7-taxel long side)**.

### Tactile data format (verified)
- `*_tactile_maps.csv`: columns `time_sec, frame, pred_0..pred_27` (28 taxels). **~175 rows per grasp @60 Hz** = the FULL close→hold→open curve (sum rises → peaks ~12,000–15,000 → falls). Empty/failed file = ~230 bytes; good ≈ 92 KB.
- **Reshape to (7,4). NEVER transpose.**
- One representative map per grasp = **hold-average** (mean of frames where sum ≥ 0.5×peak).
- **The TSF extension APPENDS every grasp in a session to the same `BASENAME_s1/s2` file** (it locks base_name at startup and ignores mid-run changes). Clean per-grasp files are obtained by **row-slicing**: track row counts before each grasp and slice out only the new rows (implemented via `row_marks` dict).
- **Object stabilization: fixed joint** (dynamic body anchored by a joint at its set pose, `LocalPos0` = object world pose, quats must be `Gf.Quatd`). Kinematic freeze KILLS the deformable sensor (230-byte files) — never use it. GPU dynamics must be on for the deformable pads.

---

## 3. CURRENT WORKING PIPELINE (verified end-to-end)

**The full loop works:** GUI (plan grid) → save `gui_config.json` → copy command → terminal Isaac run (visible or headless) → per-point tactile CSVs + `pose_history.json` in a fresh timestamped folder → back in GUI: heatmaps, pose history, per-grasp verification plots, temporal 4-step snapshots.

A 1×2 and 2-point runs completed **2/2 grasps OK**, clean 175-row files, real peaks (~12k–15.5k), pose history written.

### The run command (proven, generated by the GUI):
```bash
cd ~/Paper3_Simulation/TSF-85/examples && \
GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \
GRASP_BASENAME="gui" \
GRASP_HEADLESS="0" \
~/isaacsim/python.sh ~/Paper3_Simulation/sim/collect_from_config.py \
  --config ~/Paper3_Simulation/Data/gui_config.json
```
(`GRASP_HEADLESS="1"` = no Isaac window, faster; set by a GUI checkbox.)

---

## 4. THE MODULES (current file inventory & what each does)

All current files also live in the user's project at the noted install paths.

### 4.1 `main_gui.py` → installs to `~/Paper3_Simulation/` (run in PyCharm)
Tkinter cockpit. Features:
- **Inputs (all mm, all `StringVar` + safe parsing):** object pose x/y/z (defaults −268.06, 199.0, 1052.2); object **tilt (deg) + tilt axis (X/Y/Z)** (0 = standing); **pad offset from OBJECT CENTER — only Y and Z** (X is fixed/greyed: "centered grasp", the two pads squeeze symmetrically along X); pad rotation greyed (future); **2D grid: n steps X × n steps Y + one step size (mm)** → nx·ny grasp poses (e.g. 2×3 = 6).
- **Preview (3 panels, redrawn on Update Preview / Enter; info line shows `[upd #N]` refresh counter):**
  1. **TOP-DOWN (X-Y):** cylinder circle centered, **two pads** (crimson s1 on −X, orange s2 on +X) symmetric on the rim (visual gap `GRIP_OPEN=12mm`), shifting together with the grid — pads press the OUTER face, never through the center (user's correction).
  2. **FRONT (Y-Z):** cylinder as tilt-rotated rectangle; grid of pad rectangles (real 22×37 mm) on the face. Mapping: **Y-grid → across (Y), X-grid → up/down (Z)**.
  3. **3D scene:** cylinder as 3D surface (tilt-aware), pad pairs at every grid point, robot base marker (20.93, −337.5, 992.75 mm), rotatable.
- **Headless checkbox** → `GRASP_HEADLESS` in the command.
- **Save Config** → writes `gui_config.json` (object pose+tilt+shape+dims, pad base offsets, grid params, and the full `points` list — each point = `pad_offset_y_mm`, `pad_offset_z_mm` from object center).
- **Save + Show Run Command** → popup with the exact terminal command + Copy button (auto-launch permanently abandoned).
- **Readback buttons (all read the NEWEST `gui_run/run_*` folder):**
  - *Show Heatmaps (s1+s2)* — hold-average per grasp, top row s1 / bottom s2, one column per point.
  - *Show Pose History* — text window of real EE world pose per grasp.
  - *Make Verification Plots* — generates one Paper-2-style desired-vs-actual pad plot per grasp into `Individual_Verifications/` and opens **each in its own window** (user preference).
  - *Show Temporal Snapshots (4-step)* — runs the extractor and shows the s1/s2 snapshot grids.

### 4.2 `collect_from_config.py` → installs to `~/Paper3_Simulation/sim/` (terminal Isaac)
The config-driven collector (based on the proven `grasp_one_grid_v2.py`).
- Parses `--config` JSON; converts mm→m; builds each grasp EE world target: `X = obj_x` (centered), `Y = obj_y + dy`, `Z = obj_z + dz + TOOL_OFFSET_Z(0.19024)`.
- `GRASP_HEADLESS` env controls window.
- **Fresh timestamped output folder per run**: `gui_run/run_<stamp>/` (solves old-data mixing).
- Fixed-joint object freeze (proven), cuRobo loaded, per-point proven motion: **free move to UP → stitched-Z descent → settle → RECORD ON → close(0.55) → hold → open → RECORD OFF → stitched-Z ascent**.
- **Row-slicing** per grasp → clean `gui_ptNN_s1/s2_tactile_maps.csv`.
- **Reads the INITIAL pad pose ONCE at startup** (before any grasp) — the reference for all desired poses (user's Paper-2 logic).
- **pose_history.json** per grasp: `tag`, `ee_world_m` (FK→world), `joints_rad`, `pad_actual_pos_m` + `pad_actual_R` (read from the pad prim), `pad_desired_pos_m` (= INITIAL pad pose + that point's grid offset), `pad_initial_pos_m`. Also copies the config into the run folder (`gui_config_used.json`).
- Motion note: this is the **reliable per-point way** (arm lifts to approach height between points). Continuous sliding is a deferred optimization (see §6.2).

### 4.3 `individual_verifications.py` → installs to `~/Paper3_Simulation/viz/` (PyCharm)
Per-grasp Paper-2-style verification: 3D pad frame (RGB axes) + pad rectangle (real 22×37 mm) at **DESIRED (black) vs ACTUAL (red)**; per-axis desired-vs-actual bars; error banner (X/Y/Z error mm + TOTAL). Saves `Individual_Verifications/verify_ptNN.png`. Uses standalone `Figure`+`FigureCanvasAgg` (no global backend change). The **user's verification logic (must be preserved):** initial pad pose from joints→FK→EE→pad offset; desired = initial + commanded step; actual = joints at the reached pose→FK→pad; error per axis.

### 4.4 `temporal_snapshots.py` → installs to `~/Paper3_Simulation/viz/` (PyCharm)
Vincent's CASE-2021 temporal sampling, as **pure post-processing** of the already-recorded curves:
- `extract_snapshots(csv)` → 4 maps (7×4) at first crossing of **5% / 50% / 95% of that grasp's max sum**, plus the frame at **3.0 s after squeeze start** (`post3s`), with per-snapshot frame/time/sum metadata and a `post3s_valid` flag.
- `process_run(run_dir)` → `temporal_snapshots.json` (all grasps, both sensors — the future training input).
- `plot_run(run_dir)` → `temporal_snapshots_s1.png`, `_s2.png` (rows=grasps, cols=squeeze stages).
- **Honest current limitation:** `post3s_valid = False` on existing data because the grasp releases before 3 s. Snapshot #4 becomes meaningful once the collector's hold is lengthened to ~3 s (small change, agreed for later). Snapshots #1–3 are fully valid now. Verified live on real CSV: peak 13756; p05 frame 55 (sum 1423), p50 frame 58 (7992), p95 frame 71 (13756), post3s frame 174 (277 — after release).

### 4.5 Earlier proven/experimental files (still relevant)
- `grasp_one_grid_v2.py` (sim/) — the proven single-grasp collector all others derive from. Recurring hazard: edits near the freeze block once deleted `dn = robot.dof_names` → NameError; always keep it.
- `scene_config.py`, `run_grid_from_csv.sh`, `plot_grid_heatmaps.py` — the pre-GUI grid pipeline (banked; a clean 9-point grid `grid_20260630_111624` was collected with it).
- `collect_session.py` (sim/) — the **continuous 2-grasp session experiment** (grasp → release → move → grasp in ONE Isaac session). Contains: `solve_ik` (cuRobo IK), `move_joint_line` (direct joint-space interpolation with optional pose logging), `plan_stitched_to`, `pad_pose_world()`, a full-path logger wrapping `world.step`, row-slicing. Status: collects clean separated data, **no flail after the joint-line fix**, but the **home-detour question is unresolved** (see §6.2).
- `verification.py` (viz/) — the P1→P2 move verification plot (pad frame + full path + path-length-vs-straight-line detour detector). Its full-path logging was incomplete (path_length=0 bug — logging activated after the ascent).
- `rotate_test.py` (sim/) — pad-normal rotation test (grasp → release → rotate 30° in air → regrasp → before/after heatmaps). **Jerk is SOLVED (smooth Paper-2 style motion works). The rotation geometry is NOT solved** (pad swept/tipped instead of spinning in place; pivot attempts: L/W from wrist → wrong (62mm sweep); pivot at measured pad point → still a weird move). SET ASIDE by agreement.
- `probe_tool_frame.py`, `probe_pad_normal.py` (sim/) — frame diagnostics (write results to files; do not trust prints).
- Data check snippets: hold-average heatmap plotting one-liners used throughout.

---

## 5. WHAT WAS ACCOMPLISHED (chronological highlights of this chat)

1. **Probed tool0/pad frames** → wrist_3_link is tool0; pad offsets measured; **pad normal = wrist X confirmed**; user's W=0 insight adopted.
2. **Ported Paper-2 `_incremental_execute`** (numerical FK Jacobian, weighted damped least-squares, leash) → **smoothness/jerk problem SOLVED**; **rotation-in-place NOT achieved** (two pivot strategies failed) → rotation **set aside**.
3. **Continuous 2-grasp session built** (`collect_session.py`): joint-space line for between-points move (no more flailing; clean separated data; P1→P2 verified sub-0.1mm accurate at the endpoints) — but the user **repeatedly observed a home detour between points**, which logging never fully captured (path_length=0 logging bug). Treat the user's observation as TRUE; do not dismiss it.
4. **Pivot to GUI-first plan (user's call, correct):** regain control with a visual cockpit. Three stages agreed and DONE:
   - **Stage A:** Tkinter GUI + live preview (top-down + front, later 3D), mm units, 2D grid (nx×ny), pad offset **relative to object center** (Y,Z only; X fixed), tilt angle+axis replacing the orientation dropdown, two-pad symmetric top-down (user's corrections).
   - **Stage B:** Save Config JSON + copy-paste run command (auto-launch tried & abandoned).
   - **Stage C:** readback — heatmaps, pose history, verification plots, temporal snapshots; timestamped run folders; newest-run auto-pick.
5. **`collect_from_config.py` built & validated:** 2/2 grasps clean end-to-end from the GUI config.
6. **Verification (#6) built & the big bug fixed:** desired pad pose was first computed from the OBJECT CENTER → absurd ~1120 mm errors; fixed to **desired = INITIAL pad pose (read once at startup) + grid offset** per the user's Paper-2 logic. *A fresh confirming run showing small (mm-scale) errors is still pending.*
7. **Temporal snapshots (#3) built** matching the CASE-2021 paper exactly; validated on real data; GUI button added; snapshot #4 awaits a 3 s hold.
8. **Many GUI/plot bugs fixed:** IntVar crash → StringVar; missing `canvas.draw()`; Agg-backend hijack; combined→separate verification windows; old-data mixing → timestamped folders; headless toggle added.

---

## 6. OPEN ISSUES / KNOWN BUGS (do not lose these)

### 6.1 Fresh-run confirmation of #6 (IMMEDIATE, small)
After the desired-pose fix, run a fresh 2×2 grid and confirm the verification banners show **small mm-scale errors** (not 1120 mm). The pipeline is expected correct; this is the confirmation step.

### 6.2 Motion efficiency: home detour + occasional wild move (DEFERRED but real)
- With the per-point proven collector, the arm **lifts to approach height and free-plans between points**; occasionally cuRobo produces a **crazy/jerky wide swing** between points (seen in a 2×2 run). Harmless in sim, but for ~1000 points it wastes major time.
- In the continuous-session experiments, the user **saw the arm return to home between grasps** even when endpoint numbers were near-perfect; my full-path logging failed to capture it (path_length=0 — the logger activated after the ascent; Isaac also swallows prints). **The user's visual reports were right every time this session — trust them.**
- Berith's example (cuRobo) does only single grasps; continuous multi-grasp sliding is new ground.
- Ideas on the table: constrain IK to nearest-config (reject far branches), joint-space lines for all short hops, log the FULL path (including ascent) to a file, or integrate the ported Paper-2 incremental executor for the between-point slides.
- **The target motion model (user's explicit spec):** the pad must be able to **translate AND rotate freely within its own plane** (planar X, planar Y, rotation about the pad normal) — "peg sliding/spinning in a rectangular slot" — without tilting out of plane or disturbing the object, then regrasp.

### 6.3 Pad-normal rotation (SET ASIDE)
Smooth motion works; the pad does not yet spin in place about its own normal (pivot geometry unresolved). Needed eventually for rotated regrasps (Paper-1/2 style) and richer data.

### 6.4 Snapshot #4 (+3 s) needs a ~3 s closed hold
Small collector change: lengthen `WAIT_HOLD_SECONDS` (or add a dedicated hold) so the 4th snapshot is meaningful. Agreed to do later.

### 6.5 GUI shape/STL selector (LATER)
A tab/window to pick the object STL from `objects_stl/` for multi-shape data collection. Currently only the cylinder exists in the scene.

### 6.6 Small environment quirks
- Isaac warnings about `Object_01`/`Object_03` inertia and the `robot_gripper_joint` disjointed transform are known noise; ignore.
- CPU governor is `powersave` (warning at startup) — could be set to performance for speed.

---

## 7. THE USER'S MASTER TASK LIST (status)

| # | Task | Status |
|---|------|--------|
| 1 | **Real-robot validation run** — replay one grid on the real UR5e from the saved JSON (config + pose_history now include EE world pose + joint angles + pad poses per grasp) to confirm nothing is missing for real-world recreation | **PENDING — next major milestone; highest value** |
| 2 | Fix jerky motion / home-detour between points (2-1 slide not home; 2-2 fully integrate Paper-2 incremental execute; 2-3 realtime joint reading — **already available:** `robot.get_joint_positions()`) | **DEFERRED, documented (§6.2)** |
| 3 | Temporal tactile capture (Vincent's 4-snapshot method) | **DONE** (extractor + GUI button; #4 needs 3 s hold) |
| 4 | Heatmap **stitching** → one big extended contact map for training | **PENDING (Block 2)** |
| 5 | **Sim-vs-real data range gap** (normalization / domain alignment) | PENDING (needs #1's real data) |
| 6 | Individual verification plots (Paper-2 style, desired vs actual pad, per-axis error) | **DONE** (separate window per grasp; fresh-run confirmation pending §6.1) |

**Agreed sensible order from here:** confirm #6 with a fresh run → lengthen hold (enables snapshot #4) → **#1 real-robot validation** → #4 stitching → #2 motion optimization → #5 domain gap → Block 3 training.

---

## 8. HOW TO OPERATE THE PIPELINE (quick reference for the new chat)

1. **Plan:** PyCharm → `python3 ~/Paper3_Simulation/main_gui.py` → set object pose/tilt, pad Y/Z offset, grid nx×ny + step → *Update Preview* (watch `[upd #N]` tick).
2. **Save & run:** *Save + Show Run Command* → copy → paste in terminal (headless checkbox controls the window) → watch (or not) Isaac execute every grid point.
3. **Readback:** *Show Heatmaps* / *Show Pose History* / *Make Verification Plots* / *Show Temporal Snapshots* — all read the newest `Data/gui_run/run_*/`.
4. **Manual snapshot extraction (optional):** `python3 ~/Paper3_Simulation/viz/temporal_snapshots.py <run_dir>`.

**Debug rules of thumb:** if an Isaac script "prints nothing", it didn't fail — Isaac ate the prints; check output FILES. If a GUI plot "does nothing", check for a swallowed exception in `_read`/refresh and confirm `canvas.draw()` runs. If tactile files are ~230 bytes, the sensor died (object was kinematically frozen, or GPU dynamics off). If one grasp's file is ~2× normal size, the row-slicing marks got out of sync.

---

## 9. WORKING-STYLE NOTES FOR THE NEW CHAT (important)

- Kourosh works **one step at a time**: build the minimal verifiable piece, confirm it with him, then expand. Do not build big multi-part systems in one shot.
- He wants **modular code** — small focused files called by `main_gui.py`/`main.py`, never one giant script.
- He wants to **understand root causes**, not just receive patches. Explain simply and concretely; keep it short and clear.
- **When his visual observation contradicts the logs/plots, HE HAS BEEN RIGHT every single time** (home detour, pad normal, W=0, desired-pose bug, preview-not-updating). Never dismiss his observation as an illusion; instrument until the data shows what he saw.
- Plain-text questions (option buttons frequently fail to render for him).
- Distances in **mm** in all UI; meters internally in Isaac.
- Keep every new run reproducible: config + pose history + copies saved inside the run folder (already implemented — preserve this).

---

## 10. IMMEDIATE NEXT ACTIONS (what the new chat should start with)

1. **Confirm #6:** fresh 2×2 GUI run → *Make Verification Plots* → verify per-axis errors are now mm-scale (the desired-pose fix). If not, debug from `pose_history.json` (compare `pad_desired_pos_m` vs `pad_actual_pos_m` vs `pad_initial_pos_m`).
2. **Lengthen the grasp hold to ~3 s** in `collect_from_config.py` (so temporal snapshot #4 becomes valid) — verify `post3s_valid: true` afterwards and that files grow accordingly (~+180 rows).
3. **Prepare the real-robot validation (#1):** map `pose_history.json` (joints + EE world + pad poses) onto the real UR5e replay; identify anything missing (gripper close command mapping, timing, safety approach) before lab time.
4. Then Block 2 (**stitching**): design how per-grasp hold-average maps (+ temporal stacks) at known pad offsets combine into one extended contact map; define the (input = center contact, target = extended map) training pairs.

---

*End of handoff. The new chat should treat everything in §2 (environment facts) and §9 (working style) as hard constraints, §6 as the honest open-issues ledger, and §10 as the starting point.*
