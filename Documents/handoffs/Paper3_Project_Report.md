# Paper 3 — Synthetic Tactile Data Factory
## Complete Project Report

**Researcher:** Kourosh (PhD, CoRo Lab, ÉTS Montréal)
**Supervisor:** Vincent Duchaine
**Environment:** NVIDIA Isaac Sim 5.1 · cuRobo · Ubuntu 22.04.5 · RTX 2060 (6 GB)
**Sensor:** TSF-85 tactile pad (28 taxels, 7×4 layout), two pads (s1 = right, s2 = left)
**Robot:** UR5e + Robotiq gripper

---

## 1. Project Objective (The Big Picture)

The goal of Paper 3 is to build a **synthetic tactile data factory** in simulation that
automatically collects large amounts of tactile data by grasping objects at many poses.
This data trains a model that predicts an **extended contact map** from an **initial
partial contact** — i.e., given what the sensor feels at first touch, predict what the
fuller contact would look like.

The overarching scientific goal (Vincent's headline result) is to demonstrate the
**sim-to-real benefit**: that adding synthetic tactile data improves a model trained on
real data alone.

### The four-block plan

1. **Block 1 — Data collection:** automated grasping in simulation, collecting tactile
   maps at many grid positions on each object.
2. **Block 2 — Data prep / stitching:** combine grid grasps into extended contact maps;
   pair a center-contact INPUT with an extended TARGET.
3. **Block 3 — Training:** train two U-Net models — model A (sim-only) vs model B
   (sim + real dose).
4. **Block 4 — Result:** compare A vs B to show the sim-to-real benefit.

---

## 2. Where We Are Now (Current State)

We have a **complete, working, modular pipeline** for Block 1 (data collection),
driven by a graphical cockpit. The core loop — plan a grid, collect tactile data in
simulation, read back heatmaps and verification plots — works end to end.

### What works today

- A **Tkinter GUI cockpit** (`main_gui.py`, runs in PyCharm) to plan grids, launch
  collection, and view results.
- A **config-driven Isaac collector** (`collect_from_config.py`) that grasps every grid
  point and saves clean per-point tactile data.
- **Verification plots** (desired vs actual pad pose, per-axis error), Paper-2 style.
- **Temporal snapshot extraction** (the 4-stage tactile capture from Vincent's 2021 paper).
- **Stable object handling** (fixed-joint stabilization keeps the object still while the
  deformable sensor stays alive).

### The modular architecture

```
main_gui.py  (PyCharm cockpit)
    ├── writes gui_config.json  (object pose, tilt, pad offset, grid points)
    ├── shows a copy-paste terminal command
    └── read-back buttons: heatmaps, pose history, verification, temporal snapshots

collect_from_config.py  (Isaac, via ~/isaacsim/python.sh)
    ├── reads gui_config.json
    ├── grasps every grid point (proven approach → descend → record → ascend)
    ├── writes per-point tactile CSVs (s1 + s2)
    └── writes pose_history.json (desired + actual pad poses, joints)

viz/verification.py, viz/individual_verifications.py  (desired-vs-actual plots)
viz/temporal_snapshots.py  (4-stage temporal tactile extraction)
```

---

## 3. Key Technical Facts Established

These were verified during the project and are the foundation everything rests on.

### The "two-Python wall" (hard constraint)

Isaac scripts run **only** via `~/isaacsim/python.sh` in a terminal — never in PyCharm.
Pure-Python tools (GUI, plotters) run in PyCharm. The two communicate **only through the
filesystem** (JSON config in, CSV/JSON data out) and a copy-paste terminal command.

### Isaac swallows `print()`

In script mode, Isaac's Kit logging eats `print()` output. **All diagnostics must be
written to files**, not printed. File size is the truth for whether data landed
(~92 KB = a good grasp; ~230 bytes = empty/failed).

### Scene geometry (verified by probing)

- Cylinder: 26 mm diameter × 140 mm length, standing, center at world
  **(−0.26806, 0.199, 1.0522) m**. Table top ≈ Z 0.982.
- Robot base_link world: (0.02093, −0.3375, 0.99275).
- Pad normal = **wrist-local X (column 0)**, confirmed by probe (dot 0.985 with the
  pad→cylinder direction).
- Proven grasp EE target = (−0.26806, 0.199, **1.24244**); the Z offset of
  **0.19024 m** is the tool-to-pad vertical offset (TOOL_OFFSET_Z).

### Data format

- `*_tactile_maps.csv`: columns `time_sec, frame, pred_0..pred_27`.
- ~175 rows per grasp = the **full close → hold → open curve** at 60 Hz (NOT a single
  snapshot — this is what makes offline temporal extraction possible).
- Reshape 28 values to **(7, 4)** — never transpose.
- The TSF extension **appends** all grasps in a session to the same base file; we slice
  out each grasp's new rows to separate them cleanly.

---

## 4. What We Tried (Chronological Journey)

### 4.1 Stable data collection — SOLVED

**Problem:** the object wobbled or fell during grasping; naively freezing it
(kinematic freeze) killed the deformable sensor (empty files).

**Solution:** a **fixed-joint stabilization** — anchor the object with a joint at its set
pose while keeping it dynamic. This keeps the sensor alive AND the object still. Verified
with clean 9-point grids and object-pose-anywhere tests (standing, horizontal, tilted).

### 4.2 Pad rotation (Paper-2 incremental IK port) — PARTIALLY DONE, SET ASIDE

**Goal:** rotate the pad about its own normal (peg-in-slot motion), porting Vincent's
Paper-2 `_incremental_execute` method into Isaac.

**What worked:** the **smooth motion** (no jerk) ported correctly — the weighted
incremental Jacobian solve behaves like the real robot. We switched to a **numerical
Jacobian** (finite differences on FK), which is reliable and avoids cuRobo API fights.

**What didn't:** the rotation did not spin the pad **in place**. Attempts:
- First tried pivot at L/W offset from the wrist → pad swept a wide arc (tipped sideways).
- Confirmed via probe that the axis (wrist-X) was **correct** all along.
- Diagnosed the real issue: the **pivot** was 62 mm off-center (a wrist-vs-flange
  reference artifact). Kourosh correctly identified W should be 0 (pad centered on flange).
- Switched to pivoting at the **measured pad point** directly → still not perfect.

**Decision:** rotation was **set aside** to prioritize building the collection pipeline
and GUI. It remains an open item (see §6).

### 4.3 Continuous pose-to-pose session — ATTEMPTED, MOTION ISSUE UNRESOLVED

**Goal:** do all grasps in ONE Isaac session, sliding the gripper directly from point to
point without returning home (faster, more realistic).

**What we found:**
- cuRobo's free-plan **routes through home** between points — the "home detour."
- Straight-line stitched moves still detoured.
- Joint-space interpolation gave a **tiny, correct** move (1.2°) but the arm still
  appeared to lift ~10 cm between points.
- A dangerous **end-of-run flailing** appeared when chaining trajectories in one session.

**Verification built:** a full-path logger + Paper-2-style pad-frame plot to *see*
whether the arm detours. Confirmed the between-points translation itself was
sub-millimeter accurate (desired dY 7.95 mm vs actual 7.94 mm, error 0.03 mm).

**Decision:** the **continuous session is deferred**. The current collector uses the
**proven per-point grasp** (approach → grasp → lift → next), which is reliable but lifts
between points. Optimizing this is an open item (see §6, the "jerky motion" issue).

### 4.4 The GUI cockpit — DONE (the current main deliverable)

Built in stages:

- **Stage A:** input slots (object pose, tilt angle + axis, pad offset), live preview
  (top-down + front + 3D), grid definition (nx × ny, step mm). All distances in mm.
- **Stage B:** "Save Config" writes one JSON with everything; "Save + Show Run Command"
  produces the exact terminal command to copy (reliable bridge across the two-Python wall).
- **Stage C:** read-back buttons — heatmaps (s1 + s2 per grasp), pose history,
  verification plots, temporal snapshots.

Design decisions that shaped the GUI:
- **Pad pose is relative to the object center** (Y, Z offset; X fixed = centered grasp),
  which makes objects interchangeable and data collection easy to reason about.
- **Object orientation is a tilt angle + axis** (not a fixed dropdown), so any tilt is
  possible.
- **2D grid** (independent X and Y steps) sweeps an area like Paper 1 — e.g. 2×3 = 6 poses.
- **Top-down view** shows the cylinder with **two symmetric pads** squeezing along X.
- **Headless toggle** lets you pick whether the Isaac window opens (fast bulk runs vs
  watching).
- **Fresh timestamped run folder** per collection, so old data never mixes into readback.

### 4.5 Verification plots (#6) — DONE

Matches Paper-2: for each grasp, a plot of the **pad frame** (axes + rectangle) at
**desired (black) vs actual (red)**, per-axis position bars, and an error banner (X/Y/Z
error + total, mm). Saved to an `Individual_Verifications/` folder, one plot per point.

**Key bug fixed:** the desired pad pose was initially computed from the object center,
giving an absurd ~1120 mm error (comparing object-center vs real-pad location). Fixed by
reading the **initial pad pose once at startup** and computing each desired pose as
initial-pad + grid-offset — so desired and actual are in the same frame.

### 4.6 Temporal tactile snapshots (#3) — DONE (extraction), pending longer hold

Matches Vincent's **2021 CASE paper** ("Tactile-Based Object Recognition Using a
Grasp-Centric Exploration"). During the squeeze, capture 4 static pressure maps when the
taxel **sum** reaches:

1. **5 % of max** (first light contact)
2. **50 % of max** (mid compression)
3. **95 % of max** (near full compression)
4. **3 s after squeeze start** (post-squeeze creep — rigid vs soft object)

The first three capture how contact area and pressure evolve; the fourth reveals whether
the grasp keeps deforming after motion stops. This is the paper's **most useful modality**
(97.7 % object recognition alone).

**Implementation:** pure **offline post-processing** of the tactile CSVs we already record
(each grasp already stores the full ~175-frame curve). The extractor selects the 4 frames
at the right pressure levels — no new collection needed. Works on all past and future data.

**Honest caveat:** snapshot #4 needs the gripper to **hold closed ~3 s**. The current
grasp releases quickly, so #4 currently falls after release. Snapshots #1–3 are always
valid; #4 becomes meaningful once we lengthen the hold (small collector change).

---

## 5. Important Lessons & Principles (Do Not Forget)

- **Trust Kourosh's visual observations.** When he said the robot went home between
  grasps, it did — verified via full-path logging, not summary plots.
- **Kourosh is frequently right when pushing back.** He correctly identified the pad
  normal direction, the home-detour reality, W = 0 (pad centered on flange), and the
  source of the pose-computation error.
- **Write diagnostics to files** — Isaac swallows `print()`.
- **Never let imported modules call `matplotlib.use()` globally** — it hijacks the GUI's
  TkAgg backend and breaks window display (this bug broke the heatmap/verification
  buttons once).
- **Always call `canvas.draw()` after matplotlib updates in Tkinter** — a missing draw
  call silently froze the preview once (data updated, image didn't).
- **Use `StringVar`, not `IntVar`/`DoubleVar`, for Tkinter entry fields** — the numeric
  vars throw on partial input, silently aborting refresh.
- **Build minimal verifiable pieces first; verify each before expanding.** Debugging in
  big expensive runs is what made progress feel out of control; small file-based tests
  are faster.

---

## 6. What Is Left To Be Done (Open Items)

### From Kourosh's list of 6

| # | Item | Status |
|---|------|--------|
| 1 | **Real-robot validation run** — confirm the JSON has everything needed to recreate the motion in the real world | **OPEN — highest value** |
| 2 | **Jerky motion / home detour** between points; integrate Paper-2 incremental execution; read realtime joint space | **OPEN — hard, deferred** |
| 3 | **Three/four-step temporal tactile data** | **DONE (extraction)**; needs longer hold for snapshot #4 |
| 4 | **Heatmap stitching** — combine grids into one big contact map for training (Block 2) | **OPEN** |
| 5 | **Combine real + sim tactile data** — address the data-range gap between them | **OPEN — depends on #1** |
| 6 | **Individual verification plots** (Paper-2 style) | **DONE** |

### Other known open items

- **Pad rotation (peg-in-slot):** spin the pad about its normal in place. Smooth motion
  works; pivot geometry needs finishing. Then layer back the leash + weighted solve.
- **STL shape selector:** a GUI tab to pick object shape / STL file (currently cylinder
  only). For when multiple shapes are added.
- **Lengthen grasp hold to ~3 s** so temporal snapshot #4 (post-squeeze creep) is
  meaningful.
- **Continuous session** (all grasps in one session, no home reset) — deferred for speed
  optimization once the pipeline is validated.

### Sub-details on #2 (the motion problem)

- **2-1:** the gripper returns home between points instead of sliding directly. Root cause
  is cuRobo's free-planner choosing wide paths. Candidate fix: constrain IK to a joint
  config close to current, or a properly-tuned joint-space line.
- **2-2:** Paper-2's `_incremental_execute` is only **partially** integrated — the smooth
  motion ported, but it isn't wired into the between-points collection motion yet.
- **2-3:** reading realtime joint space **works** (`robot.get_joint_positions()`), already
  used throughout.

---

## 7. Recommended Next Steps (Priority Order)

1. **Confirm #3 end-to-end:** run a fresh grid, press "Show Temporal Snapshots," verify the
   4-stage grid shows contact building across columns. (Immediate, low-risk.)
2. **Real-robot validation run (#1):** take one GUI-planned grid to the real robot and
   confirm the JSON + pose_history are sufficient to replay it. This validates the entire
   pipeline before building more on top. (High value.)
3. **Lengthen the grasp hold** so temporal snapshot #4 becomes valid. (Small change.)
4. **Block 2 — heatmap stitching (#4):** combine grid grasps into extended contact maps,
   pairing center-contact INPUT with extended TARGET. This is the bridge to training.
5. **Then Block 3 — two-model U-Net training** (sim-only vs sim+real), and Block 4 (the
   A-vs-B comparison — Vincent's headline result).
6. **Optimization pass (deferred):** fix the jerky/home motion (#2), finish pad rotation,
   move to a continuous session for speed. Do this once the science pipeline is validated,
   not before.

---

## 8. File Inventory

### Core pipeline (in `~/Paper3_Simulation/`)

- **`main_gui.py`** — Tkinter cockpit (PyCharm). Plan grids, launch, read back results.
- **`sim/collect_from_config.py`** — Isaac collector, config-driven, all grid points.
- **`sim/grasp_one_grid_v2.py`** — proven single-grasp collector (the reliable base).
- **`viz/verification.py`** — pad-frame desired-vs-actual plot for a single move.
- **`viz/individual_verifications.py`** — one Paper-2-style plot per grasp →
  `Individual_Verifications/`.
- **`viz/temporal_snapshots.py`** — the 4-stage temporal tactile extractor.
- **`sim/probe_tool_frame.py`, `sim/probe_pad_normal.py`** — frame/geometry probes
  (write results to file).

### Data outputs

- **`Data/gui_config.json`** — the current plan (object pose, tilt, pad offset, grid).
- **`Data/gui_run/run_<timestamp>/`** — per-run output: tactile CSVs, `pose_history.json`,
  `temporal_snapshots.json`, `Individual_Verifications/`, heatmap/snapshot PNGs.

---

## 9. One-Paragraph Summary (The Elevator Version)

We have built a working, modular, GUI-driven synthetic tactile data factory in Isaac Sim.
From a Tkinter cockpit you plan a grid of grasp positions on a cylinder (relative to the
object center, with adjustable tilt), save it to a JSON config, and run a proven Isaac
collector that grasps every point and saves clean per-point tactile data for both sensors.
Back in the GUI you can view heatmaps, pose history, Paper-2-style desired-vs-actual
verification plots, and the 4-stage temporal tactile snapshots (matching Vincent's 2021
method). The object stays stable during grasping via fixed-joint stabilization, and the
whole thing respects the two-Python wall through a clean filesystem bridge. What remains:
validate one run on the real robot, optimize the between-points motion (currently reliable
but lifts between points), then move to Block 2 (stitching contact maps) and Block 3
(training the two U-Net models for the sim-to-real comparison).
