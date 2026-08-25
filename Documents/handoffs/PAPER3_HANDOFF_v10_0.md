# PAPER 3 — HANDOFF v10.0
**Date:** 2026-08-20
**Author:** Kourosh Jolaei, PhD candidate, CoRo Lab, ÉTS Montréal
**Supervisors:** Vincent Duchaine, Jean-Philippe Roberge
**Collaborator:** Berith Atemoztli De la Cruz Sánchez (Isaac Sim TSF-85 extension + CNN)

> Paste this whole file at the start of a new chat. It is written to be
> self-contained: everything below is either a measured number, a decision
> already taken, or a named open question.

---

## 0. HOW TO WORK WITH ME (read first)

- I send short, typo-heavy messages. Answer in **one sentence per question**
  unless I ask for more.
- Give me **complete modified files**, never diffs.
- Distances in **mm**.
- **One verifiable step at a time.** Do not batch three changes into one file
  drop.
- **Measure, do not guess.** If you don't know a number, ask me to run a
  command and paste the output.
- If my visual read of the simulation contradicts a diagnostic number, **I
  have been right every time so far** — treat it as a signal to add
  instrumentation, not to dismiss.
- Do not bend the pipeline's conventions to accommodate old or dirty data.
  I would rather discard data than loosen a rule.
- The whole project has one recurring principle: **bad data must refuse to
  appear, loudly, rather than appear quietly.** Every guard below exists
  because a silent failure already cost me a day.

---

## 1. THE BIG PICTURE

### Papers 1 and 2 (done)
- **Paper 1** (J. Robotics and Mechatronics, Vol.38 No.3, 2026) — *Shape-Based
  Contact Extrapolation*. A NN classifies the object as cuboid / sphere /
  cylinder from the initial tactile imprint, then shape-specific hand-written
  rules extrapolate the contact beyond the pad. Metrics: Tactile Centroid (TC)
  error, Grasp Success Rate (GSR), SSIM. **Safe-zone thresholds: TC 4.42 mm,
  GSR 33.52%** (70th percentile of the error distributions over 650 test
  points).
- **Paper 2** (submitted, Robotics and Autonomous Systems) — *Tactile-Guided
  ReGrasping*. Uses Paper 1's extrapolated manifold to run a **virtual search**
  for a better regrasp before moving. Centroidal Depth Translation (CDT) won:
  90.7% of 602 trials achieved ΔGSR ≥ 10%. Established the **30 mm haptic safe
  zone** — the distance beyond which extrapolated contact stops tracking
  reality.

### Paper 3 (this work) — the claim
Replace Paper 1's **hand-crafted, classifier-dependent** extrapolation with a
**learned contact-completion model**:

> **Input:** one 7×4 tactile imprint (pt00).
> **Output:** the extended contact map, 96×96 mm at 1 mm/cell, in the pad's own frame.
> **No shape classifier. No object pose. No vision.**

The model must be **pose-blind**: the contact's angle is already visible in the
imprint, so feeding pose separately would be telling it what it can already
see. At deployment the output is composed with pt00's measured pad pose (from
FK) to get a world location for the regrasp.

### Block structure
| Block | What | State |
|---|---|---|
| 1 | Isaac Sim data collection (grid sweeps, tactile capture) | **DONE** |
| 2 | Stitching → 96×96 training pairs | **DONE** |
| 3 | Train the contact-completion model | **RUNNING — needs more data** |
| 4 | Compare against Paper 2's method; virtual search; GSR | **NOT STARTED** |

**Block 4 is the actual scientific claim.** Paper 3 asserts the learned model
beats the shape-classifier approach, and that is a *comparison*, not a
measurement. Paper 2's extrapolation must be run on the **same 96×96 pairs with
the same metrics**, or the learned model has nothing to be better than. Not
started; flagged repeatedly.

---

## 2. ENVIRONMENT

### Machine
Single laptop, dual-boot Linux (Ubuntu 22.04) / Windows. RTX 2060, 6 GB.
External **"My Passport"** drive for run data. Disk space has caused real
failures — see §9.

### Key paths
```
~/Paper3_Simulation/
  main_gui.py                  the cockpit GUI (all sim control)
  sim/collect_from_config.py   Isaac Sim collector
  viz/stitching.py             canvas, stitching, pair export
  viz/heatmaps.py              per-grasp heatmaps
  viz/blob_axis.py             contact-axis measurement
  viz/validation.py            stitch round-trip validation
  viz/grid_accuracy.py         designed grid vs actual movement
  train/dataset.py             pair discovery + validation
  train/model.py               U-Net + metrics
  train/train.py               training loop
  train/venv/                  CPU PyTorch venv (system python has no torch)
  Real_Robot/                  ALL real-robot code lives here
  TSF-85/examples/scenes/scene_cylinder.usd
  Data/gui_run/SIM/            sim runs (default for new sessions)
  Data/gui_run/Real/           real runs
  Data/pad_offset_calibration.json
  Data/gui_settings.json       remembers the "Data folder..." choice
```

### Sim launch
Everything is driven from `main_gui.py`. The GUI writes
`Data/gui_config.json`, mints a session folder, and shells out to
`collect_from_config.py` with `GRASP_RUN_DIR` set.

### Training
```
cd ~/Paper3_Simulation/train && source venv/bin/activate
python3 train.py --scan-only                 # inventory
python3 train.py --epochs 60 --anchors interior
```
`--anchors none|interior|all`, `--roots <dir>`, `--rig sim|real|both`,
`--epochs`, `--cpu`, `--seed`, `--val-frac`.

---

## 3. HARDWARE FACTS (measured, not assumed)

| Thing | Value |
|---|---|
| Robot | UR5e, 192.168.1.101 |
| Gripper | Robotiq 2F-85, 85 mm stroke |
| Sensors | 2 × CoRo capacitive, **7×4 taxels**, 60 Hz native |
| Pad active area | **22 mm wide × 37 mm tall** |
| Taxel size | **5.5 mm × 5.29 mm** |
| `TOOL_OFFSET_Z` (sim, Ø26) | 0.15657 m flange→pad face |
| `TOOL_OFFSET_Z` (sim, Ø10) | 0.15722 m |
| Palm above pad | 69.88 mm (`TOOL_OFFSET_Z − 86.69`) |
| Object centre (sim world) | `[-268.06, 199.0, 1052.2]` mm |
| Indentation at closed grip | ~2.4 mm |

### Frame transform — VERIFIED to 0.045 mm
```
world_mm = base_link_mm + [20.930, -337.500, 992.750]      (no rotation)
```
Established by putting Isaac at the real arm's exact joint angles and comparing
`tool0`. Independently confirmed: `w2b(object)` → `[-0.289, 0.5365, 0.0595]` m,
matching the collector's own logged obstacle centre.

**Frame gotcha:** ROS `base_link` and the UR controller's `base` differ by
**180° about Z**. Comparing them without that flip gives a spurious ~956 mm
disagreement.

**Joint order gotcha:** this cell publishes `/joint_states` as
`shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan` — **pan LAST** —
and a 7th gripper joint when the bridge runs. **Always match by name.** Taking
the list positionally gives five of six joints wrong and a plausible-looking
pose.

**QoS gotcha:** two publishers on `/joint_states` with different reliability. A
RELIABLE subscriber silently receives nothing from the BEST_EFFORT one. Use
BEST_EFFORT.

---

## 4. PINNED CONVENTIONS (do not change without re-exporting everything)

### The training pair
```
canvas      96 × 96 cells
cell        1.0 mm
frame       the PAD's own frame (axes along the pad edges)
centre      the designed initial grasp, pt00
```
- **1 mm** because a taxel is 5.5 mm — 1 mm lets a rolled pad's edge be traced
  rather than stair-stepped. Finer buys nothing real.
- **96 mm** = pad (22×37) + ~30 mm margin all round = **Paper 2's measured safe
  zone**. The canvas extent *is* the model's prediction extent, so this is a
  claim, not a container.
- **Pad frame** because the model's output must arrive in the frame the regrasp
  is commanded in (Paper 2's CDT maps Δx, Δy onto `tool0.Y`/`tool0.Z`). A
  world-aligned canvas would return the answer in a frame depending on the
  wrist's roll. Contract: **in = pad frame, out = pad frame**; composition with
  FK happens outside the model.
- A run needing more than 96 mm **raises**, never crops.

Env overrides: `STITCH_PAIR_RES_MM`, `STITCH_PAIR_SIZE_MM`,
`STITCH_PAIR_PAD_FRAME`.

### The file contract — this is why real and sim share everything downstream
Every consumer (stitcher, pair export, blob axis, heatmaps, grid accuracy)
reads exactly three things from a run folder:
```
pose_history.json                     ee_world_m + pad_actual_R per grasp
<base>_ptNN_<s>_tactile_maps.csv      the 7×4 stream, per grasp per sensor
gui_config_used.json                  the grid that was asked for
```
`collect_from_config.py` (Isaac) and `collect_real.py` (UR5e) both produce
these. **Nothing downstream knows or cares which rig it came from.**

The stitcher prefers `pose_history.json [pad centre from EE+FK]`, computing
`pad = ee_world + R[:,2] · TOOL_OFFSET_Z`. It reads `TOOL_OFFSET_Z` from
`reachability_report.json`.

### Masked loss — DECIDED
About 60% of the canvas is never visited by any pad. Those cells mean **"nobody
looked"**, not "no contact", and are **excluded from every training and
evaluation number**. `target_mask_<s>` in the npz marks visited cells.

The model still *outputs* a full 96×96 at inference — that unmasked region is
the extrapolation Paper 3 is about. It simply can't be scored.

### Split by run, never by pair
Several pairs can come from one sweep. Splitting by pair would put near-copies
on both sides and measure memorisation.

---

## 5. WHAT IS BUILT AND PROVEN

### 5.1 Simulation pipeline (Block 1–2) — DONE
- **Grid designer** (`main_gui.py: design_grid`, button *"Design grid from
  object geometry"*). Computes `nx`, `ny`, `pad_dz` from diameter, length,
  calibration and the 96 mm canvas. **Step comes from the box** (use **6 mm**);
  the designer computes the rest. Prints its reasoning line by line and
  **refuses** rather than shrinking.
  - Four limits: across = contact band `√(indent·(D−indent))`; along = rod end,
    palm clearance ≥5 mm, and canvas.
  - The across-band formula gives **15.1 mm at Ø26**, independently matching the
    **15.0 mm** fitted from a real upright run.
- **Pad-axis stepping** (checkbox, default on). Rolled pads sweep a clean tilted
  rectangle instead of a sheared quilt. Verified: pad-frame extent
  54.00 × 69.00 mm exactly, roll 0 bit-identical.
- **Initial-grasp guard.** `INITIAL_GRASP = "pt00"`. If pt00 is missing, the
  figure is titled in red and **no training pair is written**
  (`STITCH_ALLOW_FALLBACK=1` overrides, recording the substitution).
- **Retry + abort (solution D).** pt00 failing → one retry from a fresh plan →
  abort the whole run with the reason. Every attempt recorded in
  `execution_ledger.json`; a retried run is flagged `initial_retry_required`.
  Proven on hardware both ways.
- **Grid accuracy** button: designed vs actual pad position, reporting **bias**
  and **scatter** separately (bias shifts the whole map; scatter blurs it).
  Verified by injecting a known offset and recovering it to 0.002 mm.
- **Data folder button.** New sessions can be written straight to the external
  drive; remembered across restarts; checks writability when you pick, not
  mid-run; falls back with a message if the drive is unplugged.

### 5.2 Real robot (Block 1, real side) — PROVEN, then parked
All three layers work on hardware:
- **Motion:** 50 mm commanded, **0.010 mm error**; sideways drift <0.01 mm.
- **Gripper:** open/close via pendant digital outs.
- **Tactile:** live 7×4 from both sensors over TCP.
- **`collect_real.py`:** full grasp cycle writing the three contract files. Its
  output went through the **unmodified** `stitching.py` and produced
  `(96,96) pad frame, pt00 designed`.

**Non-negotiable operating facts:**
1. The pendant program **`external_control4` must be LOADED AND RUNNING**
   (Local Control, press Play). It is an *External Control* program that also
   contains the gripper thread. Without it the controller **accepts goals that
   silently never reach the arm** — this cost hours, three separate times.
2. **Ctrl-C does not stop the arm** by itself; the controller buffers an
   accepted trajectory and resumes after an e-stop is cleared.
   `collect_real.py` now calls `cancel_goal_async()` first.
3. The flange target must use the **measured** tool axis, not an assumed
   vertical one. The real tool z is `[+0.005, −0.005, −1.000]` (~0.4° off); over
   the 156 mm lever that displaces the pad **1.1 mm**. Fixed; round trip now
   closes to 0.000000 mm.

**Launch sequence (Paper 3 version — Paper 2's manual omits MoveIt):**
```
0. PENDANT: load external_control4, Local Control, press Play, confirm running
1. ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e \
     robot_ip:=192.168.1.101 use_fake_hardware:=false launch_rviz:=false \
     use_sim_time:=false description_package:=ur_description_gripper \
     description_file:=ur.urdf.xacro
2. ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e \
     launch_rviz:=true use_sim_time:=false \
     description_package:=ur_description_gripper description_file:=ur.urdf.xacro
3. python3 ~/ur5e_ws_Gripper/bridges/gripper_moveit_bridge.py
4. ros2 service call /controller_manager/switch_controller \
     controller_manager_msgs/srv/SwitchController \
     "{activate_controllers: ['scaled_joint_trajectory_controller'], \
       deactivate_controllers: [], strictness: 2}"
5. VERIFY: ros2 control list_controllers -c /controller_manager | grep scaled   -> active
           ros2 service list | grep compute_                                     -> fk and ik
```
Tactile also needs the **Qt app** running (it is the server; Python is a TCP
client to `127.0.0.1:12345`).

**Real-robot scripts in `Real_Robot/`:** `read_pose.py`, `sim_pose_check.py`,
`move_test.py`, `real_motion.py`, `collect_real.py`, plus copies of
`get_fk2.py`, `get_trajectory.py`, `execute_trajectory.py`,
`define_ur5e_robot.py`, `gripper_io2.py`, `tactile_DataReadSave3.py`.

### 5.3 Training (Block 3) — WORKING
`train/{dataset,model,train}.py`. Proven end to end.
- **Scans every run folder** and buckets it: has a pair / has grasps but no
  pair (with the reason: aborted, pt00 missing, never stitched) / no grasps.
  Nothing is silently omitted.
- **Validates** each pair: 96×96, pad frame, 1.0 mm, `initial_status=designed`,
  target must cover more than input, real runs need real calibration.
- **Baselines every run** against *zero* and *copy-the-input*. **Copy is the bar
  that matters** — most of a target *is* the initial imprint, so a model that
  merely echoes already scores well.
- Metrics: masked L1, masked SSIM, TC error (mm).
- Outputs named by mode: `20260820_155437_interior/report_interior.txt`,
  `preds_interior.png`, `curves_interior.png`, `best_interior.pt`,
  `history_interior.csv`.

### 5.4 Multi-anchor pairs — BUILT AND MEASURED
One sweep yields many pairs by re-centring the canvas on different grasps, at
**zero robot cost**. `stitching.export_anchor_pairs()`, GUI checkbox (default
on) under *Export Training Pair*.
- `training_pair.npz` is **untouched** and stays the pt00 pair; extras go to
  `Stitched/pairs/pair_ptNN_{interior,edge}.npz`.
- **Interior** = other grasps exist in all four directions (a 5×5 grid → the
  middle 3×3 = 9). **Edge** = rim; its target is lopsided because half the
  canvas hangs over unswept ground. Verified: 9/25, 25/49, 0 for a 1-D sweep.
- These are **augmentation, not new objects**. `split_by_run` keeps a run's
  anchors together.
- **Validation is always pt00-only**, whatever `--anchors` is set to.

---

## 6. THE OPEN SCIENTIFIC PROBLEM (blocks half the data)

### 6.1 Diagonal contacts render as blobs
Berith's Isaac CNN reproduces contacts correctly at **0° and 90°** pad roll and
**fails at every angle between**. Measured on one Ø10 rod, same day, same
settings:

| pad roll | axis error | measured elongation | expected | ratio |
|---|---|---|---|---|
| 0° | **−1.18°** | 3.47 | 3.38 | **1.03** |
| 25° | **+18.35°** | 1.74 | 4.40 | **0.40** |
| 90° | **+0.03°** | 2.86 | 2.90 | **0.99** |

Reproduced on Ø26 as well. **The failure gets worse as the true ridge gets
sharper** — at 25° the expected elongation was the highest ever recorded (4.40)
and the sensor did worst.

**Ruled out by measurement:** pad placement (orientation error 0.006°, X-symmetry
0.18 mm, pads level to 0.013 mm), object motion (0.000 mm), pad overhanging the
rod, the angle estimator itself (unbiased on a synthetic ridge), grip force,
and grasp height.

**Jean-Philippe says the CNN's training set included all kinds of data**, so the
next suspect is the **33×57 deformation map → 7×4 taxel mapping**: if anything
there treats rows and columns separately, a diagonal loses signal in both
directions while 0° and 90° survive. **This is the question for Berith.**

### 6.2 Fixed per-taxel pattern ("row 2 dip")
Row 2 reads **57%–62%** of its neighbours' average, on both sensors, at every
grasp height, on multiple diameters. A rod pressing along the pad cannot dip in
the middle of its own contact line. Not yet confirmed as a defect — the clean
test is a **flat-plate grasp**, where every taxel should read nearly the same
and whatever pattern comes back is the renderer's fixed bias, measured
per-taxel.

### 6.3 §7b consequence
**Rolled-pad data should not train the model** until this is resolved: the
targets themselves misrepresent diagonal ridges, and a model reproducing them
faithfully would be learning the renderer's failure.

### 6.4 Berith email — three questions, still unsent/unanswered
1. Is the 33×57 → 7×4 step separable? Is anything reducing rows and columns
   independently?
2. Why does the Qt tactile server deliver only **~2.5 Hz** when the sensor runs
   at 60 Hz — is the rate settable?
3. Has he run Isaac Sim on Calcul Québec (Apptainer container)?

---

## 7. TRAINING RESULTS SO FAR

### Run A — 6 rolled-pad runs (90/45/75/−25/45°), 60 epochs
| mode | L1 | SSIM | TC mm |
|---|---|---|---|
| copy baseline | 0.209–0.214 | 0.461 | — |
| pt00 only | **0.1662** | **0.717** | 5.96 |
| interior | 0.1701 | 0.714 | 3.58 |
| interior+edge | 0.1839 | 0.708 | **3.00** |

**Finding:** anchors barely move L1/SSIM but **halve TC** (5.96 → 3.00 mm),
crossing Paper 1's **4.42 mm** threshold. Coherent: anchoring teaches that the
answer's *position* depends on the input's position — which is what TC measures
and L1 does not reward.

### Run B — 12 runs (6 new upright Ø13–60 + 6 old rolled), `--anchors interior`
```
320 pairs from 12 runs; 236 train / 6 val
copy input      L1 0.1622   SSIM 0.513
model           L1 0.0832   SSIM 0.837   TC 6.48 mm
VERDICT: BEATS copy
```
**L1 improved hugely** (20% → 49% better than copy) **but TC got worse**
(3.58 → 6.48 mm).

**Almost certainly an artefact of a mixed validation set:** val was
`pad75` (rolled) + two upright runs. In `preds_interior.png` the upright rows
show a clean vertical band closely matching the target, while the `pad75` rows
show a diagonal smear. One average hides two behaviours. Also TC bounced 5.5–7.2
across epochs while loss fell monotonically, and **`best.pt` is chosen by loss,
not TC**.

**Not yet done:** `--anchors all` on this data, and an **upright-only** train
*and* validate — which is what §7b clears anyway and would settle it.

---

## 8. DATA COLLECTION PLAN — CYLINDER

### What you choose per run: **diameter** and **pad roll**. That is all.
Step, `nx`, `ny` and `pad_dz` come from the grid designer. **Use step = 6 mm.**

### Points per run (designer output, L = 140 mm, step 6 mm)
| Ø mm | 0° | 25° | 45° | 75° | 90° | row |
|---|---|---|---|---|---|---|
| 13 | 35 | 49 | 49 | 63 | 63 | 259 |
| 20 | 35 | 49 | 63 | 81 | 63 | 291 |
| 26 | 35 | 49 | 63 | 81 | 63 | 291 |
| 32 | 35 | 49 | 63 | 81 | 63 | 291 |
| 42 | 35 | 63 | 63 | 81 | 81 | 323 |
| 60 | 49 | 63 | 63 | 81 | 81 | 337 |
| **total** | **224** | **322** | **364** | **468** | **414** | **1792** |

> **Correction to an earlier table:** a table circulated during the session
> (45/63/63/77/91 …) computed `ny` from the **canvas only** and omitted the
> palm-clearance window. The numbers above are the real designer's output and
> supersede it.

### Cost, at ~2 min/point
| set | runs | points | time |
|---|---|---|---|
| upright only (0°) | 6 | 224 | **7.5 h** |
| 25° | 6 | 322 | 10.7 h |
| 45° | 6 | 364 | 12.1 h |
| 75° | 6 | 468 | 15.6 h |
| 90° | 6 | 414 | 13.8 h |
| **all 30** | 30 | 1792 | **60 h** |

`pad_dz` chosen by the designer: 0°→+28.31, 25°→+26.85, 45°→+27.13,
75°→+29.85, 90°→+32.06 mm.

### Notes
- **Length does not matter.** The pad-height window is **46.4 mm wide
  regardless of rod length** — set by palm clearance, not the rod. A longer rod
  just moves the window up.
- **Grasp height is nearly useless for a cylinder.** One sweep at `ny=3` already
  uses 36 mm of that 46.4 mm window, and a cylinder is identical along its
  length anyway. It becomes valuable for shapes with features.
- **Ø13–Ø42 all give the same upright grid** (nx=2, ny=3, 35 pts): the pad's own
  22 mm width dominates the contact band until Ø60.
- **Calibrate before batching.** The designer and collector both refuse a
  diameter with no `pad_offset_calibration.json` entry.
- **Ø60 vs the 2F-85's 85 mm stroke** — worth one manual grasp before batching.
- The sweep is sized to the **contact band**, not the rod's silhouette. On Ø60
  only ~23.5 mm of the 60 mm width can touch the pad. This is why the Ø60
  preview looks "contained" inside the rod while Ø13 spills past it. Not a bug.
  `across_margin_mm` (currently 3) trades reach for coverage if wanted.

### Status
**Six upright runs (Ø13, 20, 26, 32, 42, 60) COLLECTED**, with pt00 + interior
+ edge pairs exported, on the external drive at:
```
/media/kourosh/My Passport/KOUROSH/Project_ETS/20_Linux_19_August_2026_Experiments/
```
(the 6 older rolled runs are in the `Sim/` subfolder there).

**IN PROGRESS / FAILING:** Ø13 at pad roll 25° — failed twice, apparently
including the pt00 retry. **The terminal output was never captured.** To
diagnose, capture the `[pt00 …]` lines and:
```
python3 -c "
import json; d=json.load(open('<run_dir>/execution_ledger.json'))
print('aborted', d.get('aborted'), '|', d.get('abort_reason'))
print('retries', d.get('initial_retries_used'))
for p in d['points'][:3]: print(p)
"
```
The three failure stages mean very different things: `free_move_to_up`
(planning from home), `stitched_descent`, or the residual guard.

---

## 9. KNOWN ISSUES AND GOTCHAS

| Issue | Detail |
|---|---|
| **Collision world unused** | `ur5e_gripper.yml` (v4 tool spheres) exists and `GRASP_TOOL_COLLISION` selects it, but every run so far logs `tool_collision: false`, `robot_yaml: ur5e.yml`. **Nothing is lab-safe yet.** |
| **Tactile at 2.5 Hz** | Qt server, not the network — `TCP_NODELAY` moved it 2.4→2.9 Hz only. Workaround: `--baseline-s 8 --hold-s 8`. `hold_average` picks baseline/hold frames from each file's own range and needs ≥3 in each. |
| **uint16 wrap** | Real taxels resting near zero return `65536+x` for negative readings (observed 65395–65512 = −141…−24). Threshold is **60000**, *not* half-scale: genuine rest values reach **36336**, and 32767 would have turned 36036 into −29500. |
| **s1/s2 peak asymmetry** | 12–15% typical, up to 46% at 45°. X-symmetry measured good (0.18 mm), so unexplained. Parked. |
| **Object is bolted in sim** | `cylinder BOLTED (dynamic)`, measured movement 0.000 mm. Real rig uses a fixture. Paper 1's adapter allowed conforming without rigid anchoring. |
| **Disk space** | Root partition hit 100% and broke Linux boot. A cuRobo CUDA compile that dies from a full disk leaves a **stale lock** in `~/.cache/torch_extensions/py311_cu128/` — delete the whole subfolder, not just the `lock` file. Never delete that cache casually: a rebuild is 20–40 min. |
| **UR calibration warning** | The driver warns the robot's calibration doesn't match the URDF, so FK is the *nominal* UR5e. Puts a mm-level floor on real pad accuracy. `ur_calibration` extraction not yet done. |

---

## 10. PARKED (deliberately, with the reason)

| Item | Why parked | Unblocked by |
|---|---|---|
| Real-robot data collection | No 3D-printed rod; no real `TOOL_OFFSET_Z`; no measured object position | Vincent's printed rod + one calibration |
| Sim/real GUI toggle | `collect_real.py` works standalone; toggle is ~100 lines | A short session at the robot |
| Rolled-pad datasets | §7b — renderer misrepresents diagonals | Berith |
| Other shapes (sphere, cuboid) | Waiting on Berith's meshes; also the designer/calibration/reachability all assume a cylinder | Meshes + per-shape sweep limits |
| Calcul Québec | Free HPC, TB of scratch. **Not a config change** — Isaac needs an Apptainer container, GPU passthrough, headless operation, cuRobo compiled inside. Does not make one run faster; enables many in parallel | JP's sponsorship (requested) + a consultation (booked) |
| Per-point pad-symmetry probe | The probe reads the live stage at closed grip — the exact moment that once exploded PhysX. 25×/run is 25× the exposure | Only worth it for provenance |
| GSR metric | Needs Paper 2's network sampling a 7×4 window out of the 96×96 prediction at each candidate pose | Block 4 |
| Early stopping / `--max-roll-deg` filter | Small changes; irrelevant until there is more data | — |

---

## 11. IMMEDIATE NEXT STEPS

1. **Diagnose the Ø13 @ 25° failure** — capture the terminal output and the
   ledger (§8).
2. **Upright-only training run.** Train *and* validate on the six new upright
   runs only. This is the §7b-clean result and would settle whether TC 6.48 was
   a mixed-validation artefact. Also run `--anchors all` for comparison.
3. **Send Berith the three questions** (§6.4). This gates 24 of the 30 runs.
4. **Flat-plate grasp** to measure the per-taxel bias map (§6.2) — one run,
   turns a description into evidence.
5. Then either: collect the 90° set (trustworthy per §4 findings, ~14 h), or
   start **Block 4** — Paper 2's method on the same pairs.

---

## 12. FILE MANIFEST (current versions)

| File | Location | Notes |
|---|---|---|
| `main_gui.py` | project root | grid designer, pad-axis stepping, anchored-pairs checkbox, Data folder button, grid accuracy |
| `collect_from_config.py` | `sim/` | retry+abort, ledger with attempts |
| `stitching.py` | `viz/` | pinned canvas, `resolve_initial`, `anchor_kinds`, `export_anchor_pairs`, pad tip, shared colour scale |
| `heatmaps.py` | `viz/` | 4 panels: s1, s2, EXPECTED, pose; pad tip everywhere; band derived from diameter |
| `blob_axis.py`, `validation.py`, `grid_accuracy.py` | `viz/` | |
| `dataset.py`, `model.py`, `train.py` | `train/` | `survey_runs`, `_run_name`, `--anchors`, pt00-only validation |
| `collect_real.py`, `real_motion.py`, `move_test.py`, `read_pose.py`, `sim_pose_check.py` | `Real_Robot/` | |

**If a file is needed for a change, ask me to upload it** — do not work from
memory of an older version. Several bugs in this project came from exactly
that.

---

## 13. BUGS FIXED THIS SESSION (so they are not reintroduced)

1. `INITIAL_GRASP = "first"` silently substituted pt01 when pt00 was missing.
2. `export_pair` painted its input from `center_index` while all metadata
   described `_initial_index` — one file describing **two different grasps**,
   42 mm apart.
3. The GUI reported a refused export as success, in green, with the filename
   "None".
4. Rolled pads stepped along **world** Y/Z, shearing the swept region.
5. Pairs were 74×74 / 87×48 / 99×99 / 116×116 — unbatchable. Now pinned.
6. Column 3 of the stitch figure had **no `vmin`/`vmax`**, so identical
   pressure rendered as a different colour from column 1.
7. `heatmaps.py` called `stitching._draw_pad_tip` with newer arguments and
   silently lost the marker on older versions — the dependency was removed
   entirely.
8. `to_pad_frame` **raised** when pt00 was missing, so `export_pair` crashed
   with a traceback instead of refusing cleanly.
9. Run names for anchored pairs resolved to **"Stitched"**, which would have
   scattered one sweep's anchors across train *and* validation.
10. `--anchors` changed the **validation set**, so `copy` — a fixed rule — scored
    differently in each run and the three modes were graded on different exams.
11. Real-robot flange targets assumed a vertical tool axis: **1.1 mm** pad error.
12. Ctrl-C did not cancel the buffered trajectory; the arm resumed after e-stop.
13. `uint16` unwrap at half-scale would have turned a genuine 36036 into −29500.
14. `record()` restarted its clock per call, making the CSV time column
    non-monotonic.
