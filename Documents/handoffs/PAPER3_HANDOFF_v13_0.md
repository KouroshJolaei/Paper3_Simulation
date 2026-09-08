# PAPER 3 — HANDOFF v13.0

Written 5 September 2026. Supersedes `PAPER3_HANDOFF_v12_0.pdf` where they disagree.

Keep for history and not repeated here in full:

- **PAPER3_HANDOFF_v12_0.pdf** — the immediate predecessor. Grid thinning (`coarse_interior`), the throat jam and its three root causes, the expected panel, `grid_quality` / `grid_accuracy` / `run_manifest`, the closing-mode decision.
- **PAPER3_HANDOFF_v11_2.pdf** (internally titled "v13.0" — the numbering drifted on disk; go by file name) — shape rules `contact_band_mm`, the scaled-cuboid primitive, shape-keyed calibration, the six limits of `design_grid`.
- **PAPER3_HANDOFF_v10_0.md** — Block 2 / Block 3 and the diagonal-blob finding.
- **PAPER3_HANDOFF_v9_0.pdf** — oblique-contact characterisation and the motion-pipeline repairs.

---

## 0. THE THIRTY-SECOND VERSION

**Block 2 is closed.** Both in-flight sweeps were analysed and passed. Three real faults were found and fixed on the way, one of which had been silently corrupting maps in every run ever collected.

**Calibration was redone from scratch** on one uniform basis: seven objects, both closing modes, deformation target 1.40 mm.

**Six cuboid sweeps were collected** over the weekend batch. Six cylinders are queued and not yet run.

**The real UR5e is now a working collector.** Frame verified to hundredths of a millimetre, pad-to-pad motion, a measured arrival guard, a constructed vertical home, a tilt gate, and cuboid support — all proven in free air. A **Real Robot tab** now exists in the GUI.

**Two things block real data:** the second tactile pad is missing from the lab, and no real-rig `TOOL_OFFSET_Z` has been measured.

**Block 3 has still not started.** That remains the honest priority.

---

## 1. THE OBJECTIVE (unchanged)

Kourosh Jolaei, PhD candidate, CoRo Lab, ÉTS Montréal. Supervisors Vincent Duchaine and Jean-Philippe Roberge.

- **Paper 1** (J. Robotics and Mechatronics 38(3), 2026) — shape-classifier + hand-crafted extrapolation. 94.3% held-out, 80.8% unseen real objects. Safe-zone TC threshold **4.42 mm**.
- **Paper 2** (submitted, Robotics and Autonomous Systems) — tactile-guided regrasping, CDT paradigm, 602 trials.
- **Paper 3** (this work) — replace the classifier and the hand-crafted extrapolation with a **learned contact-completion model** trained on large-scale Isaac Sim data, then validate against the real rig.

**Four-block plan.** Block 1 collection: complete. Block 2 stitching: **complete and closed**. Block 3 U-Net training: **not started**. Block 4 A/B against Paper 1's rules: not started.

**Stack.** UR5e + Robotiq 2F-85, two TSF-85 capacitive pads (7×4 = 28 taxels each), Isaac Sim 5.1, TSF-85 extension by Berith Atemoztli De la Cruz Sánchez, cuRobo, Ubuntu 22.04. Real cell adds ROS 2 Humble, MoveIt, `ur_robot_driver`.

---

## 2. WHERE THE PROJECT IS

| Area | State |
|---|---|
| Block 2 stitching | **Closed.** Both shapes pass all five §0 checks of v12.0. |
| Calibration | **Redone.** 7 keys, both files, one basis (deformation / 1.40 mm). |
| Cuboid sweeps | 6 collected in batch over the weekend. Not yet analysed. |
| Cylinder sweeps | 6 designed, **not yet run**. |
| Real robot | Motion, guards, home and shapes all working in free air. |
| Real tactile | **Blocked** — one pad missing from the lab. |
| Real calibration | **Not measured.** Every real run so far is `sim_fallback`. |
| Block 3 | Not started. |
| Spheres | Untouched, deliberately. |
| Rolled / tilted data | Still blocked on Berith's CNN. Collect 0° and 90° only. |

### Verify file freshness before editing

| File | Signature to grep for |
|---|---|
| `viz/stitching.py` | `HOLD_PCTL` |
| `viz/grid_quality.py` | `FRAME_RATIO_MAX` / `FRAME_MIN` |
| `sim/collect_from_config.py` | `pad_to_pad_residual` |
| `examples/run_batch.py` | `GRASP_OBJECT_BOX_MM` |
| `Real_Robot/collect_real.py` | `home_vertical` / `arrival_residual` / `cal_key` |
| `main_gui.py` | `_build_real_tab` / `real_use_session` |
| `viz/heatmaps.py` | `MIRROR_S2 = False` and `INDENT_MM = 1.4` |

---

## 3. CLOSING OUT THE TWO SWEEPS IN FLIGHT

Both runs from v12.0 §0 were analysed against the five checks.

**Cuboid — `run_20260903_141200_obj0_pad0`, 30×120×140, 64 points, coarse interior ON (N=2, band −11).**

Four checks passed outright. Check 2 failed on a **single point**:

- `pt32` landed **4.20 mm high in Z** and 0.59 mm short in Y — a miss of 4.243 mm, **77.14% of a taxel**.
- Excluding pt32, scatter is **0.17% of a taxel** — exactly the historical figure. The reported 9.80% was entirely that one point.
- The close itself was normal: `close_rad_stopped` 0.49105, deformation 1.21 mm. So it is a **placement fault, not a jam**.
- Recorded as `exec_stage: complete`, `predicted_ok_executed_ok`, 0 of 64 flagged.

**Cylinder — `run_20260903_173806_obj0_pad0`, Ø26×140, 45 points, thinning OFF.** All five checks passed. Scatter 0.14% of a taxel, worst miss 1.21%. Two WEAK points (pt36, pt37) at the outermost across column — real geometry, not a fault.

**The throat fix held on both.** Cuboid bottom row (z = 25.29) reached 1.04–1.40 mm; cylinder bottom row (z = 23.53) was the *strongest* row at median 1.52 mm. Close-angle shortfall spread was 2.74 mrad (cuboid) and 1.31 mrad (cylinder) against 16.8 mrad for the original jam.

**Verdict: the collection pipeline is cleared for cuboid and cylinder.**

---

## 4. THE HOLD-WINDOW BUG — THE BIGGEST FIND OF THE SESSION

### 4.1 How it was found

`grid_quality.py` was extended to keep the frame count that `stitching.hold_average` had always returned and every caller threw away, and to compare s1 against s2 per point.

It fired on **13 of 45 cylinder points (29%)** and **3 of 64 cuboid points (4.7%)**. The low side was not marginal — it was 1, 2, 3, 4 frames against a healthy ~210.

The flags clustered at the extremes of the sweep, and mirrored: on the cylinder, s1 collapsed across the whole `y = −8.47` column while s2 collapsed at `y = +15.53`.

### 4.2 The cause — the rule, not the data

Every frame was always in the CSV. `hold_average` selected almost none of them.

`hold_mask = s >= smin + HOLD_FRAC * (peak − smin)` with `HOLD_FRAC = 0.9`, and `peak = s.max()`.

Measured on `pt43`, printing every frame's taxel sum:

| | plateau | frames at plateau | release spike | threshold | frames kept |
|---|---|---|---|---|---|
| s1 | 2027 | ~210 | 2170 (1.07×) | 1977.8 | **210** |
| s2 | 1647 | 213 | **2102 (1.28×)** | 1916.6 | **1** |

s2's single surviving frame was the **release transient** — the instant the gripper let go. That map went into the stitch and into the training pairs with nothing saying so. Its sum was 1846 against a true hold of 1391: **33% too bright**.

**The rule breaks whenever the release spike exceeds the plateau by more than about 11%.** Nothing to do with contact quality, which is why healthy-looking points were flagged.

### 4.3 The fix

`stitching.hold_average` now takes the top of its range from the **95th percentile** (`HOLD_PCTL = 95.0`) instead of `s.max()`, so a 1–3 frame transient cannot set the bar.

Measured on the same two traces: **s1 210 → 211, s2 1 → 213.** Healthy grasps move by a frame; broken ones are fully recovered — the property that makes it safe to apply to already-collected runs.

After the fix and a re-stitch, **every flag on both runs disappeared.** The cylinder's only remaining verdicts are pt36/pt37 WEAK.

### 4.4 What it did NOT explain

Overlap sigma was unchanged across the fix — cylinder stayed at **46%**, cuboid at **36%**. So the collapsed hold windows are **not** why the cylinder smears more than the cuboid. **The row-gain artifact remains the only candidate standing.**

### 4.5 The flag itself has two tests

`FRAMES` is a *ratio* test between sensors and is blind when both collapse together (`pt41` came through as `1/14` and only fired because 14 > 2×1). `FRAME_MIN = 50` was added as an absolute floor with its own `FRAMES-LOW` verdict.

---

## 5. THE RESIDUAL GUARD WAS NEVER ON pt32's PATH

`GRASP_MAX_RESIDUAL_MM = 3.0` was never wrong. **It was never consulted.**

The guard lives inside the `if not approached:` descent branch — the lift-and-descend route, which **only the first point takes**. Every later point moves pad-to-pad, which ran the trajectory, set `approached = True`, and jumped straight to closing. No measurement, no check. The guard covered **1 point in 64**, and pt32 was visit 33.

**Fix.** `collect_from_config.py` now measures reached-vs-commanded EE on the pad-to-pad path, trims with a short stitched line if over 0.2 mm, and refuses to close past the limit — logging `exec_stage: pad_to_pad_residual`.

**Verified on hardware-equivalent sim:** 3/3 complete at the default, printing `pad-to-pad arrival residual 0.04 mm (limit 3.0 mm) — OK` at every point.

---

## 6. THE GUI WAS RUNNING TWO DIFFERENT COPIES OF stitching.py

After the `hold_average` fix, the Stitch button used the new file and Grid Quality used the old one — the frame counts came back **byte-identical** to the pre-fix run.

`main_gui._load_stitching_module` executes `viz/stitching.py` fresh on every press but **never assigned it into `sys.modules`**. `grid_quality._peak_sums` does a plain `import stitching`, which resolves to whatever was cached at GUI startup by `heatmaps` or `blob_axis`.

**Fix:** `sys.modules["stitching"] = mod` after `spec.loader.exec_module(mod)`, plus `sys` on the import line. Applied. **The GUI must be restarted for it to take effect** — a stale module already in memory is not evicted by the new line.

This cost two full rounds of confusing results. Any future edit to `stitching.py` would otherwise have reached the stitch and not grid quality, blob axis or the manifest.

---

## 7. TWO SMALLER FIXES

**`heatmaps.MIRROR_S2` → `False`.** The heatmap panel flipped s2 left-to-right while the stitcher did not (`MIRROR_S2_IN_OVERLAY=False`), so the same grasp looked a column apart between the two views. Confirmed from the training pair: s1 and s2 both peak at Y = 198.0 with the same 192–202 span. The data was always right; only the display flipped.

**`heatmaps.INDENT_MM` 2.4 → 1.4.** The expected panel printed a 15.1 mm band on Ø26 against a measured strip of 10 mm at half-max. Now prints 11.7 mm. It recomputes from `scene["d"]` per run, so Ø18 → 9.6, Ø40 → 14.7, Ø60 → 18.1 automatically. **Cuboids are unaffected** — `blob_axis` ignores the band for flat shapes and uses `d/2` as the half-width (uniform pressure wherever pad and face overlap), which is why a cuboid correctly prints "whole 120 mm face".

---

## 8. CALIBRATION — REDONE, ONE BASIS

All seven objects were re-calibrated in mode **both (compare)**, signal **deformation**, target **0.0014 m**, then throat-probed. Both `pad_offset_calibration_fixed.json` and `_contact.json` now hold all seven keys.

| key | TOOL_OFFSET_Z (contact) | close_rad | hold indentation | target reached |
|---|---|---|---|---|
| `12.0` | 0.15717 | 0.6628 | 1.54 mm | true |
| `18.0` | 0.15695 | 0.6079 | 1.45 mm | true |
| `26.0` | 0.15649 | 0.5451 | 1.59 mm | true |
| `40.0` | 0.15473 | 0.4097 | 1.83 mm | true |
| `60.0` | 0.15048 | 0.2168 | 1.57 mm | true |
| `72.0` | 0.14677 | 0.0930 | 1.44 mm | true |
| `30.0\|cuboid` | 0.15601 | 0.4994 | 1.72 mm | true |

**Note the spread: `TOOL_OFFSET_Z` ranges 0.1468–0.1572 m.** Borrowing the wrong entry is worth up to 10 mm of pad position. This is why the real collector's old "fall back to the Ø26 value" behaviour was removed.

**Throat probe** wrote all seven keys. Ø12 and Ø18 report `min_depth 14.0 mm` against 102–111 mm for the rest. **This is correct, not a fault** — `min_depth` is the shallowest depth from which everything deeper is admissible, where admissible means no gripper part sits closer to the axis than the rod's radius. A 6 mm radius clears the profile past depth 14; a 13 mm radius does not clear until 102; a 30 mm radius until 111. Same profile, four thresholds. Verified against a real Ø12 3-point sweep: no jam, indentation 1.38–1.64 mm.

**One thing to watch:** Ø40 contact shows s1 1.83 mm vs s2 1.35 mm — the widest pad-to-pad asymmetry in the set.

---

## 9. THE OBJECT SIZE PLAN

**Cylinders — vary diameter only, length fixed at 140 mm.** Spaced so the *contact strip width* steps evenly (~2 mm), not the diameter, because the strip goes as √D and that is what the pad actually sees.

| Ø (mm) | strip | % of the 22 mm pad |
|---|---|---|
| 12 | 7.7 | 35% |
| 18 | 9.6 | 44% |
| 26 | 11.7 | 53% |
| 40 | 14.7 | 67% |
| 60 | 18.1 | 82% |
| 72 | 19.9 | 90% |

72 is the ceiling: at δ = 1.4 a cylinder needs Ø86 to fill the pad and the jaw only opens 85.

**Cuboids — vary the ACROSS face only. Grip fixed at 30 mm, along fixed at 140 mm.**

| W × D × H (mm) | across | % of pad | what the map shows |
|---|---|---|---|
| 30 × 8 × 140 | 8 | 36% | both side edges inside one pad |
| 30 × 14 × 140 | 14 | 64% | both side edges, wider |
| 30 × 22 × 140 | 22 | 100% | edges exactly at the pad edges |
| 30 × 34 × 140 | 34 | 155% | pad inside the face, edge one step away |
| 30 × 60 × 140 | 60 | 273% | edge several steps away |
| 30 × 120 × 140 | 120 | 545% | validated; edge only via auto-y-anchor |

**Why grip is fixed at 30 for every cuboid:** on a flat face the jaw gap changes nothing about the tactile map — it is a nuisance parameter. Fixing it means **one** calibration and **one** throat probe cover all six. Design-time throat still uses each box's real dimensions, so nothing is lost.

**Why length is fixed at 140:** the end is the end. It also keeps clear of the "100 vs 140, do not pool" caveat entirely.

**Range vs Paper 1:** its classifier trained on 22–42 mm cylinders and 20–42 mm cuboid thicknesses, externally validated on 10–60. This 12–72 covers and extends both.

### Grid designer settings

| checkbox | cylinders | cuboids |
|---|---|---|
| auto y anchor | **OFF** | **ON** |
| let the pad hang past the ENDS | ON | ON |
| centred grid | ON | OFF |
| step along pad axis | ON | ON |
| coarse interior | **OFF** | **ON** (N=2, band −11) |

Auto-y is off for cylinders because the contact strip stays at the rod axis wherever the pad goes, so the sweep must be symmetric about it — auto-y would push it to one side, which is what gave the Ø26 run its near-empty `+15.53` column (peak sum 2565 against 10732 at centre).

---

## 10. THE BATCH RUNNER RAN EVERY CUBOID AS A CYLINDER

All six queued cuboids failed in one second each: `no cylinder calibration for 30.0 mm (looked for key '30.0')`.

`collect_from_config.py` takes the shape **only** from `GRASP_OBJECT_SHAPE` and the box dimensions **only** from `GRASP_OBJECT_BOX_MM`, both defaulting to `"cylinder"` and `40,18,100`. It reads diameter and length from the config but not the shape. `run_batch.run_one` passed run dir, basename, headless, log_mesh, tool collision, cal file and pad rotation — and never the object block.

**Fix:** `run_batch.py` now reads `shape`, `diameter_mm`, `across_mm` and `length_mm` from each config and emits `GRASP_OBJECT_SHAPE` and `GRASP_OBJECT_BOX_MM=W,D,H`, in exactly the format `main_gui._obj_env` writes. It prints the shape and the resolved calibration key at the top of every run.

**Same fault class as the real collector's:** the config knew more than the code read.

### Disk

`log_mesh` OFF for all batch runs. The 64-point cuboid wrote **6.3 GB** of `mesh_state` — about 99 MB per grasp. Twelve sweeps would have been ~70 GB. Nothing in the analysis chain reads `mesh_state` or `deformations`; `run_manifest.py` merely inventories them and both are marked not-required. Indentation comes from the ledger.

---

## 11. THE REAL ROBOT — NOW A WORKING COLLECTOR

### 11.1 The contract (unchanged, and the reason this works)

`collect_real.py` writes exactly the three files Block 2 already reads:

- `pose_history.json` — where the pad actually was, per grasp
- `<base>_ptNN_<s>_tactile_maps.csv` — the 7×4 stream, per grasp per sensor
- `gui_config_used.json` — the grid that was asked for

Plus `execution_ledger.json` and `reachability_report.json`. So stitching, pair export, blob axis, heatmaps and grid accuracy all work on real runs **with no new code**.

### 11.2 The frame is confirmed

`world_mm = base_link_mm + BASE_IN_WORLD_MM`, no rotation, with

```
BASE_IN_WORLD_MM = [20.930, -337.500, 992.750]
```

measured 12 Aug 2026 and re-confirmed 5 Sept: bias over nine points was **(−0.001, +0.001, −0.009) mm**. That question is closed.

### 11.3 What was added this session

**Pad-to-pad motion.** The file lifted `approach_mm` and re-descended in 2 mm steps for *every* point — 8 needless round trips on a 9-point grid with 6 mm steps. Now, if the previous point succeeded, the arm is still at grasp height and the next target is within `--p2p-max-mm` (default 25), it moves sideways. Run time went **488 s → 131 s**. The at-grasp-height flag is cleared on every failure exit, so a failed point always falls back to approach-from-above, and a retry of the initial point never uses the shortcut.

**A measured arrival guard.** `--pose-tol-mm` only ever checked the *planned* IK solution. `--arrival-tol-mm` (default 1.0) reads the arm after it stops, compares to the commanded pose, and refuses to close past the limit, logging `arrival_residual`. Real arrivals run 0.002–0.169 mm, so 1.0 mm is roughly a 30× margin.

**This guard earned its keep on its first outing.** On one run the pendant program dropped mid-run; the driver went on accepting trajectories and the arm silently stayed at pt00 for eight points. Every refused distance was exactly the straight-line distance from pt00. Before this guard, that run would have written nine cheerful "ok" grasps at poses the pad never reached.

**A constructed vertical home.** Every point is commanded at `quat_home`, so whatever tilt home carries is inherited by the whole grid. A jogged home read **2.645°** off vertical — 7.2 mm of lateral pad offset over the 156 mm flange-to-pad lever, and *undetectable by the run*, because the collector correctly commands the tilted tool to put the pad on target. What it cannot do is make the pad *face* parallel to the object.

`--home-vertical` forces the tool z-axis straight down and keeps the existing yaw, so the operator still aims the closing axis by jogging. The flange does not move; the wrist rotates under it and `q_home` is re-solved at the same position. Verified: tilt 2.645° → **0.0037° max across all nine points**, yaw preserved to three decimals, rotation matrix orthonormal, det 1.

**`--max-tilt-deg`** refuses the run above a chosen tilt. Fired correctly at 2.65° against a 0.50° limit, before anything moved.

**`--home-above-object`** puts home directly above the first grid point at the approach height, as the Isaac scene does.

**Shape-keyed calibration and cuboid support.** Reads `shape`, `across_mm`, `length_mm`; builds `26.0` or `30.0|cuboid` with **no cross-shape fallback**; refuses unsupported shapes by name. Default calibration file is now `pad_offset_calibration_contact.json`, overridable with `--cal-file`. Shape, key and file are recorded in the ledger and the reachability report. Run folders are named `run_..._real_cuboid30_pad0`.

**Removed:** the old "no entry at all → use the Ø26 value 0.15657" default. Given the 0.1468–0.1572 spread that was a wrong number that looked normal. It now refuses and lists the keys the file does hold.

### 11.4 Measured real performance (free air, 9 points, Ø26 and 30|cuboid)

| | value |
|---|---|
| pad miss per point | 0.006 – 0.044 mm |
| bias | (0.005, −0.003, −0.004) mm |
| scatter | 0.13% dY, 0.27% dZ of a taxel |
| arrival residual | 0.002 – 0.169 mm |
| tool tilt | 0.00° |

That is as good as the best sim run.

### 11.5 `--object-here`, and how to use it

`--object-here` **overwrites** the config's object centre with wherever the pad is right now. It is both the safe way to test in free air *and* the way to **measure** a real object's position:

1. Put the real object in place.
2. Jog until the pads straddle it at the height you want pt00.
3. Run **dry** with `--object-here` and read the printed `object centre` line — that is the measured world position.
4. Type those three numbers into the Collection tab's object centre. Save Config.
5. From then on run with `--object-here` **off**, and `--home-above-object` will place the arm over the real object every time.

**`--home-here` and `--home-above-object` do not conflict** — the first supplies the orientation and joint seed, the second moves the position. The pair that conflicts is `--object-here` versus a typed object centre.

---

## 12. THE REAL ROBOT TAB

A sixth tab, after Batch. **Design stays entirely on the Collection tab** — shape, dimensions, object centre, tilt, pad roll, grid, steps, anchors — unchanged. Nothing on that tab knows a real robot exists.

The tab holds only what is real-only:

- a read-back of the config it will run, with the calibration key, the config path and the run path
- the home block: `home here`, `straighten wrist`, `home above object`, `object here`
- tilt limit, speed, arrival limit
- `allow SIM calibration`
- a **Pre-flight checklist** dialog
- **Show dry-run command** and **Show run command**

**It does not move the robot.** Like every other command this GUI produces, it writes a line to paste into a terminal, so the `type GO to confirm` prompt, the live log and Ctrl-C all stay in front of you.

**Session folder.** With the box ticked it reads `<session>/gui_config_used.json` and writes to `<session>/Real/REAL_<stamp>/`. The session copy is used rather than the scratch `Data/gui_config.json` because the scratch file is overwritten by the next Save Config and a run pointed at it could not be reproduced later.

**Two refusals live in the tab itself:** a config with non-zero pad roll produces no command at all, so the cell is never brought up for a run `collect_real.py` would reject anyway; and ticking the session box with no session set refuses rather than falling back.

**The pre-flight dialog** lists all nine cell start-up steps from the lab manual in order, each with a Copy button and a tickbox: pendant program loaded **and running**, no protective stop, `ping`, `ur_control.launch.py`, `ur_moveit.launch.py`, `gripper_moveit_bridge.py`, controller switch, controller **verify**, and a **freshly started** tactile Qt app.

That last one matters: the Qt server ran at **21 Hz freshly started and 1.4 Hz stale**, and the hold window is built from whatever frames arrive.

---

## 13. WHAT IS LEFT

### 13.1 Blocking Block 3 — nothing

The sim data path is clear. Block 3 can start as soon as the sweeps finish.

### 13.2 To finish the dataset

1. **Run the six cylinder sweeps.** Designed, not queued. Add the folders in the Batch tab, press *Write batch + show command*, run.
2. **Analyse the six cuboid sweeps** collected over the weekend — Stitch → Grid Accuracy → Grid Quality → Save Run Manifest on each.

### 13.3 Blocking real data

3. **The second tactile pad is missing from the lab.** All real tactile numbers so far are meaningless — one run wrote all-zero maps across every taxel.
4. **No real `TOOL_OFFSET_Z`.** Every real run is `calibration_source: sim_fallback`, which puts a constant unverified offset into every real map. A real-rig calibration procedure does not exist yet.
5. **The tactile frame rate.** 1–3 Hz stale gives 5–10 hold frames against the sim's ~210. Restarting the Qt app helps (21 Hz), but it decays through a run, which points at server load rather than a fixed limit.

### 13.4 Known-broken / not-yet-done

- `viz/validation.py` still assumes an axis-aligned pad footprint. Correct on flat runs only.
- **GSR is saturated.** Do not quote it.
- **No ACROSS-side limit on a hand-typed y.** Auto-y derives a safe value; a typed y is unchecked and on a 120 mm face, y beyond ~55 runs the pad off entirely.
- **A refused design does not block Save Config.**
- **Cuboid yaw is undefined.** The scene applies no yaw, so pads meet the +X/−X faces. Correct for a FACE grasp, but nothing records it, and an EDGE or corner grasp is a different signature Paper 1's rule does not cover.
- **`probe_finger_throat.py` models every object as a rod** (`rod_radius_mm`). The design-time check uses the real box; the probed depth behind it does not.
- **`plot_scale.json` is `shared: true, fixed_vmax: null`** — each run scales to its own maximum, so brightness is not comparable across runs. Set a fixed max (2400, matching Paper 2's figures) before pooling sweeps or putting two runs side by side in a figure.
- **The batch queue labels every entry "Ø30x140"** because `main_gui` writes the label from the grip width. Cosmetic; the folder path beside it is what runs.
- **`--home-above-object` with `--object-here` OFF has never been run.** Every real test derived the object from the current pad. The first time a typed object centre is used, dry-run first and check `first point` and `flange Z` — and note **the real cell has no collision world**, so IK will approve a path that sweeps through the table.

### 13.5 Parked deliberately

- **Spheres.** Untested end to end. Berith's "spheres" are 40×40×90 and 65×65×90 capsules, not spheres.
- **Rolled / tilted data.** Blocked on the CNN (§14.1). `collect_real.py` refuses pad roll outright.
- **Sequential tactile data.** The raw CSVs keep all ~250 frames per grasp permanently, so sequential pairs can be rebuilt from the same runs at any time without re-collecting. **Raise this again at the start of Block 3** — it is a pair-construction decision, not a collection one.
- **A live tactile view for the real rig.** Use the Qt app; a second client on the same socket would compete for frames that are already scarce.

---

## 14. CARRIED FORWARD

### 14.1 The diagonal blob artifact — still the biggest scientific limitation

Berith confirmed it is real and is working on it. 0° gives a clean vertical ridge and 90° a clean horizontal one; 20° and 45° collapse into broad blobs — at 45° the map has the **highest sum (12195)** with the **lowest peak (999)**. Collect 0° and 90° only. A completion model trained on maps where diagonal ridges render as round blobs would learn the renderer's failure, and no downstream metric would reveal it.

Berith is delivering a new TSF-85 Isaac Sim extension and later a Newton port. This is a new sensor model, not just a scene, so it is the thing that could actually fix the artifact. **Acceptance test, unchanged:** re-run 0/20/45/90 on the new extension and confirm 20° and 45° give elongated ridges before letting any tilted grasp into a training set.

### 14.2 The overlap-sigma question is still open

Cylinder 46%, cuboid 36%, unchanged by the hold-window fix. The **row-gain artifact** is the only candidate left standing. Squeeze variation was ruled out in v12.0 §7; collapsed hold windows were ruled out this session.

### 14.3 A small no-contact floor

Both sensors hold ~85 counts (7% of peak) at Y 221–225 on the Ø26 stitch, nine to thirteen millimetres outside the silhouette, after baseline subtraction. If you train on no-contact cells rather than masking them — which ShapeGrasp's free-space argument supports — that floor is what the model will learn as "no contact".

### 14.4 A related paper worth citing

**ShapeGrasp: Simultaneous Visuo-Haptic Shape Completion and Grasping**, Rustler & Hoffmann, arXiv:2605.02347v2, June 2026 (unrefereed manuscript — cite as such). Three things to take from it: their free-space constraint is the argument for training on no-contact cells rather than masking them; their metric set (CD, HD, JS, F1, and especially the precision/recall split separating hallucinated from missing geometry) is sharper than SSIM alone for Block 4; and their reported failure mode is that tactile input *inflates* the completion — pre-register that as a check.

---

## 15. NEXT STEPS, IN ORDER

1. **Run the six cylinder sweeps** in the batch queue.
2. **Analyse all twelve sweeps** as a set — `grid_quality_s1.txt` and `stitch_report.txt` from each. Pad asymmetry like the Ø40 s1/s2 1.83 vs 1.35 is far easier to spot across widths than within one run.
3. **Block 3 — train the U-Net**, and nothing else. The pipeline exists (`dataset.py`, `model.py`, `train.py`) with masked loss, split-by-run, and baselines against zero and copy-input. **Use edge-straddling anchors, not pt00** — measured earlier, interior anchors halved TC error, 5.96 → 3.00 mm, crossing Paper 1's 4.42 mm safe-zone threshold.
4. **Then Block 4** — A/B against Paper 1's shape rules.

Real-rig work runs in parallel when lab access allows, and is **not** on the critical path to Block 3:

- Fit the second tactile pad; re-measure the frame rate.
- Devise and run a real-rig `TOOL_OFFSET_Z` calibration.
- Measure the real object position with `--object-here`, type it into the Collection tab, then run with it off.
- First real grasp on a real rod, then a real stitched map.

---

## 16. CONSTANTS AND FORMULAS

| Name | Value | Provenance |
|---|---|---|
| `PALM_DROP_MM` | 86.69 | measured: 10.79 + 75.9 |
| `THROAT_MARGIN_MM` | 1.0 | bracketed: dead −0.06, D26 lowest +1.88, calibration pose +3.13 |
| `DEAD_GRASP_FLOOR` | 0.0002 m | rest 0.000, firm 1.3–1.5 |
| `CONTACT_TARGET` | 0.0014 m | 1.4 mm — now the uniform basis for every shape |
| `across_margin_mm` | 3.0 mm | pad still touching at the outermost column |
| `INDENT_MM` (heatmaps) | **1.4 mm** | was 2.4; measured 1.16 on Ø26 |
| `indent_mm` (coarsen) | 1.4 mm | from the Ø26 measurement |
| `HOLD_FRAC` | 0.9 | unchanged |
| `HOLD_PCTL` | **95.0** | NEW — see §4 |
| `FRAME_RATIO_MAX` | **2.0** | NEW — s1 vs s2 hold-frame ratio |
| `FRAME_MIN` | **50** | NEW — absolute floor; healthy is ~210 |
| `BLOCK_SHORTFALL_RAD` | 0.005 | good points scatter 0.06 mrad, bad row 16.8 |
| `WEAK_DEFORM_FRAC` | 0.70 | fraction of the run's own median |
| `canvas_mm` | 96.0 | 4 halvings |
| `PAD_W`, `PAD_H` | 22.0, 37.0 mm | 4 cols × 5.5, 7 rows × 5.29 |
| `OBJ_BASE_Z` | 982.2 mm | obj_z = 982.2 + L/2 |
| grid step default | 6.0 mm | 0.5 mm of sub-taxel shift per step |
| contact half-width | √(2·R·δ) | compliance-limited, NOT silhouette |
| `GRASP_MAX_RESIDUAL_MM` | 3.0 | sim; now on BOTH routes |
| `--arrival-tol-mm` | 1.0 | real rig; arrivals run 0.002–0.169 |
| `--p2p-max-mm` | 25.0 | real rig pad-to-pad ceiling |
| `BASE_IN_WORLD_MM` | [20.930, −337.500, 992.750] | measured 12 Aug, re-confirmed 5 Sept to 0.009 mm |
| real `TOOL_OFFSET_Z` | **unmeasured** | every real run is `sim_fallback` |

---

## 17. FILE INVENTORY — CHANGED THIS SESSION

| File | Change |
|---|---|
| `viz/stitching.py` | `HOLD_PCTL = 95.0`; `hold_average` percentile-referenced; hold rule printed in the stitch report; `hold_pctl` in the provenance CSV |
| `viz/grid_quality.py` | `FRAME_RATIO_MAX`, `FRAME_MIN`; reads both sensors' hold-frame counts; `FRAMES` and `FRAMES-LOW` verdicts; `frms` column |
| `sim/collect_from_config.py` | arrival check + trim + refusal on the **pad-to-pad** route; `exec_stage: pad_to_pad_residual` |
| `examples/run_batch.py` | emits `GRASP_OBJECT_SHAPE` and `GRASP_OBJECT_BOX_MM`; prints shape and resolved key per run |
| `Real_Robot/collect_real.py` | pad-to-pad routing; measured arrival guard; `--home-vertical`, `--home-above-object`, `--max-tilt-deg`; shape-keyed `cal_key`; cuboid support; `--cal-file`; wrong-object default removed; route/home/shape fields in the ledger |
| `main_gui.py` | `sys.modules["stitching"] = mod`; **Real Robot tab** (`_build_real_tab`, `_real_*` helpers, pre-flight dialog, session wiring) |
| `viz/heatmaps.py` | `MIRROR_S2 = False`; `INDENT_MM = 1.4` |

Unchanged but relevant: `viz/blob_axis.py`, `viz/grid_accuracy.py`, `viz/run_manifest.py`, `viz/validation.py`, `examples/probe_finger_throat.py`, `Data/build_object_library.py`.

**The sweep sequence** (unchanged): Design grid → Save Config → run → Stitch → Grid Accuracy → Grid Quality → Save Run Manifest. The manifest goes last so it captures the other outputs rather than recording them as absent.

---

## 18. HOW TO WORK ON THIS

- One verifiable step at a time. **Measure, don't guess.** Distances in mm.
- Complete files, never diffs. Commands on one line — multi-line `\` continuations silently drop env vars, and a prefix before `cd` does not survive `&&`.
- One-line answers per question unless depth is asked for.
- **Kourosh's visual read of the simulation overrides diagnostics when they conflict.** It has been correct every time.
- **Bad data must refuse LOUDLY rather than degrade silently.**
- Never present schematic pseudocode as a quotable line from an actual file.

### The pattern behind almost every fault in this project

**A wrong number that looked normal.** A hardcoded "cylinder" in a config. A grasped width taken as the minimum dimension. A calibration key two shapes shared. A gripper cloud filtered on the wrong axis. A ledger field that could not fail. A warning that was computed and then discarded.

**Three added this session, and they rhyme:**

- **A guard on only one of two routes.** `GRASP_MAX_RESIDUAL_MM` was never wrong; it was never consulted, because it lived in a branch 1 point in 64 takes.
- **A selection rule that discarded the data it was meant to average.** Every frame was in the CSV. The window kept one.
- **The config knowing more than the code read.** `run_batch.py` and `collect_real.py` both had the shape sitting in a JSON file they never looked at.

None of these threw an error. **Every check added since prints WHAT IT MEASURED, not just its verdict — keep that up.**

---

*End of handoff v13.0.*
