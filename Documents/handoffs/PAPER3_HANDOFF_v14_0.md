# PAPER 3 — HANDOFF v14.0

Written 8 September 2026. Supersedes v13.0 where they disagree.

Keep for history, not repeated in full: **v13.0** (hold-window bug, residual guard, calibration redo, real-robot collector, Real Robot tab), **v12.0** (grid thinning, throat jam, grid_quality / grid_accuracy / run_manifest, closing mode), **v11_2** (shape rules, scaled cuboid, shape-keyed calibration — internally titled "v13.0"; go by file name), **v10_0** (Block 2/3, diagonal blob), **v9_0** (oblique contact, motion pipeline).

---

## 0. THE THIRTY-SECOND VERSION

Since v13.0 the work has been almost entirely in `design_grid`. A **transpose bug** was found that made every rolled-pad grid wrong, and fixing it properly meant replacing two independent limit calculations with one coupled solve valid at any pad roll and any object tilt. A second fault — an **unchecked across anchor** — was found immediately afterwards and closed, which retires the last item on the known-broken list.

Twelve upright sweeps are collected (6 cuboid, 6 cylinder). A **rolled 90° cylinder batch** is being started now, on five diameters.

**Kourosh is on Paper 2 reviewer responses for the next two weeks.** Nothing in this document is urgent. Block 3 remains the priority when he returns.

---

## 1. THE OBJECTIVE (unchanged)

Kourosh Jolaei, PhD candidate, CoRo Lab, ÉTS Montréal. Supervisors Vincent Duchaine and Jean-Philippe Roberge.

- **Paper 1** (J. Robotics and Mechatronics 38(3), 2026) — shape classifier + hand-crafted extrapolation. 94.3% held-out, 80.8% unseen real. Safe-zone TC threshold 4.42 mm.
- **Paper 2** (Robotics and Autonomous Systems) — tactile-guided regrasping, CDT, 602 trials. **Reviewer comments received; response in progress.**
- **Paper 3** (this work) — replace the classifier and hand-crafted extrapolation with a **learned contact-completion model** trained on large-scale Isaac Sim data, then validate on the real rig.

**Four blocks.** Block 1 collection: complete. Block 2 stitching: complete and closed. Block 3 U-Net: **not started**. Block 4 A/B vs Paper 1: not started.

**Stack.** UR5e + Robotiq 2F-85, two TSF-85 pads (7×4 = 28 taxels each), Isaac Sim 5.1, TSF-85 extension by Berith Atemoztli De la Cruz Sánchez, cuRobo, Ubuntu 22.04. Real cell adds ROS 2 Humble, MoveIt, `ur_robot_driver`.

---

## 2. WHERE THE PROJECT IS

| Area | State |
|---|---|
| Block 2 stitching | Closed. |
| Calibration | 7 keys, both files, one basis (deformation / 1.40 mm). |
| Upright cuboid sweeps | 6 collected. **Not yet analysed.** |
| Upright cylinder sweeps | 6 collected. **Not yet analysed.** |
| Rolled 90° cylinder sweeps | Batch starting now, 5 diameters. |
| `design_grid` | Rewritten for arbitrary roll and tilt. Anchor now owned by the designer. |
| Real robot | Motion, guards, home, shapes all working in free air. |
| Real tactile | **Blocked** — one pad missing from the lab. |
| Real calibration | **Not measured.** Every real run is `sim_fallback`. |
| Block 3 | Not started. |
| Spheres | Untouched, deliberately. |

### Verify file freshness before editing

| File | Signature |
|---|---|
| `main_gui.py` | `THE COUNTS ARE SOLVED TOGETHER` and `across_max_mm` |
| `viz/stitching.py` | `HOLD_PCTL` |
| `viz/grid_quality.py` | `FRAME_RATIO_MAX` / `FRAME_MIN` |
| `sim/collect_from_config.py` | `pad_to_pad_residual` |
| `examples/run_batch.py` | `GRASP_OBJECT_BOX_MM` |
| `Real_Robot/collect_real.py` | `home_vertical` / `arrival_residual` / `cal_key` |
| `viz/heatmaps.py` | `MIRROR_S2 = False` and `INDENT_MM = 1.4` |

---

## 3. THE ROLLED-GRID TRANSPOSE — THE MAIN FIND

### 3.1 What was wrong

Every limit in `design_grid` is stated in the **object's** frame: the contact band constrains motion **across** the object, the palm and the ends constrain motion **along** it. But `_offsets` builds the lattice and then calls `rotate_offsets(offs, pad_roll_deg)`, which turns it into the **pad's** frame *after* those limits have been applied.

At 0° the rotation is the identity, which is why every upright design ever made is correct and this survived unnoticed for months. **At 90° it transposes them**: the count derived from the along-window lands across the object, and vice versa.

**Measured**, on a Ø26 × 140 design at 90° roll: `across_max` was 23.03 mm and the sweep reached **±36.0 mm** across the object, while only ±18.0 mm of the ±36.0 mm along-window was used. The two spans were simply swapped. **42 of its 91 points could not touch the rod at all.** Nothing refused, the maps would have stitched, and the outer columns would just have been empty.

### 3.2 The general rule

A lattice of (nₐ, nₗ) steps of `step`, rotated by φ, reaches its extremes at the corner:

```
across = step · (nₐ·|cos φ| + nₗ·|sin φ|)
along  = step · (nₐ·|sin φ| + nₗ·|cos φ|)
```

The two counts are **coupled** at every angle except 0° and 90°, and no pair of independent ceilings can express that.

### 3.3 The fix

`design_grid` now solves both counts together: the largest point count whose corner satisfies all four limits — the across band, the along window, and both canvas axes — with ties going to the squarer grid. The search is small and exact.

**φ is measured against the OBJECT, not the world.** Tilting the rod rotates its across direction just as rolling the pad rotates the grid, and only the angle *between* them decides how the sweep projects. The canvas keeps the world roll, because the 96 mm canvas is world-axis aligned.

**Regression: 52 combinations at 0°** — six cylinder widths, six cuboids, a sphere, both lengths, overhang on and off — **all bit-identical** to the old arithmetic. Nothing upright changes.

**Ø26 × 140 across roll:**

| roll | n_across | n_along | points |
|---|---|---|---|
| 0° | 2 | 3 | 35 |
| 30° | 2 | 3 | 35 |
| 45° | 2 | 3 | 35 |
| 75° | 3 | 3 | 49 |
| 90° | 4 | 3 | 63 |

**Note:** 90° gives *more* points than 0°, because the pad's long axis now lies across the rod where the tight band is, so the pad reaches further past the strip. The rolled dataset is therefore **denser than the upright one at the same step** — correct, but the two batches are not comparable point-for-point.

---

## 4. THE ACROSS ANCHOR — THE SECOND FIND

Immediately after the transpose fix, six rolled cylinder designs were built with the y anchor left at **+28.42 mm** from an earlier auto-y experiment. Every report said ACCEPTED. Between **3 and 45 points per design could not touch the rod.**

The four scalar limits describe the sweep's **half-span**; nothing checked where its **centre** was. `z` had always been derived (the midpoint of the along window); `y` was the one anchor left typed, and that asymmetry is what let a stale number through.

**Fixed.** `design_grid` now owns both anchors:

- **Cylinder or sphere** → y is forced to 0, with a dialog naming the old value and why it was wrong. A rod has no side edge; the contact strip sits at its axis wherever the pad goes, so the only correct anchor is 0.
- **Cuboid or cube** → auto-y works exactly as before; a typed y is now **gated against `across_max`** and reset to the derived edge anchor if the outermost column would sit past what the pad can reach.

This retires **"No ACROSS-side limit on a hand-typed y"**, on the known-broken list since v12.0 §9.2.

### The checkbox rules, restated

| checkbox | upright cylinders | cuboids |
|---|---|---|
| auto y anchor | OFF (now ignored anyway) | ON |
| let the pad hang past the ENDS | ON | ON |
| centred grid | ON | OFF |
| step along pad axes | ON | ON |
| coarse interior | OFF | ON (N=2, band −11) |

For **rolled** cylinders, auto-y and pad-hang make no difference: they relax scalar limits, but the binding constraint at 90° is the **measured gripper body**, which they do not touch. Verified on Ø72: with both on, the along window widened 44.08 → 63.08 mm and the trim went 6 → 8 steps, landing on the same 3 points.

---

## 5. THE ROLLED 90° CYLINDER BATCH

Designed at y = 0, step 6 mm, no overhang, no auto-y, no thinning.

| Ø | across_max | n_across × n_along | points | trimmed | worst rod-to-gripper gap |
|---|---|---|---|---|---|
| 12 | 20.30 | 9 × 7 | 63 | no | +32.96 mm |
| 18 | 21.62 | 9 × 7 | 63 | no | +22.41 mm |
| 26 | 23.03 | 9 × 7 | 63 | no | +16.09 mm |
| 40 | 25.00 | 9 × 9 | 81 | no | +4.88 mm |
| 60 | 27.26 | — | 21 | 4 steps | +2.46 mm |
| 72 | 28.42 | — | 9 | 6 steps | +4.55 mm |

**The trim on Ø60 and Ø72 is physics, not a fault.** On a fat rod the surface falls away fast — at 24 mm off the axis a 36 mm radius has dropped 9.2 mm behind the tangent plane. The compliance band describes the *contact strip*, not the gripper **body**, and rolled 90° the pad's world-Y half-extent is 18.5 mm instead of 11.0, so the fingers reach the rod's shoulder long before the pad runs out of strip.

**Decision: Ø72 was dropped from the rolled batch.** Nine grasps over a 37 × 34 mm swept area carries almost no extension for a completion model to learn from. Ø60 at 21 points is marginal and was kept.

**Note on the CNN:** collecting rolled data at all is only defensible because 0° and 90° are the two angles Berith's CNN renders correctly (v9.0 §3.3: 0° reads −1.25°, 90° reads +89.7°). Intermediate angles still collapse into round blobs. `design_grid` now handles any angle correctly, but **the renderer does not** — do not collect 20°, 30° or 45° until the new extension passes its acceptance test (§8.1).

---

## 6. THE REPORT WINDOW

`design_grid` used a `messagebox`, which does not scroll — the Ø72 diagnosis was unreadable. It now writes a full **design report**: a scrollable window with **Copy** and **Save to session folder**, and Save Config drops the same text beside `gui_config_used.json` automatically.

So from now on every run folder records **why its grid has the counts it has** — the band, the window, the coupling, the reach against each limit, and any point-check trim. The twelve upright sweeps predate this and have no report.

**One bug of mine, now fixed:** the report formatter referenced `self.vars['coarsen_interior']` (the *function* name) instead of `coarse_interior`, which threw a `KeyError` before the window was ever built. That is why the scroll and Save appeared not to exist.

---

## 7. WHAT IS LEFT

### 7.1 Immediately on return

1. **Finish the rolled 90° cylinder batch** (5 diameters, in progress).
2. **Analyse all sweeps** — 6 upright cuboid, 6 upright cylinder, 5 rolled cylinder. Stitch → Grid Accuracy → Grid Quality → Save Run Manifest on each, then send `grid_quality_s1.txt` and `stitch_report.txt` as a set. Pad asymmetry like the Ø40 s1/s2 1.83 vs 1.35 mm is far easier to spot across widths than within one run.
3. **Block 3 — train the U-Net**, and nothing else. Pipeline exists (`dataset.py`, `model.py`, `train.py`) with masked loss, split-by-run, and baselines against zero and copy-input. **Use edge-straddling anchors, not pt00** — interior anchors halved TC error, 5.96 → 3.00 mm, crossing Paper 1's 4.42 mm threshold.
4. **Then Block 4** — A/B against Paper 1's shape rules.

### 7.2 Blocking real data (parallel, not on the critical path)

5. **Second tactile pad missing from the lab.** All real tactile numbers so far are meaningless — one run wrote all-zero maps across every taxel.
6. **No real-rig `TOOL_OFFSET_Z`.** Every real run is `sim_fallback`, putting a constant unverified offset into every real map. A real calibration procedure does not exist yet.
7. **Tactile frame rate.** 21 Hz on a freshly started Qt app, decaying to 1–3 Hz through a run — server load, not a fixed limit. At 1–3 Hz the hold window holds 5–10 frames against the sim's ~210.

### 7.3 Known-broken / not-yet-done

- `viz/validation.py` still assumes an axis-aligned pad footprint. **Correct on flat runs only — skip Validate on rolled runs.**
- **GSR is saturated.** Do not quote it.
- **A refused design does not block Save Config.**
- **Save Config must be pressed after Design grid.** Three configs were sent this session that still held the previous design. The report says the true count; the JSON says what was last saved.
- **Cuboid yaw is undefined.** Pads meet the +X/−X faces. Correct for a FACE grasp, but nothing records it, and an EDGE or corner grasp is a different signature Paper 1's rule does not cover.
- **`probe_finger_throat.py` models every object as a rod.** The design-time check uses the real box; the probed depth behind it does not.
- **`plot_scale.json` is `shared: true, fixed_vmax: null`** — each run scales to its own max, so brightness is not comparable across runs. Set 2400 (matching Paper 2) before pooling or putting two runs side by side.
- **The batch queue labels every entry "Ø30x140"** — `main_gui` writes the label from the grip width. Cosmetic; the folder path is what runs.
- **`--home-above-object` with `--object-here` OFF has never been run.** The real cell has **no collision world**, so IK will approve a path that sweeps through the table. Dry-run first with a typed object centre.
- **100 mm vs 140 mm bodies** — do not pool without saying so. One free-air real cuboid test used H=100.

### 7.4 Parked deliberately

- **Spheres.** Untested end to end. Berith's "spheres" are 40×40×90 and 65×65×90 capsules.
- **Intermediate pad rolls (20°, 30°, 45°).** `design_grid` handles them correctly now; the renderer does not.
- **Sequential tactile data.** The raw CSVs keep all ~250 frames per grasp permanently, so sequential pairs can be rebuilt from the same runs at any time without re-collecting. **Raise this at the start of Block 3** — it is a pair-construction decision, not a collection one.
- **A live tactile view for the real rig.** Use the Qt app; a second client on the same socket would compete for frames that are already scarce.

---

## 8. CARRIED FORWARD

### 8.1 The diagonal blob artifact — still the biggest scientific limitation

Berith confirmed it is real and is working on it. 0° gives a clean vertical ridge and 90° a clean horizontal one; 20° and 45° collapse into broad blobs — at 45° the map has the **highest sum (12195)** with the **lowest peak (999)**. A completion model trained on maps where diagonal ridges render as round blobs would learn the renderer's failure, and no downstream metric would reveal it.

Berith is delivering a new TSF-85 Isaac Sim extension and later a Newton port. **Acceptance test, unchanged:** re-run 0/20/45/90 on the new extension and confirm 20° and 45° give elongated ridges before letting any intermediate-angle grasp into a training set.

### 8.2 Overlap sigma is still unexplained

Cylinder 46%, cuboid 36%. Squeeze variation was ruled out in v12.0 §7; collapsed hold windows were ruled out in v13.0 §4.4. **The row-gain artifact is the only candidate left standing.**

### 8.3 A small no-contact floor

Both sensors hold ~85 counts (7% of peak) at Y 221–225 on the Ø26 stitch, 9–13 mm outside the silhouette, after baseline subtraction. If you train on no-contact cells rather than masking them — which ShapeGrasp's free-space argument supports — that floor is what the model learns as "no contact".

### 8.4 A related paper worth citing

**ShapeGrasp: Simultaneous Visuo-Haptic Shape Completion and Grasping**, Rustler & Hoffmann, arXiv:2605.02347v2, June 2026 (unrefereed — cite as such). Three things to take: their free-space constraint argues for training on no-contact cells; their metric set (CD, HD, JS, F1, and the precision/recall split separating hallucinated from missing geometry) is sharper than SSIM alone for Block 4; and their reported failure mode is that tactile input *inflates* the completion — pre-register that as a check.

---

## 9. CONSTANTS AND FORMULAS

| Name | Value | Provenance |
|---|---|---|
| PALM_DROP_MM | 86.69 | measured: 10.79 + 75.9 |
| THROAT_MARGIN_MM | 1.0 | bracketed |
| DEAD_GRASP_FLOOR | 0.0002 m | rest 0.000, firm 1.3–1.5 |
| CONTACT_TARGET | 0.0014 m | uniform basis for every shape |
| across_margin_mm | 3.0 mm | pad still touching at the outermost column |
| INDENT_MM (heatmaps) | 1.4 mm | was 2.4; measured 1.16 on Ø26 |
| indent_mm (coarsen) | 1.4 mm | from the Ø26 measurement |
| HOLD_FRAC | 0.9 | unchanged |
| HOLD_PCTL | 95.0 | v13.0 §4 |
| FRAME_RATIO_MAX / FRAME_MIN | 2.0 / 50 | healthy is ~210 frames |
| BLOCK_SHORTFALL_RAD | 0.005 | good points scatter 0.06 mrad |
| WEAK_DEFORM_FRAC | 0.70 | of the run's own median |
| canvas_mm | 96.0 | 4 halvings; **world-axis aligned** |
| PAD_W, PAD_H | 22.0, 37.0 mm | 4 cols × 5.5, 7 rows × 5.29 |
| OBJ_BASE_Z | 982.2 mm | obj_z = 982.2 + L/2 |
| grid step default | 6.0 mm | 0.5 mm sub-taxel shift per step |
| contact half-width | sqrt(2·R·delta) | compliance-limited, NOT silhouette |
| GRASP_MAX_RESIDUAL_MM | 3.0 | sim; now on BOTH routes |
| --arrival-tol-mm | 1.0 | real; arrivals run 0.002–0.169 |
| --p2p-max-mm | 25.0 | real pad-to-pad ceiling |
| BASE_IN_WORLD_MM | [20.930, −337.500, 992.750] | measured 12 Aug, re-confirmed 5 Sept to 0.009 mm |
| real TOOL_OFFSET_Z | **unmeasured** | every real run is sim_fallback |

### Calibration keys (all 7, both files, deformation / 1.40)

| key | TOOL_OFFSET_Z (contact) | close_rad | hold indentation |
|---|---|---|---|
| 12.0 | 0.15717 | 0.6628 | 1.54 mm |
| 18.0 | 0.15695 | 0.6079 | 1.45 mm |
| 26.0 | 0.15649 | 0.5451 | 1.59 mm |
| 40.0 | 0.15473 | 0.4097 | 1.83 mm |
| 60.0 | 0.15048 | 0.2168 | 1.57 mm |
| 72.0 | 0.14677 | 0.0930 | 1.44 mm |
| 30.0\|cuboid | 0.15601 | 0.4994 | 1.72 mm |

TOOL_OFFSET_Z spans 0.1468–0.1572 m: borrowing the wrong entry is worth up to 10 mm of pad position.

### Object size plan

**Cylinders**, length 140 mm, spaced so the contact strip steps evenly (~2 mm), since the strip goes as sqrt(D):

Ø12 (strip 7.7, 35% of pad), Ø18 (9.6, 44%), Ø26 (11.7, 53%), Ø40 (14.7, 67%), Ø60 (18.1, 82%), Ø72 (19.9, 90%). Ø72 is the ceiling: at delta = 1.4 a cylinder needs Ø86 to fill the pad and the jaw opens 85.

**Cuboids**, grip 30 mm and along 140 mm fixed, across varied: 8, 14, 22, 34, 60, 120 mm. Grip is fixed because on a flat face the jaw gap changes nothing about the map — it is a nuisance parameter, and fixing it means one calibration and one throat probe cover all six.

---

## 10. FILE INVENTORY — CHANGED SINCE v13.0

| File | Change |
|---|---|
| `main_gui.py` | `design_grid` counts solved together for arbitrary roll and object tilt; `across_max_mm` returned; y anchor owned by the designer (forced 0 for round shapes, gated for flat); scrollable design report with Copy and Save; report written beside every saved config; `coarse_interior` key typo fixed |

Unchanged since v13.0: `viz/stitching.py`, `viz/grid_quality.py`, `viz/grid_accuracy.py`, `viz/run_manifest.py`, `viz/heatmaps.py`, `viz/blob_axis.py`, `viz/validation.py`, `sim/collect_from_config.py`, `examples/run_batch.py`, `examples/probe_finger_throat.py`, `Real_Robot/collect_real.py`, `Data/build_object_library.py`.

**The sweep sequence:** Design grid → **Save Config** → run → Stitch → Grid Accuracy → Grid Quality → Save Run Manifest.

---

## 11. HOW TO WORK ON THIS

- One verifiable step at a time. **Measure, don't guess.** Distances in mm.
- Complete files, never diffs. Commands on one line — multi-line `\` continuations silently drop env vars, and a prefix before `cd` does not survive `&&`.
- **One-line answers per question unless depth is asked for.**
- **Kourosh's visual read of the simulation overrides diagnostics when they conflict.** It has been correct every time.
- **Bad data must refuse LOUDLY rather than degrade silently.**
- Never present schematic pseudocode as a quotable line from an actual file.
- **Compute before asserting.** Twice this session a number was stated from memory and was wrong — a contact band given as 39.2 mm when it was 7.53, and a dead-point count that followed from it. The diagnosis survived; the arithmetic did not. Run it.

### The pattern behind almost every fault in this project

**A wrong number that looked normal.** A hardcoded "cylinder" in a config. A grasped width taken as the minimum dimension. A calibration key two shapes shared. A gripper cloud filtered on the wrong axis. A ledger field that could not fail. A warning computed and then discarded. A guard on only one of two routes. A selection rule that discarded the data it was meant to average. A config knowing more than the code read.

**Two added this session, and they rhyme with all of them:**

- **Limits applied in one frame, then the geometry rotated into another.** The transpose was invisible at 0° and total at 90°.
- **A half-span checked and a centre not.** Every one of the six bad designs reported ACCEPTED.

None threw an error. **Every check prints WHAT IT MEASURED, not just its verdict — keep that up.**

---

*End of handoff v14.0.*
