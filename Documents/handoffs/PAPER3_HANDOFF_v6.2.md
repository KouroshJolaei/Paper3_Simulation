# PAPER 3 — HANDOFF v6.2
**Supersedes v1–v6.1. Date: 2026-07-27.**
Paste this into a new chat together with the current code files. The next
instance can pick up at §8 (CURRENT STATE) without missing anything.

**Newest state is at the END: read §14 (v6.1) then §15 (v6.2) last.** §15 has
the four-angle tilt series, the resolved brightness-gradient explanation, and
the three related-work papers (Smith NeurIPS 2020 is the key one).

**v6.1 addendum (2026-07-24 → 27):** clean tilted run, first visible tilt in
a stitch, the stitch round-trip validation feature (`viz/validation.py`),
the real GSR recipe + its saturation caveat, and the Tuesday-meeting plan are
all in **§14 at the end**. Read §14 last — it is the newest state.

---

## 0. WHO / WHAT

**Kourosh Jolaei**, PhD candidate, CoRo Lab, ÉTS Montréal
(supervisors: Vincent Duchaine / Jean-Philippe Roberge).

**Paper 3 goal:** replace Papers 1–2's hand-crafted, shape-classifier-driven
contact extrapolation with a **learned tactile contact-completion model**,
trained on large-scale *simulated* tactile data. Removing the classifier
dependency answers the main reviewer criticism of Papers 1–2.

**Simulation:** Isaac Sim 5.1 · UR5e + Robotiq 2F-85 gripper + two TSF-85
tactile pads (7×4 = 28 taxels each) · grid-based sequential grasps on a
cylinder.

**Reference work in play:**
- Paper 1 (JRM 2026) — shape-based contact extrapolation; metrics: Tactile
  Centroid error, GSR-based error, SSIM, Safe Zone.
- Paper 2 — "Tactile-Guided Regrasping: Virtual Search Optimization via
  Extrapolated Haptic Contact" (Robotics and Autonomous Systems); CDT search.
- Berith De la Cruz Sánchez & Roberge, *Frontiers in Robotics and AI* 2025 —
  hybrid elastic–hyperelastic tactile sim; the TSF-85 Isaac extension we use.
- Roberge et al., CASE 2021 — temporal snapshots at 5% / 50% / 95% / +3 s.

---

## 1. THE BIG PICTURE — where this project is going

```
  STAGE A : DATA FACTORY            [~95% done, verified]
     GUI cockpit (main_gui.py)  ->  config JSON
     Isaac collector (collect_from_config.py)  ->  per-grasp tactile CSVs
                                                    + pose_history.json
                                                    + probes / logs

  BLOCK 2 : STITCHING               [done, verified]
     viz/stitching.py  ->  one extended contact map per sensor
                       ->  training_pair.npz  (INPUT -> TARGET)

  BLOCK 3 : MODEL TRAINING          [not started]
     Two U-Nets trained on the pairs:
       INPUT  = single centre grasp (what a robot actually feels)
       TARGET = full stitched map   (what the whole object surface looks like)
     i.e. "complete the contact you cannot feel"

  BLOCK 4 : EVALUATION              [not started]
     Reuse Paper 1/2 metrics: Tactile Centroid error, GSR error, SSIM,
     Safe Zone. Compare learned completion vs Papers 1–2 classifier method.

  BLOCK 5 : PAPER                   [not started]
     Includes an honest limitations section on simulated-sensor fidelity
     (see §9 open items — we now have measurements, not guesses).
```

**Where we are right now:** Stage A and Block 2 are finished and *verified by
measurement*. The immediate next milestone is generating a real dataset
(many grids / diameters / tilts), then Block 3.

---

## 2. ARCHITECTURE (stable)

**Stage A collector**
- `collect_from_config.py` — runs inside Isaac via `python.sh`.
- `main_gui.py` — cockpit (normal python, no Isaac). Tabs: Collection /
  Calibrate / Stitching.
- Grid of grasps. `pt00` reached by lift + descend; `pt01+` by pad-to-pad
  stitched straight lines (cuRobo), no lifting between points.
- Per-run outputs: `*_ptNN_s1/s2_tactile_maps.csv`, `pose_history.json`,
  `pad_truth_probe.json`, `reachability_report.json`, `gui_config_used.json`,
  `run_progress.log`, `temporal_snapshots.json`, plus the extension's own
  `gui_s1/s2_mesh_state.csv` and `gui_s1/s2_deformations.csv`.

**Block 2 stitcher — `viz/stitching.py`**
- World-frame mm canvas (horizontal = world Y, vertical = world Z).
- Per grasp: **hold-average** map = mean of frames whose taxel sum ≥ 50% of
  that grasp's peak sum (project convention).
- Each taxel painted as a pitch-sized block at its true world position;
  overlapping cells **averaged**.
- **overlap σ** = mean per-cell standard deviation where ≥2 grasps overlap.
  Poses are ground truth, so the *correct* geometry minimises σ. This is our
  standing quality alarm — it is what caught the flipped row convention.
- `export_pair()` → `training_pair.npz`
  (INPUT = centre grasp, TARGET = full stitch, + temporal snapshots).
- `calibrate` mode: tries all 4 flip combinations, ranks by σ.

**How stitching works, in one paragraph** (useful for explaining to others):
divide the swept area into small real-world rectangles (canvas cells of
`res_mm`); every taxel of every grasp is placed at the world position it
physically occupied; where several taxels land in the same cell, average
them. That's it. INPUT and TARGET are painted on the *same* grid, so they are
pixel-aligned by construction, independent of any calibration error.

---

## 3. KEY NUMBERS & CONVENTIONS — ALL MEASURED, TRUST THESE

### 3.1 Calibration chain (re-verified 2026-07-23)

| Quantity | Value | How it was established |
|---|---|---|
| `TOOL_OFFSET_Z` (Ø26 mm) | **0.15651 m** | Calibrate run, `measured_live_pad` |
| `PAD_CENTER_ABOVE_CASE_M` | **0.0221 m** | **diag13**, read from the extension's own mesh log (22.10 mm) |
| EE → Case-origin, Z | **−134.42 mm** | Live Case prim read, reproduced in every probe |
| Pad centre | Case Z − **22.10 mm** | diag13 |
| EE ↕ pad centre | **156.51 mm** | 134.42 + 22.10 |
| Placement accuracy | **+0.13 mm** | Live Case read vs GUI target, two independent runs |

### 3.2 Scene geometry

- Cylinder: d = 26 mm, L = 140 mm, centre world ≈ (−268.06, 199.0, 1052.2) mm.
- Rod top (upright) = **1122.2 mm**.
- Palm: `base_link` origin = flange − 10.79 mm = **top** of the black housing.
  Housing **bottom** face = origin − 75.9 mm (`PALM_HOUSING_DROP_MM = 75.9`).
  The GUI's brown "palm bottom" line + clearance annotation is the safety
  check — trust it, it is EE-anchored and updates with calibration.
- Robot base_link world ≈ (20.93, −337.5, 992.75) mm.

### 3.3 Pad / taxel geometry

- Stitcher currently assumes pad face **22 mm (4 cols, world Y) × 37 mm
  (7 rows, world Z)** → pitch **5.50 / 5.286 mm**. Never transpose.
- **Measured (diag13) deformable body**: 26.05 × 3.08 × **41.08** mm, with an
  18 × 12 = 216-node sensing grid feeding Berith's CNN (which outputs 7×4).
- **OPEN QUESTION** (§9.1): the 41 × 26 figure is the *soft layer*; the
  nominal sensor (Berith's paper) is 22 × 37, i.e. exactly 2 mm smaller per
  side. Does the 7×4 output span 37×22 or 41×26? Keep 37×22 until Berith
  confirms — the ladder zero-crossing (§5.9) supports 37.

### 3.4 Array conventions (PROVEN — do not re-litigate)

- **Columns:** CSV column index **increases toward world −Y**.
  Both sensors share ONE convention — there is **no mirror** between s1 and s2
  anywhere in the pipeline.
- **Rows:** CSV **row 0 = physical BOTTOM** of the pad (−Z), row 6 = top.
  Proven four independent ways (§6.2).
- The single documented home of the axis convention is
  `_taxel_centers()` in `viz/stitching.py`. `CAL` stays **all-False**,
  meaning "no deviation from the shared convention". Never encode a
  convention fix in `CAL`.
- **Display:** all per-grasp heatmaps now use `origin="lower"` so that
  image-up = world-up. PNGs generated *before* 2026-07-22 are vertically
  mirrored — do not compare old and new screenshots by eye.

### 3.5 σ baselines (regression alarms)

| Run type | res_mm | σ (s1 / s2) |
|---|---|---|
| Vertical rod, 2×3, z=+40, step 8 | 1.0 | **107 / 100** (historical 108 / 101) |
| Same run | 0.75 | **88 / 82** |
| Tilted 20°, 3×1, step 6 | 1.0 | **111 / 131** |
| Census (half pad off rod), 1×5 | 1.0 | 72 / 70 |

- Correct geometry ≈ 31–32% of overlap signal. **Alarm if σ heads toward
  70% of signal** (that is what a broken convention looks like: the flipped
  rows read 148 / 161).
- **Use `res_mm = 0.75` as the standard going forward.** At 1.0 the 5.286 mm
  row pitch rounds to 5 whole cells, leaving a 1-cell seam that becomes a
  visible dark line on single-height grids. 0.75 → 5.286/0.75 ≈ 7.05 cells,
  seam gone.

### 3.6 Operational

- Always prefix cuRobo runs with `TORCH_CUDA_ARCH_LIST="7.5"` to avoid JIT
  rebuild hangs.
- Gripper closes **blind** to a fixed 0.55 rad. Tactile data is used only as a
  *gate* afterwards (calibration refuses to store if peak sum < 1000 = the
  pads closed on air). Nothing stops on contact. **This becomes a blocker for
  multiple diameters — see §9.4.**

---

## 4. HARD RULES (binding, carried from v1–v5, all still true)

1. **Kourosh's visual read of the sim beats any diagnostic.** Every
   disagreement so far, he was right. (Twice more this chat — see §6.4.)
2. **When a diagnostic and independent evidence disagree, interrogate the
   diagnostic.** Two diagnostics produced confident wrong verdicts this chat.
3. PhysX moves links in **Fabric**; the USD stage keeps authored transforms →
   read poses via cuRobo FK or the live Case prim, **never**
   `UsdGeom.XformCache` on articulation links.
4. The TSF sensor prim is an empty Xform; the real deformable body is one
   level deeper (the `Case` prim; `TSF_*_CASE_is_live = true`).
5. Never attach `SingleRigidPrim` to live links (this once corrupted PhysX and
   exploded the close).
6. Never grasp the upright rod at z = 0 centre — the housing bottom strikes
   the rod top. (The GUI palm annotation shows this.)
7. Judge placement only from **normal** runs, never from calibrate runs (the
   calibrate grasp deliberately uses a provisional 0.15 m offset, so its
   "landing error" is always ≈ 150 − TOOL_OFFSET_Z mm and means nothing).
8. **Measure, don't guess.** Pin thresholds and conventions with probes before
   coding around them.
9. **NEW — Prefer reading the source over inferring from contact patterns.**
   The sensor's geometry is authored in `sensor_config.json` and the mesh
   logs. Contact-pattern inference is unreliable in partial-contact regimes.

---

## 5. FULL HISTORY — WHAT HAPPENED, IN ORDER

### 5.1 Earlier chats (v1–v5 condensed)

- **June:** Paper 2 finalised for submission (CRediT, highlights, cover
  letter, corrected PDF). Paper 3 planned as a synthesis of Papers 1–2 plus
  Berith's sim pipeline.
- **Late June / early July (Block 2 build):** `viz/stitching.py` created with
  world-frame canvas, outlier rejection, overlap-σ metric; fixed frozen
  `pad_actual_pos_m` via FK; fixed cylinder tilt application; added
  signed/anchored grid + Save/Load Experiment to the GUI.
- **Early–mid July (pad-pose hunt #1):** pads landed ~25–30 mm too high.
  Robot arm verified accurate to sub-mm; error traced to an overestimated
  `TOOL_OFFSET_Z`. Fixed a bookkeeping bug (`pad_desired_pos_m` was anchored
  to the home pose instead of the object centre). Discovered the Fabric-vs-USD
  pose problem and the empty-Xform sensor prim.
- **v5 chat (2026-07-21/22):**
  - 2×3 verification grid PASSED 6/6 (±0.03 mm landings, symmetry 0.015 mm,
    orientation 0.00°, no object motion, peaks ~12.5k / ~4.6k edge).
  - `pt00` descent shortfall (−1.1 mm) fixed with the **FINAL TRIM** block in
    `grasp_one_point` (FK the residual, close it with one 3-step stitched
    line). Now lands +0.03 mm.
  - **s2 display mirror eliminated** — `diag7` proved both sensors report the
    same hot column, so the CSVs already share one convention.
    `MIRROR_S2 = False` everywhere.
  - GUI palm line fixed to the housing **bottom** (−10.79 − 75.9 mm).
  - **Stitcher column-axis bug fixed**: `ys` reversed in `_taxel_centers`
    (col 0 → +Y). σ fell 248 → 108.
  - `flip_ud` left undecided — vertical-rod data cannot separate it.

### 5.2 This chat, step by step (2026-07-22 → 2026-07-23)

**(a) Option A tilted run** — `run_20260722_141424`, 20° about X, y=−15,
z=+42, 3×1, step 6 mm. Machinery flawless (landings +0.03 mm, orientation
0.00°, palm clearance +117.8 mm, object motion ≤0.03 mm). But **no visible
tilt in the stitch**, and σ = 148 / 161 (vs 108 / 101 baseline).

**(b) Why the tilt was invisible.** 6 mm of climb only shifts contact 2.18 mm
in Y (0.4 of a column), so the diagonal had to come from the *within-grasp*
bands lining up across grasps. They did not → suspicion fell on the row
convention.

**(c) Three judges convict `flip_ud`:**
1. `diag8_row_slope.py` — per-row column centroid slope: **6 of 6 maps
   positive** (consistent sign, weak magnitude).
2. Stitcher `calibrate` mode — `flip_ud=True` wins by **37 σ** (s1
   148.3 → 111.1) and **30 σ** (s2 160.5 → 130.5), same winner both sensors.
3. Census run (below) — rows died where a flipped convention predicts.

**(d) The fix** — one line in `_taxel_centers`, mirroring the earlier `ys`
fix, **not** via `CAL`:
```python
zs = (np.arange(N_ROWS) - (N_ROWS - 1) / 2.0) * PITCH_Z   # row 0 = BOTTOM
```
Re-running `calibrate` then showed all-False winning with **identical σ
values, labels swapped** (111.1 / 130.5) — the same equivalence proof the
column fix produced. Re-stitch: σ = 111 / 131 (tilted baseline).

**(e) The "dome" discovery.** In full contact the pad's row sums are *not*
uniform — the middle row reads ~2× the ends. The pad presses hardest in the
middle regardless of contact. This means you cannot read "hot = object here"
off a raw heatmap.

**(f) Census run** — `run_20260722_162052`, tilt 0, y=−10…+6, z=+70, 1×5,
step 4 (half the pad hanging above the rod top). `pt00` was skipped:
reachability gate fired with `cond(J) = 3712` because the direct IK returned a
**flipped arm branch** (shoulder_lift −3.52 rad). **Patched** — see §7.

**(g) `diag9_rim_mask.py`** — divides the census row profile by a
full-contact reference profile, cancelling the dome, leaving a 0–1 "was this
row backed?" mask. It gave a beautifully clean step and concluded **"pad sits
10.6 mm low"**.

**(h) Acting on diag9 — the wrong turn.** `PAD_CENTER_ABOVE_CASE_M` was
changed 0.0223 → **0.0329**; recalibration gave `TOOL_OFFSET_Z = 0.16731`.
This actually put the pad **10.8 mm too high**.

**(i) Verification census** — `run_20260723_141636`. The mask came back
**self-contradictory** (0.85, 1.33, 0.82, 0.55, 0.55, 0.24, 0.06) — no single
rim position can produce two 0.55 rows and a 1.33 spike. Diagnosis: at ~50%
backing the dome assumption **collapses**, because the inner finger tips
under an off-centre load and the whole pressure distribution reshapes.
**diag9 is only valid at ≳70% backing.**

**(j) `diag10_ladder.py`** — assumption-free idea: total signal ∝ backed
area, so climbing z must drive signal linearly to zero exactly when the pad
bottom clears the rim. Ladder run `run_20260723_153318` (z = 70…90, step 4,
y = 0) showed contact dying at z ≈ 76–79 instead of 88.5 → **"pad 12 mm
high"**, contradicting diag9's "6 mm low". Two inferences, two answers.

**(k) The pivot — stop inferring, read the source.** Kourosh asked whether the
exact pad-pose reference point could be found in Berith's code. It could:
- `TSF_85_Ext/data/sensor_config.json` — grid 18 × 12, with explicit
  `node_ids` mapping each sensing node to a mesh node.
- `gui_s1_mesh_state.csv` — long format (one row per node per frame) with
  node positions **in the case-local frame**, plus the case pose.

**(l) `diag13_sensing_geometry.py` — GROUND TRUTH:**
```
ALL mesh nodes (deformable pad body):  X=26.05  Y=3.08  Z=41.08 mm
SENSING nodes (18x12 grid):            same extents, 216 nodes
CASE ORIGIN -> sensing-array centre  =  +22.10 mm
grid row 0 sits at the POSITIVE end of the long axis
mean row spacing = -2.417 mm
```
- **22.10 mm confirms the ORIGINAL 0.0223** to within 0.2 mm. The 0.0329 was
  wrong.
- Case-local +Z maps to world **down**, and grid row 0 is at the +Z end →
  **row 0 is the physically lowest row**. A **fourth**, source-level
  confirmation of the row convention.

**(m) Revert + re-verify.** `PAD_CENTER_ABOVE_CASE_M = 0.0221`; recalibration
gave `TOOL_OFFSET_Z = 0.15651` — only **0.2 mm** from the original 0.15671.
**The entire excursion was a round trip; no run before 2026-07-23 was ever
mis-placed.**

**(n) Ladder re-run** — `run_20260723_171434`. Air-subtracted peaks 6394,
5096, 4495, 1867, 1070 → linear zero-crossing at **z ≈ 88.9** vs geometric
prediction **88.5**. Live Case read: pad centre **1122.33** vs target
**1122.20** = **+0.13 mm**. Placement confirmed by two independent methods.

**(o) `diag7` on that ladder — the answer to the original "bleeding"
question.** As the rim recedes 0 → −16 mm, the contact's top edge in the
tactile map moves only ~42% as far, and the peak row stays stuck at row 1.
**For partial / edge contact the tactile map is not a spatially faithful
image of the contact patch.** Most plausible cause: Berith's CNN is being run
out of its training distribution (49 indenters pressed *onto* the sensor
face, never crossing its edge), and its fully-connected output layer has no
obligation to preserve locality. This is *not* a placement bug and *not*
simple blur.

**(p) Confirmation run** — `run_20260723_183658`, 2×3, z=+40, step 8, y = 0 /
+8 / +16 (same design as the historical baseline):
- Peaks 12930 / 12741 / 13182 / 12844 (pad over rod) and 4700 / 4664 (pad at
  y=+16, hanging off the rod's side) — matches the historical
  "~12.5k centred / ~4.6k edge pair".
- Placement +0.13 mm; orientation error 0.02°; object motion 0;
  reachability 6/6; trim not needed.
- Bonus re-confirmation of the **column** convention: the contact stripe
  images the rod's tangent line at world Y ≈ 199 regardless of pad Y, so it
  marches toward higher column index as the pad steps to +8 and +16 — exactly
  as observed.
- **σ = 107.0 / 100.3 at 1.0 mm/cell** vs historical 108 / 101. Baseline
  reproduced. **σ = 88 / 82 at 0.75 mm/cell.**
- `training_pair.npz` exported. **This is the first fully verified
  training-pair source.**

---

## 6. THE MEASUREMENT CHAIN — how each fact was proven

### 6.1 Placement (pad lands where the GUI says)
1. Live Case prim read at closure, minus the measured 22.10 mm → **+0.13 mm**
   from target, in two separate runs.
2. Ladder zero-crossing: **88.9** observed vs **88.5** predicted.
3. Negative control: at the top of the ladder the pad reads pure air; if
   placement were ~10 mm low, the bottom rows would still be loaded.

### 6.2 Row convention (CSV row 0 = physical bottom) — four judges
1. `calibrate` σ margin: 37 (s1) / 30 (s2) σ units, both sensors agreeing.
2. `diag8` slope sign: 6 of 6 maps.
3. Census: the rows that died were the ones a flipped convention predicts.
4. `diag13`: grid row 0 sits at the case-local +Z end, which maps to world
   down. Read from the sensor model itself.

### 6.3 Column convention (index increases toward −Y)
1. `diag7` physical Y-sweep (v5 chat), both sensors.
2. Re-confirmed by the 2×3 confirmation run's stripe march (§5.2p).

### 6.4 Two diagnostics that lied (keep this in mind)
- **diag9** (dome-cancelling mask) — valid only at ≳70% backing; at 50% it
  produced a confident, wrong 10.6 mm verdict that cost a full day.
- **diag10** (total-vs-z intercept) — sound in principle, but heavily
  confounded when the finger tips in extreme partial contact.
- In both cases the *cross-checks* and Kourosh's insistence that something
  did not add up are what saved the calibration.

---

## 7. FILES TOUCHED THIS CHAT

| File | Change |
|---|---|
| `viz/stitching.py` | `zs` line reversed in `_taxel_centers` → row 0 = bottom. `CAL` stays all-False. |
| `viz/heatmaps.py` | `origin="lower"` on the imshow. |
| `viz/temporal_snapshots.py` | `origin="lower"` on both imshow calls. |
| `sim/collect_from_config.py` | `PAD_CENTER_ABOVE_CASE_M` → **0.0221** (0.0293 ghost deleted). Reachability **branch-retry** patch inside `precheck_reachability`. |

**The reachability patch** (goes right after
`res = evaluate_reachability(q_seed, q_goal, MANUAL_LIMITS)`): when the gate
fails, redo the IK the way the real run actually moves (free-move to
grasp + `APPROACH_H`, then stitched descent) and re-test. Fixes false
"unreachable" verdicts caused by the direct IK returning a flipped arm
branch. Verified: `pt00` now passes.

### Diagnostic toolbox (and what each is valid for)

| Tool | Purpose | Validity |
|---|---|---|
| `diag7_show_contact.py` | Raw 7×4 hold-average dump, hot row/col | Always valid, model-free |
| `diag8_row_slope.py` | Per-row column centroid slope on tilted rod | Weak signal — corroboration only |
| `diag9_rim_mask.py` | Contact mask via census ÷ full-contact reference | **Only ≳70% backing** |
| `diag10_ladder.py` | Zero-contact intercept from total-vs-z | Good for the zero-crossing; confounded mid-range |
| `diag13_sensing_geometry.py` | Sensor geometry from the extension's mesh log | **Definitive** — no physics, no contact |

(`diag11` — static USD probe — failed: `pxr` is not exposed to bare
`python.sh` in Isaac 5.1. `diag12` assumed the wide log format and was
superseded by `diag13`.)

---

## 8. CURRENT STATE → what to do next

**Everything about the data factory is verified and working.** Placement
0.13 mm, conventions pinned, σ baselines reproduced, first verified training
pair exported.

**The immediate next step is to start producing real dataset runs** — i.e.
scale up grids on the Ø26 cylinder, then add diameters and tilts (§9.8).

Before large-scale generation, three cheap items are worth closing:
`WAIT_HOLD_SECONDS` (§9.6) and the `SUBTRACT_BASELINE` decision (§9.5),
because both change what gets baked into every training pair — and the
closure question (§9.4) before any second diameter is attempted.

### 8.1 Tilt-visibility test — runnable with the CURRENT GUI, no code changes

Goal: finally *see* the 20° tilt as a slanted band in the stitched map. The
earlier tilted runs failed for two different reasons: the first had too much
Y drift (contact slid off the pad), the second (Option A) had too little
(4.4 mm total < one column). The fix is to size the sweep to the drift budget
and **pre-compensate the Y anchor**, both of which the current GUI can do.

Geometry: at 20° about X the rod centreline drifts `dY/dZ = −tan20 = −0.364`.
The contact stripe is ~2 columns (~11 mm) wide on a 22 mm pad, so the stripe
centre can travel about **±5.5 mm** before clipping → a usable Z climb of
~30 mm.

**Config:** tilt = 20°, axis X · pad **y = −9.0** · pad **z = +10** ·
grid **nx = 6, ny = 1** · step **6 mm**.
(The −9.0 anchors the sweep so the contact starts at +5.4 mm on the pad and
ends at −5.6 mm, staying on-pad throughout. `nx` is the Z axis in this GUI.)

**Pre-flight:** Update Preview (6 dots in a vertical column inside the tilted
rod outline) → Check Reachability (expect 6/6) → palm annotation green
(clearance ≈ 40 mm at the top point).

**Pre-registered prediction:** the stitched band's centre should walk
Y ≈ 195.4 → 184.4 mm as Z goes 1062.2 → 1092.2, i.e. a straight slanted band
with slope ≈ −0.36 (= 20° from vertical). σ expected in the 110–130 range.
If instead the band is vertical or zig-zags, stop — something regressed.

**Free first:** the **old tilted 1×5 run (2026-07-22, 8 mm steps)** has the
largest Y drift of any tilted run (~11.6 mm) and was collected at the correct
calibration epoch, but was **only ever stitched with the flipped row
convention**. Re-stitch it with the current code before running anything new:
```
python3 viz/stitching.py <that_run_dir> 0.75
```
The diagonal may already be there.

---

## 9. OPEN LEDGER — everything deferred, in priority order

**9.1 — Taxel pitch question (ask Berith).**
Does the 7×4 tactile output represent the nominal **37 × 22 mm** PCB area or
the **41.08 × 26.05 mm** deformable layer measured by diag13? Stitching
currently assumes 37 × 22 (pitch 5.286 / 5.50). If it is 41 × 26, the pitches
become 5.869 / 6.514 and every stitched map is ~10% compressed toward its
centre. *Note: the row-death method cannot answer this, because edge contact
is not spatially faithful (§9.3).*

**9.2 — Bug report to Berith.** `gui_s2_mesh_state.csv` uses `s1_` column
names in its header. Harmless for us (pads are symmetric) but wrong.

**9.3 — Edge-contact fidelity (ask Berith).** For contacts crossing the
sensor edge, the response tracks the true contact boundary at only ~42%, and
extends ~1–1.5 rows past it. Were edge-crossing contacts in the CNN's
training distribution? Is the real TSF-85 equally blurry, or sharper? This
bounds what the U-Net can learn and belongs in the paper's limitations.

**9.4 — CONTACT-AWARE / FORCE-NORMALISED CLOSURE (blocks multi-diameter
scale-up).**
Today the gripper closes **blind** to a fixed `CLOSE_RAD = 0.55` rad. At that
angle the pad-to-pad gap is a fixed distance G, so the compression a grasp
applies is `(D − G)/2` per pad — i.e. **it depends entirely on the object
diameter**:
- D ≈ 26 mm (current rod): fingers reach 0.55 exactly, compression is right.
- D noticeably larger: fingers stall before 0.55 → very high force, saturated
  tactile maps.
- D smaller than G: **no contact at all**, and the calibration contact gate
  will (correctly) refuse to store.

Consequences if left unfixed:
- Every new diameter needs a hand-tuned `CLOSE_RAD` as well as its own
  `TOOL_OFFSET_Z` — trial and error, easy to get subtly wrong.
- Tactile magnitudes would not be comparable across the dataset, so the U-Net
  could learn diameter-specific intensity artifacts instead of geometry.
- It diverges from the real robot (Papers 1–2 close to contact/force), which
  hurts sim-to-real.

Proposed fix: close incrementally and **stop on a tactile criterion** (e.g.
taxel sum crossing a target, or a fixed increment past first contact), rather
than on a joint angle. The machinery already exists — the collector reads
tactile CSVs live and `ramp_gripper()` is already frame-by-frame — so this is
a modification to `ramp_gripper` / `grasp_one_point`, not new infrastructure.
Decide *before* generating the multi-diameter dataset; changing it afterwards
invalidates every earlier run's force scale.

**9.5 — `SUBTRACT_BASELINE` decision** for `export_pair` (currently `False`).
ON removes the pad-locked fixed pattern that otherwise smears across
multi-grasp targets — likely ON for training export. Verify on the verified
pair from `run_20260723_183658`.

**9.6 — Temporal snapshot #4. [RESOLVED 2026-07-27]** Was:
`WAIT_HOLD_SECONDS = 1.0`, so the +3s snapshot fell AFTER release and read
air (~270 sum every grasp — looked "repeated"). `temporal_snapshots.py` was
correct (it flagged `post3s_valid=False` and fell back to the last frame);
the hold was simply too short for 3 s of held-closed data to exist in the
CSV. FIX: `collect_from_config.py` line 282 → `WAIT_HOLD_SECONDS = 3.5`.
VERIFIED on the 1×6 vertical run: +3s sums jumped to ~95% levels (pt00 275→
6181, pt01 275→7371, pt02 276→7406), the "needs 3s hold" label is gone, and
each grasp's +3s column now shows distinct real contact. The sequence now
reads physically: pressure rises through the squeeze then holds/slightly
relaxes (soft-pad creep) — the Papers 1–2 / CASE-2021 convention working.
COST: +2.5 s hold per grasp (~15 s on a 6-grasp run; scales with dataset
size). EDGE-CASE noted: on the top grasp (pt05) the barely-backed sensor s2
crept to air by 3 s (sum 267) while s1 held — a marginal edge grasp slipping
under the longer hold, NOT a bug; further evidence for contact-aware closure
(§9.4).

**9.7 — Grid-tilt handling.** Option A (short world-aligned segments) vs
Option B (grid follows the tilt). For long tilted sweeps the GUI needs an
optional per-point Y tracking feature (≈ −2.9 mm Y per 8 mm Z at 20°); it
currently cannot do per-point offsets. **Not needed for short sweeps** — see
§8.1 for a tilt-visibility test the current GUI can already run.

**9.8 — Dataset policy (now evidence-based).** Build training pairs from
grasps where the pad is **fully backed**. Treat edge / partial-contact grasps
as a separate, flagged category — they are physically interesting (they carry
a distinctive edge signature) but not spatially faithful.

**9.9 — Scale up.** Bigger grids → multiple diameters (**each diameter needs
its own calibrate run**, and see §9.4) → tilt and diameter variations for the
full dataset.

**9.10 — Per-point live Case recording.** Record
`0.5*(right+left)` Case pose during the closed hold for *every* point into
`pose_history.json`, and add it as candidate #0 in `load_offsets()`. This
replaces the "constant finger swing" assumption with a direct measurement at
the moment of contact — and would have caught this whole episode on day one.
Natural companion to the per-diameter 3D offset vector architecture.

**9.11 — Block 3.** Two U-Nets trained on `training_pair.npz` pairs.

---

## 10. RUNS REGISTRY

| Run | Config | Status / use |
|---|---|---|
| `run_20260721_141246` | 2×3, z=+40, step 8 | Pre-trim (pt00 −1.1 mm); used for the diag7 column proof |
| `run_20260721_152151` | 2×3, z=+40, step 8 | Clean baseline σ=108/101 @1.0; **reference run for diag9** |
| tilted 1×5 (2026-07-22) | 20°, fixed-Y column | Design flaw (contact slid off-pad); machinery valid; do not train on |
| `run_20260722_141424` | 20° tilt, 3×1, y=−15, z=+42, step 6 | **flip_ud verdict data.** σ 148/161 → 111/131 after the fix |
| `run_20260722_162052` | 1×5, y=−10…+6, z=+70, step 4 | Census. pt00 skipped (cond J). Source of the (wrong) diag9 verdict |
| `run_20260723_141636` | census repeat | Ran at 0.0329 → pad **10.8 mm high**. **Do not use** |
| `run_20260723_153318` | ladder 1×6, z=70…90 | Ran at 0.0329. Placement data invalid, **but its `mesh_state.csv` gave diag13 the ground truth** |
| `run_20260723_171434` | ladder 1×6, z=70…90, y=0 | At 0.0221. **Placement confirmed** (zero at 88.9) |
| `run_20260723_183658` | 2×3, z=+40, step 8 | **Confirmation run.** σ=107/100 @1.0, 88/82 @0.75. **First verified training-pair source** |

**Calibration epochs** (which runs sat where):
- Everything up to and including `run_20260722_162052`: `TOOL_OFFSET_Z =
  0.15671` → correct within 0.2 mm.
- `run_20260723_141636` and `run_20260723_153318`: `0.16731` → **10.8 mm
  high**.
- `run_20260723_171434` onward: `0.15651` → correct.

---

## 11. CHECKLIST — EVERYTHING DONE SO FAR

**Motion & control**
- [x] Proven single-grasp routine (approach → descend → close → record → open)
- [x] Pad-to-pad straight-line moves between grid points (no lift, no jerk)
- [x] Orientation anchoring to fixed `tq` (killed compounding tilt drift)
- [x] Symmetric gripper close (both 4-bar linkages driven) — killed the 72×
      s1/s2 asymmetry
- [x] FINAL TRIM after descent (pt00 lands +0.03 mm)
- [x] Reachability pre-check ported from Paper 2 (IK + limit gates + cond(J))
- [x] Reachability branch-retry patch (fixes false "unreachable")
- [x] Physics-explosion / NaN watchdog
- [x] Object bolted with a centre-preserving fixed joint; tilt applied from
      the GUI config

**Calibration & geometry**
- [x] Per-diameter calibration store, refuses to run uncalibrated objects
- [x] Live Case prim identified as the true pad reference
- [x] Contact gate (refuses to store a calibration from a grasp on air)
- [x] `PAD_CENTER_ABOVE_CASE_M` **measured** from the sensor model (22.10 mm)
- [x] `TOOL_OFFSET_Z` = 0.15651 for Ø26, placement verified to 0.13 mm
- [x] Palm geometry pinned (housing bottom = EE − 86.7 mm) + GUI clearance
      annotation

**Conventions**
- [x] Column convention proven (index → −Y), both sensors
- [x] s2 display mirror eliminated everywhere
- [x] Row convention proven (row 0 = bottom) — four independent judges
- [x] Display orientation fixed (`origin="lower"`)
- [x] `CAL` semantics settled: all-False = "no deviation"; conventions live
      only in `_taxel_centers`

**Stitching (Block 2)**
- [x] World-frame canvas, hold-average, fixed-size taxel splats
- [x] Overlap averaging + overlap-σ quality metric
- [x] Outlier rejection, degenerate-source detection, GUI-frame re-anchoring
- [x] Overlay column showing the swept shape on the object
- [x] `calibrate` mode (4 flip combos ranked by σ)
- [x] `export_pair()` → `training_pair.npz`, pixel-aligned by construction
- [x] Seam artifact understood and avoided (`res_mm = 0.75`)
- [x] σ baselines established and reproduced after recalibration

**GUI cockpit**
- [x] Object pose + tilt, pad offset, signed/anchored grid, live 3-panel preview
- [x] Save/Load Experiment recipes
- [x] Reachability colouring, heatmaps, pose history, verification plots,
      temporal snapshots, pad-truth check
- [x] Calibrate tab with measured-pad overlay
- [x] Stitching tab

**Diagnostics built**
- [x] diag7 (raw contact), diag8 (row slope), diag9 (rim mask),
      diag10 (ladder), diag13 (sensor geometry from the mesh log)

**Understanding gained**
- [x] The pad's intrinsic pressure "dome"
- [x] The partial-contact regime change (finger tipping)
- [x] Edge contact is not spatially faithful (CNN out-of-distribution)
- [x] Why totals ≠ area in partial contact

---

## 12. CHECKLIST — EVERYTHING NOT DONE YET

**Questions for Berith / Vincent**
- [ ] Taxel pitch: 37×22 or 41×26? (§9.1)
- [ ] Edge-crossing contacts in the CNN training set? Real-sensor blur? (§9.3)
- [ ] `s2` mesh log column-name bug (§9.2)

**Before dataset generation**
- [ ] **Contact-aware / force-normalised closure decision (§9.4)** — must be
      settled before any second diameter
- [x] `WAIT_HOLD_SECONDS` 1.0 → **3.5** for temporal snapshot #4 — DONE &
      verified 2026-07-27 (§9.6)
- [ ] `SUBTRACT_BASELINE` decision, verified on a real pair (§9.5)
- [ ] Written dataset policy: full-backing vs edge grasps (§9.8)
- [ ] Tilt-visibility test (§8.1) — re-stitch the old 1×5, then the 6×1 sweep

**Data factory features**
- [ ] Per-point Y tracking in the GUI for long tilted sweeps (§9.7)
- [ ] Per-point live Case pose recording + `load_offsets` candidate #0 (§9.10)
- [ ] Calibration as a 3D offset vector per diameter (architecture designed,
      not implemented)

**Dataset**
- [ ] Large grids on Ø26
- [ ] Additional diameters (calibrate each)
- [ ] Tilt variations
- [ ] Dataset assembly / storage format for many `training_pair.npz`

**Model & paper**
- [ ] Block 3: two U-Nets, training loop, validation split
- [ ] Block 4: evaluation with Paper 1/2 metrics (TC error, GSR error, SSIM,
      Safe Zone); comparison against the Papers 1–2 classifier method
- [ ] Block 5: paper writing, including the simulated-sensor fidelity
      limitations section

---

## 13. STANDING PROMPTS FOR THE NEXT CLAUDE

- Trust the registry numbers in §3. **Do not re-derive solved calibration.**
- Ask for run artifacts (`pose_history.json`, `pad_truth_probe.json`,
  `run_progress.log`, heatmaps, stitched PNGs) before concluding anything.
- When Kourosh's eyes and a diagnostic disagree, **the diagnostic is wrong.**
- When two measurements disagree, do not average them and do not pick the
  more convenient one — find the assumption that breaks, or go read the
  source.
- Prefer reading authored data (`sensor_config.json`, mesh logs, code) over
  inferring geometry from contact patterns.
- Keep answers concrete: exact file, exact function, exact line, predicted
  numbers, pass/fail criteria stated **before** the run.
- Before proposing a code change, state what measurement would falsify it.

---

*End of Handoff v6 core. §14 below is the v6.1 addendum — the newest state.*

---

## 14. v6.1 ADDENDUM (2026-07-24 → 27) — READ THIS LAST, IT'S NEWEST

### 14.1 The tilt is now VISIBLE in a stitched map (goal reached)

Two tilted sweeps closed the "can we see the tilt?" question:

- **`run_20260724_162921`** — 20° about X, pad y=−9, z=+10, grid 6×1 (nx=6
  is the Z axis), step 6 mm. The Y-anchor pre-compensation (y=−9) kept contact
  on-pad across the whole climb. **Result:** the hot column marches
  monotonically **1 → 3** across pt00→pt05 (col index increases toward −Y =
  the rod's top leaning −Y = the 20° tilt), and the stitched band **leans**.
  First run where the stitch carries the tilt. BUT pt00 (deepest grasp) came
  out slightly compromised: orientation error 0.45° (others ~0.00°), fingers
  closed unevenly (finger_joint stalled at 0.531, not 0.55), descent trim
  fired 4.4 mm, and the rod was nudged ~0.1 mm. → treat as demo, not training.

- **`run_20260724_17xxxx` (z=18 start)** — same as above but pad z=+18, which
  drops the deep pt00 grasp. **pt00 orientation error 0.00°, object motion
  0.02 mm, all six grasps clean, hot col still marches 1→3.** This is the
  **first training-grade tilted set.**

**How to read tilt in the data (correction to an earlier mistake):** the hot
column moves **LEFT → RIGHT (col 1 → col 3)** as the pad climbs. In the
individual heatmaps the bright vertical stripe walks sideways ~one column
every two grasps; in the stitch those offset stripes line up into one slanted
band. (An earlier chat message said right-to-left — that was wrong; the col
sums prove 1→3.)

**Two effects that make a tilted stitch look "hotter at one end" — both
benign, both already measured:** (1) the pad's intrinsic **dome** (middle row
presses ~2× the ends even in full contact); (2) **edge/partial-contact
non-fidelity** (the CNN runs out of its training distribution past the sensor
rim, so contact there is not spatially faithful). Neither is a placement or
stitching bug. See §5.2(e) and §5.2(o).

### 14.2 NEW FEATURE — stitch round-trip validation (`viz/validation.py`)

**Idea (Kourosh's):** we know each grasp's original tactile map AND the exact
world pose it was taken at. So: stitch → then SAMPLE THE CANVAS BACK at each
grasp's 28 taxel world-positions → recover a 7×4 → compare recovered vs
original with the Paper 1/2 metrics.

**What it validates:** the STITCHER AS A CONTAINER — how much resampling +
overlap-averaging distort a single grasp. **It is NOT model completion.**
High SSIM / low TC error is expected, especially for the center grasp (whose
INPUT is literally itself). A grasp in heavy overlap differs most, because
averaging blends it with neighbours — i.e. this is the per-grasp, metric-space
view of the global `overlap sigma`. **Keep this distinction explicit with
supervisors** (it is exactly the "compared to what?" question JP kept asking).
The real Paper-3 metric use — predicted extended map vs TARGET — is Block 4,
once the model exists.

**Design (important):** `validation.py` **imports `stitching.py`'s own
geometry** (`_taxel_centers`, `build_canvas`, `CAL`, pitches, the block-splat
footprint) and inverts exactly that, so the round-trip can never drift from
the real paint step. Verified: an isolated grasp round-trips at SSIM 1.000 /
TC 0.00 mm at res=0.75 (block pitch divides evenly).

**Metrics implemented:**
- **SSIM** — `skimage.metrics.structural_similarity` (data_range set
  explicitly), with a Wang-window fallback if skimage is absent.
- **TC error (mm)** — distance between the two maps' pressure-weighted
  centroids, scaled by the real taxel pitch (5.50 / 5.286 mm). Standard
  center-of-mass definition; matches Paper 1's "tactile centroid".
- **GSR** — uses the REAL Paper-2 pipeline by importing
  `network_gsr.predict_grasp_success` directly from
  `/home/kourosh/Pipeline_ws/ros2_ws/Python_Modules` (no reimplementation).
  GSR is **grasp-level** (needs s1+s2 together → 56 values), so it is computed
  per grasp, not per sensor.

**Results on the verified baseline `run_20260723_183658` (res 0.75):**
```
 s1: backed pt00-pt03  SSIM 0.92-0.97, TC 0.3-0.9 mm
     edge   pt04-pt05  SSIM ~0.86,     TC 1.9-2.9 mm      (overlap sigma 88)
 s2: similar; SUMMARY SSIM 0.928, TC 0.98 mm              (overlap sigma 82)
 GSR: 100.00% original AND recovered on every grasp -> GSR_err 0.00
```
**Reading:** SSIM+TC are meaningful and independently re-find the backed-vs-
edge boundary (pt04–05 are the pad-hanging-off-the-side grasps). **GSR is
saturated** — firm Ø26 cylinder grasps all read 100% stable, so GSR currently
has NO discriminating power on this data. GSR_err = 0 is a real (easy) pass
("stitching preserves stability exactly"), but GSR only becomes informative
near grasp failure (partial contact, low force). Revisit once contact-aware
closure (§9.4) gives force variation.

**GSR caveat to raise with Berith/JP (NEW open item):** `network_gsr.py`
subtracts a REAL-sensor baseline (`Free_30_Seconds_Data.csv`) from our
SIMULATED maps. Real vs sim zero-load offsets differ, so absolute sim-GSR has
an unquantified offset. Question: what baseline should GSR use on simulated
tactile data? Treat sim-GSR as indicative, not exact.

**Run it:**
```
python3 ~/Paper3_Simulation/viz/validation.py <run_dir> 0.75
```
Outputs `<run>/Stitched/validation_report.txt` and `validation_metrics.json`.
GSR needs TensorFlow (it loads in Kourosh's normal python, with harmless
cuFFT/cuDNN/GPU warnings); if TF or the model or the baseline CSV is missing,
GSR prints one "disabled" line and SSIM+TC still run.

### 14.3 VALIDATION — status
- [x] **"Validate Stitch (SSIM/TC/GSR)"** button wired into the Stitching tab
      of `main_gui.py` — DONE 2026-07-27. Runs `validation.validate_and_save`
      in a background thread (GSR/TF load can take seconds), pops a scrollable
      report window, saves `validation_report.txt`. "include GSR" checkbox
      (default off). Verified by syntax + wiring checks; run the GUI from
      normal python (GSR loads there).
- [ ] Decide `SUBTRACT_BASELINE` for `export_pair` and re-verify a pair
      (still §9.5).

### 14.4 TUESDAY MEETING PLAN (supervisor sync)

**What they asked for (from the 2026-07-21 meeting):** cylinder working →
grasp at different places → move in one direction then another → express the
imprints in the OBJECT/inertial frame → try stitching. JP's key warning:
sliding around a curved object re-images the same tangent line, so a naive
stitch gives a line/cylinder, not the object — placement must be in a fixed
object frame. Side ideas: benchmark vs large image models; mix real+synthetic.
Human note from both supervisors: small verified goals, weekly syncs, don't
build the whole thing before validating, don't stress the timeline.

**You are ahead of that ask.** Demo as three acts, one figure each:
1. **Factory works** — GUI preview + one clean heatmap set + pad-truth numbers
   (placement 0.13 mm, orientation 0.00°). "Verified, not hand-waved."
2. **JP's stitching insight, demonstrated** — two sweeps on the SAME cylinder,
   both stitched in the object frame: a **vertical** sweep → a vertical stripe
   ("you get a line" — his exact point, and correct, because a straight rod
   re-images one tangent); the **tilted z=18** sweep → a **slanted band**
   (geometry changes across the sweep, so the stitch captures it). Same
   machinery. This proves you understood his concern AND that your object-
   frame placement is right.
3. **First training pair + honest limitation** — show one `training_pair.npz`
   (INPUT single grasp → TARGET full stitch), then state the measured limit:
   edge/partial contact isn't spatially faithful, so training pairs come from
   fully-backed grasps. Optionally show the validation table as "stitcher
   fidelity" (SSIM 0.92, backed clean / edge degrades) — labelled as container
   validation, NOT model metrics.

**One cheap run to prep — DONE (2026-07-27):** the clean **vertical 1×6**
sweep (tilt 0, y=0, z=+70 in the verified run `run_20260727_151835`;
`run_20260727_1534xx` after the hold fix) gives the expected **vertical
stripe** at Y≈199 — the "you get a line" half of Act 2. Pair it against the
tilted z=18 run for the line-vs-diagonal contrast. (Note the demo runs used
z=+70, i.e. half the pad off the rod top, which also happens to show the
edge-fidelity limit for Act 3; a fully-backed vertical sweep at z=+10 is
even cleaner if you want the stripe without the edge effect.)

### 14.5 FILES ADDED/CHANGED since v6
| File | State |
|---|---|
| `viz/validation.py` | **NEW** — round-trip validator (SSIM+TC+GSR). Final, tested. |
| `main_gui.py` | **CHANGED** — added "Validate Stitch (SSIM/TC/GSR)" button + "include GSR" checkbox in the Stitching tab; new methods `do_validate`, `_load_validation_module`, `_show_validation_report`; new var `stitch_want_gsr`. |
| `sim/collect_from_config.py` | **CHANGED** — `WAIT_HOLD_SECONDS` 1.0 → **3.5** (§9.6). Still 0.0221, reachability patch in place. |
| `viz/stitching.py` | unchanged since the zs fix (all-False CAL, 0.0221-era) |

Diagnostic toolbox now also includes (from the v6 chat):
`diag13_sensing_geometry.py` (definitive sensor geometry from the mesh log).

### 14.6 IMMEDIATE NEXT STEPS (in order)
1. ~~GUI "Validate Stitch" button~~ — DONE (§14.5).
2. ~~Clean vertical 1×6 run for the Tuesday contrast~~ — DONE (§14.4).
3. ~~`WAIT_HOLD_SECONDS` → 3.5~~ — DONE & verified (§9.6).
4. `SUBTRACT_BASELINE` decision + verify a training pair (§9.5).
5. Contact-aware closure (§9.4) before any second diameter.
6. Then scale up (grids/diameters/tilts) → Block 3 (train the U-Nets).

---

## 15. v6.2 ADDENDUM (2026-07-27, later) — TILT SERIES + BRIGHTNESS CAUSE + RELATED WORK

### 15.1 Full tilt series collected — the Tuesday "geometry-capture" figure

Ran the SAME cylinder at four tilt angles, all clean object-frame stitches at
0.75 mm/cell, 6×1 (5 grasps at 45°), pad Y pre-compensated so contact stays
on-pad:

| tilt | pad y | pad z | σ (s1) | result |
|---|---|---|---|---|
| 0° | 0 | +18/+70 | 77 / 108 | vertical stripe ("you get a line") |
| 20° | −9 | +18 | 111 | gentle lean |
| 35° | −15 | +18 | 108 | clear lean + small off-rod cold wedge |
| 45° | (steeper) | — | 107 | steep lean + LARGE off-rod cold wedge, 1 grasp dropped |

Lined up 0→20→35→45 this is an unmistakable demonstration that the stitch
captures object geometry in the object frame — exactly JP's line-vs-diagonal
concern, answered. **For the headline, anchor on 20° and 35°** (both clean,
full 6 grasps, obviously different); show 45° as the "pushed further" extreme
(marginal — big fraction of each grasp off-rod, one grasp dropped). The
GROWING cold wedge across the series is its own sub-story: as tilt steepens,
more of the pad misses the rod at the top of the sweep, and the map correctly
shows where contact does and doesn't exist.

Pad-Y pre-compensation rule (keeps a tilted sweep on-pad): at tilt θ the
contact drifts dY/dZ = −tanθ, so bias the Y anchor negative and keep the Z
climb modest. Values that worked: 20°→y=−9, 35°→y=−15. Keep the whole sweep
in the middle ~60 mm of the rod's 140 mm length (avoid both ends) and within
±~10 mm of the rod centreline in Y (rod is only 26 mm wide).

### 15.2 RESOLVED — why the stitched band is brighter at one end (NOT force, NOT motion)

Kourosh asked why a straight *vertical* centred sweep still shows a top-hot /
bottom-cool gradient. Chased it properly this time (I guessed wrong twice
first — coverage, then grasp-force — both refuted by the data):

- **Grasps apply EQUAL force.** diag7 per-grasp totals were flat (~11.4–12k
  across all 6). The object did NOT move (0.02 mm). Orientation dead straight.
  So blind closure does NOT explain it — Kourosh was right to reject that.
- **The real cause: the pad's intrinsic pressure profile — hottest near its
  TOP edge (rows 5–6), weakest at the tip (rows 0–2), in EVERY grasp.**
  Physical explanation (Kourosh's, correct): the Robotiq 2F-85 fingers are
  hinged near the base and rotate as they close, so contact pressure is
  highest near the hinge (top of pad) and lowest toward the fingertip
  (bottom). A pivoting finger, not a pure parallel press.
- **Why that makes the MAP top-hot:** row 0 = physical bottom, so the hot
  rows 5–6 sit at zs = +5.3 / +10.6 mm — they paint ABOVE each grasp's
  centre. Stack grasps climbing the rod and those "hot-above-centre"
  contributions pile toward the top. VERIFIED by simulating the exact
  `_taxel_centers` geometry with the measured bottom-heavy profile: it
  reproduced hottest-at-Z≈1113, fading down — matching the real map.
- **Tuesday sentence:** "The pad presses hardest at its top edge because the
  fingers are hinged near the base and rotate as they close — highest
  pressure near the hinge, lowest at the tip. Stacked up the rod this makes
  the map brightest at the top. I verified the grasps apply equal force and
  the object doesn't move — it's finger mechanics, faithfully captured."
- This is a fourth independent piece of evidence for contact-aware closure
  (§9.4) being the right next step (a force-referenced grip would flatten the
  per-grasp profile).

### 15.3 RELATED WORK — three 2020/2026 papers reviewed (all VBTS, cite as motivation/contrast)

All three use VISION-BASED tactile sensors (GelSight/DIGIT), not our
capacitive TSF-85 taxels, so none is a method to import. Ranked by usefulness
to Paper 3:

**(A) Smith et al., "3D Shape Reconstruction from Vision and Touch", NeurIPS
2020 — THE closest to our problem; lead related-work with it.**
Fuses vision + multiple touches into a 3D mesh via chart-based GCN, trained
on SIMULATED touch. Two findings ARE our thesis: (i) "reconstruction quality
increases with the number of grasps" = our multi-grasp stitch; (ii) "touch
extrapolates to its local neighbourhood beyond the touch site" = contact
completion, the core of Paper 3. Also uses a U-Net to turn a raw touch into
local geometry (our two-U-Net plan is the same family), and trains on sim
because real data is expensive (our exact justification). Differences to
state honestly: they FUSE WITH VISION (we are touch-only — cleaner, harder,
a novelty axis); they output a full 3D MESH (we output a 2D extended contact
map like Papers 1–2). Framing line: "Smith et al. showed simulated touch lets
a network extrapolate geometry beyond the touch site and that more grasps
help — but they rely on vision and reconstruct 3D meshes; we do the same
extrapolation from touch alone, as 2D contact maps, replacing our hand-crafted
classifier with a learned model."

**(B) TacLoc, "Global Tactile Localization … Registration Perspective",
arXiv Mar 2026 — justifies our object-frame stitching.**
Estimates OBJECT POSE by registering touch point-clouds to a known CAD model.
Different goal (localization, not completion) and needs a CAD model (we
don't). BUT its pose-composition (T_base^obj = T_base^ee · T_ee^obj) is the
formal basis for placing each grasp by its recorded pad pose — the principled
justification for object-frame stitching that JP was probing. Its future-work
explicitly names "fusion of multiple tactile measurements" — which our
extended map provides. Cite exactly there.

**(C) Tacmap, "Bridging the Tactile Sim-to-Real Gap via … Penetration Depth
Map", arXiv May 2026 — motivation + limitations reference.**
Vision-based sim-to-real via a penetration-depth representation for RL
policies. Different sensor, different task (policy transfer). Useful only as:
(i) evidence the field endorses simulated tactile data + learned translation
(justifies using Berith's data); (ii) a citable example of the sim-fidelity
gap for our limitations section. Do NOT claim overlap with our method.

### 15.4 CURRENT STATE — meeting-ready

Everything for Tuesday is in hand and verified: data factory (placement
0.13 mm, conventions pinned), stitch validation feature live in the GUI
(SSIM/TC/GSR), tilt captured at four angles, temporal fixed (all 4 columns
valid on backed grasps), first verified training pair exported, and a
measured physical explanation for every feature of the maps (edge bleed,
brightness gradient, cold wedge). The one genuinely open technical item
before scaling to more diameters is CONTACT-AWARE CLOSURE (§9.4), now backed
by four independent pieces of evidence (GSR saturation, edge-grasp fidelity
drop, pt05-s2 hold slip, and the per-grasp brightness profile).

*End of Handoff v6.2.*
