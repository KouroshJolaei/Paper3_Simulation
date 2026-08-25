# PAPER 3 — HANDOFF v8.0
**Written 4 August 2026, end of session.** Supersedes v7.0 (31 July) where they
disagree. **Keep v7.0 alongside this file** — v8.0 records the state as of today
and everything that changed on 3–4 August, but v7.0 still holds the full
narrative of the July debugging cascade (temporal snapshot timing, USD Fabric vs
PhysX, the frozen `pad_actual_pos_m`, the stitcher column-axis reversal, the s2
mirror elimination, the diag7 evidence). Do not discard it.

---

## 0. HOW TO USE THIS FILE

Read sections 1–3 to know where the project is. Read section 4 before touching
any code, because five source files changed today and one of them changed the
geometry of every stitched map. Section 8 is the next-step list.

**Verify file freshness before editing.** Uploads to a new chat are not
guaranteed to match disk. Grep for a known recent signature first — for this
session the signatures are:

| File | Signature to grep for |
|---|---|
| `sim/collect_from_config.py` | `pivot correction` |
| `viz/stitching.py` | `pad centre from EE+FK` |
| `main_gui.py` | `SESSION FOLDER` |
| `viz/heatmaps.py` | `_resolve_vmax` |
| `viz/temporal_snapshots.py` | `_scale_lbl` |

---

## 1. BIG PICTURE

Kourosh Jolaei, PhD candidate, CoRo Lab, ÉTS Montréal. Supervisors Vincent
Duchaine and Jean-Philippe Roberge (JP). Dissertation is three papers on
tactile-guided robotic grasping.

- **Paper 1** (J. Robotics and Mechatronics, Vol.38 No.3, 2026) — shape-based
  contact extrapolation. A classifier labels the object cuboid / sphere /
  cylinder from the initial tactile imprint, then shape-specific rules complete
  the unobserved contact. 94.3 % held-out, 80.8 % on unseen real objects. Metrics:
  Tactile Centroid (TC) error, Grasp Success Rate (GSR), SSIM, Safe Zone.
- **Paper 2** (submitted to *Robotics and Autonomous Systems*, Elsevier, June
  2026) — Tactile-Guided ReGrasping: virtual search over (Δx, Δy, θ) on the
  extrapolated manifold, introducing Centroidal Depth Translation (CDT). 602
  physical trials across 86 precarious configurations, 15 objects. ΔGSR ≥ 10 % in
  90.7 % of trials, 85 % stability plateau, mean estimation error −0.7 %.
- **Paper 3** (this work) — **replace the hand-crafted classifier +
  extrapolation pipeline of Papers 1–2 with a learned contact-completion model**,
  trained on large-scale Isaac Sim 5.1 data. This directly answers the
  classifier-robustness criticism of Paper 2 (large-diameter cylinders
  misclassified as cuboids — 9 of 14 cylinder errors).

**The four-block plan.**

| Block | What | Status |
|---|---|---|
| 1 | Data collection in Isaac Sim (grid sweeps, tactile CSVs) | **Complete, validated** |
| 2 | Stitching single 7×4 maps into an extended contact map | **Complete, validated; geometry corrected today** |
| 3 | Train a U-Net contact-completion model on (initial map → extended map) pairs | **Not started** |
| 4 | A/B comparison: learned completion vs Paper 1's shape rules | **Not started** |

**Hardware / sim stack.** UR5e + Robotiq 2F-85, two TSF-85 capacitive tactile
sensors (7×4 = 28 taxels per pad). Isaac Sim 5.1, TSF-85 extension by Berith
Atemoztli De la Cruz Sánchez, cuRobo for motion planning (injected via
`sys.path`, **not** pip-installed). Ubuntu 22.04, RTX 2060.

**Vincent's stated priorities:** fast verifiable steps; a benchmark against
frontier vision models. Meetings Tuesdays and Fridays.

---

## 2. THE SCIENTIFIC PROBLEM THAT DOMINATES THIS SESSION

### 2.1 The tilt finding

Grasping a cylinder tilted about world X, the contact line on the flat pad
should lean by the tilt angle. It does not:

| Object tilt | Measured blob axis | Fraction of truth |
|---|---|---|
| 0° | ~0° | — |
| 20° | ~1° | 5 % |
| 35° | ~16° | 47 % |

Measured with `viz/blob_axis.py`, which reproduces **Paper 2's own method**
(`virtual_search.generate_eigen_align`): upsample 7×4 → 70×40 cubic, threshold at
0.4375 of range, pressure-weighted covariance in mm, principal eigenvector. The
one deliberate difference from Paper 2 is documented in the file: Paper 3 stores
**row 0 = pad's physical BOTTOM** (proven 2026-07-22), whereas Paper 2 stores row
0 = top, so copying Paper 2's sign would negate every angle.

Ruled out as causes: pad orientation error vs commanded (0.007°), object motion
during close (0.00 mm, FixedJoint), diameter (reproduced on Ø20, Ø26, Ø50 over
60+ grasps), indentation depth (4× depth moved the angle only +0.89°), stitching
(the effect is in the single 7×4 map, before stitching).

### 2.2 THE DECISIVE NEW EXPERIMENT (3 August)

Instead of tilting the object, **roll the sensor pad** against a straight
vertical rod. Geometrically identical — 20° of relative obliquity — but through a
completely different code path (tool quaternion + cuRobo, versus the object's USD
transform + PhysX FixedJoint).

**Result: `run_20260803_200108`, pad rolled 20°, expected ±20°:**

```
s1: angle -5.97 deg   elongation 2.09   1148 cells
s2: angle -1.82 deg   elongation 3.11   1309 cells
```

Both elongations are well above the 1.5 floor, so the axis is meaningful and not
a round-blob artifact. **Both setups under-read by a large factor.** This
exonerates the object-side scene setup entirely, leaving the sensor model — i.e.
Berith's CNN — as the common cause.

**Caveats that must be stated if this is quoted:**
1. The 0° control at `GRASP_ROT_DEG=0` **has not been run**. Without it you
   cannot claim −5.97° is signal rather than the metric's noise floor.
2. s1 and s2 disagree by 3× on a symmetric grasp. Unexplained.
3. Single grasp. Not repeated.

### 2.3 The mechanism, and Kourosh's own framing

Berith's CNN was trained on 13,000 maps from a rig where the sensor sat fixed on
a base and a Mark-10 dynamometer pressed 49 interchangeable indenters
**straight down** into it, 0–50 N, depth read by a dial indicator. Validation in
his paper is normal indentation or a parallel gripper closing on an upright
object. **Oblique contact was never in the evaluation set.**

Kourosh's insight, recorded verbatim because it is the right framing: a model
trained only on axis-aligned normal indentation has no reason to ever produce a
leaning imprint, so 0° is not the CNN "getting it right" — it is the CNN's
default output happening to coincide with the truth, which is why error grows as
you rotate away from it.

**A second, possibly more central hypothesis (Kourosh's):** the training set may
also lack **edge contact** — a dynamometer indenter pressed into a fixed sensor
almost certainly produces fully-enclosed imprints, never a contact patch running
off the pad boundary. Paper 3's entire premise is imprints that continue past
the frame edge. Partial counter-evidence: straight-rod runs *do* show contact
reaching the frame edge and stitching into a continuous band, which suggests
edge contact works and orientation is what collapses. Not settled.

### 2.4 The unresolved row anomaly (opened today)

In `run_20260804_...` with pad offset y = −16, z = +65 (pad hanging over the rod
tip), after the placement fix the geometry is:

- pad spans Z 1098.7 – 1135.7 mm; rod top is at Z 1122.2 mm
- so rows 4, 5, 6 (centres 1122.5, 1127.8, 1133.1) sit **past the end of the rod**
- they should read zero. **They read roughly 330, 270, 250.**

In Y the geometry checks out perfectly: only the rightmost column centre
(191.25 mm) falls inside the rod's 186–212 mm span, and that is exactly the one
hot column observed. So geometry explains the column but not the rows.

This is the sharpest evidence yet for the edge-contact hypothesis. **Confounds:**
it is a weak grazing grasp (max 334 vs 1181 centred), and the row-gain artifact
(§3.4) is the same order as the signal.

**Proposed next probe:** `*_mesh_state.csv` sits *before* the CNN and would
separate "the physics presses rows 4–6" from "the CNN invents them" — the one
question the flat-plate test cannot answer. Note Berith said on 31 July that the
deformation files are "at one point just making garbage," so treat with care.

---

## 3. VERIFIED CONSTANTS AND FACTS

### 3.1 Calibration (`Data/pad_offset_calibration.json`)

**WARNING: the file currently contains ONLY Ø26.** The Ø20, Ø50, Ø18 and Ø13/Ø52
entries were wiped when Ø26 was re-measured on 3 August. **Restore from
`pad_offset_calibration_BACKUP.json` before any non-26 run.**

Current Ø26 (measured `20260803_142653`, good):

```
TOOL_OFFSET_Z            0.15657
TOOL_OFFSET_Z_case_origin 0.13447
pad_center_above_case_m  0.0221
close_rad                0.557
finger_joint_rad         0.5572      (reaches command — earlier 0.5474 did not)
tactile_peak_sum         15006.2
```

Other diameters measured earlier in the session (restore these):

| Ø mm | TOOL_OFFSET_Z | close_rad | peak sum |
|---|---|---|---|
| 13.0 | 0.15720 | 0.679 | 10914.7 |
| 18.0 | 0.15706 | 0.632 | 14000.1 |
| 20.0 | 0.15696 | 0.613 | 15177.0 |
| 50.0 | 0.15320 | 0.330 | 15704.5 |
| 52.0 | 0.15279 | 0.311 | 14700.6 |

### 3.2 Pipeline conventions (unchanged from v7.0)

```
HOLD_FRAC             = 0.9      hold-average window
SUBTRACT_BASELINE     = True
MIRROR_S2_IN_OVERLAY  = False    (s2 mirror convention ELIMINATED — diag7 proved
                                  both sensors share one world-Y column convention)
WAIT_HOLD_SECONDS     = 3.5
APPROACH_H            = 0.10     DO NOT lower — see §4.6
row 0 = pad's physical BOTTOM
PAD_W, PAD_H          = 22.0, 37.0 mm
PITCH_Y, PITCH_Z      = 5.5, 5.2857 mm
```

### 3.3 Contact width is compliance-limited, not silhouette-limited (settled)

A flat pad on a round rod touches only near the middle. Half-width ≈ √(2Rδ).
With δ ≈ 1.16 mm indentation on Ø26 the band is ±6–7.5 mm (≈2 columns of
5.5 mm), against a geometric edge at ±13 mm. Lighting all four columns would need
δ ≈ 4.65 mm — an unrealistically hard grip. **This is correct, learnable
structure, not a bug.**

Diameter sweep confirms it is below taxel resolution: predicted full widths
7.8 / 11.0 / 15.6 mm for Ø13 / Ø26 / Ø52 — a 4× diameter change moves the band by
only ~1.4 taxels on a 5.5 mm pitch. **Do not use bicubic upsampling to "see" it**
(that invents smoothness); measure the pressure-weighted **column spread** in mm
on the raw 7×4 instead.

### 3.4 The row-gain artifact (open, characterised)

On a straight vertical rod every row presses on identical geometry, so all rows
should read equally. Measured row sums: **308, 482, 305, 500, 563, 482, 403** —
brightest is 1.85× dimmest, and it **zigzags** rather than ramping. No physical
lever (tilt, hinge, finger wear) produces a jagged profile, so it is almost
certainly baked into the CNN. Survives baseline subtraction (46 % → 48 %). Does
not move the blob angle (0.134 → 0.128).

**Not proven.** The clean confirmation is the flat-plate test (§8), where every
taxel must read the same by construction, yielding a full 7×4 correction map.
Vincent independently spotted this on 28 July ("the top left taxel is always
highly activated") and already attributed it to the model rather than the object.

### 3.5 Training-pair noise floor

2.4 % RMS. Two independent runs of the same sweep gave visually indistinguishable
stitched maps (overlap sigma 118 vs 114). This is the reproducibility evidence
for Block 2.

---

## 4. WHAT CHANGED THIS SESSION (3–4 August)

Five files changed. All were built on verified baselines and compile-checked.
**Back-ups were made with `_before_*` suffixes.**

### 4.1 Pad roll enabled end-to-end (NEW CAPABILITY)

`collect_from_config.py` already contained a complete but disabled in-plane spin
block (`ROT_DEG = 0.0  # ... (yet)`). One line was changed to:

```python
ROT_DEG = float(os.environ.get("GRASP_ROT_DEG", "0.0"))
```

**The correct axis is `y`.** `GRASP_ROT_AXIS` defaults to `"z"` in the file, but
z is the tool approach axis (a null test — the gripper spins around the rod
without changing contact) and `x` tips the gripper off the rod entirely.
**Verified visually 3 August: `y` rolls the pad in its own face plane.**

Usage: `GRASP_ROT_DEG="20" GRASP_ROT_AXIS="y"` on the run command; the GUI now
emits both automatically.

### 4.2 PIVOT CORRECTION (BUG FIX — this made rolled grasps actually work)

`collect_from_config.py` line ~233 computed the EE target as
`gz = OBJ_CENTER[2] + dz + TOOL_OFFSET_Z` — a **pure world-Z offset**. That is
only valid while the tool hangs vertically. Rolling the tool swung the pad on the
156.57 mm flange-to-pad lever arm instead of spinning it in place:

| Roll | Y shift | Z shift |
|---|---|---|
| 10° | −27.2 mm | −2.4 mm |
| 20° | −53.6 mm | −9.4 mm |
| 35° | −89.8 mm | −28.3 mm |

The first 20° attempt closed on **air** (peak sum 289 instead of ~15,000).

Fix, inserted after the spin block, guarded by `abs(ROT_DEG) > 1e-6` so
non-rotated runs are bit-identical:

```python
v_world_new = R(tq) @ (R(tq_base).T @ [0, 0, TOOL_OFFSET_Z])
```

Then each `GRID_POINTS` entry is re-derived from its pad target. Safe to place
there because `GRID_POINTS` is not consumed until line ~1680.

**Verified:** after the fix, EE landed within 0.05 mm of the corrected target and
peak sums returned to 17,881 / 14,069.

### 4.3 Stitcher: TRUE pad centre (MAJOR GEOMETRY FIX)

`pad_actual_pos_m` in `pose_history.json` **is not the pad centre.** Measured
against the same run's `pad_truth_probe.json`, it matches the sensor **CASE**
prim at **OPEN** grip to 0.26 mm. It is therefore wrong by two independent
amounts:

```
open -> closed finger swing    13.04 mm
pad centre above the case      22.10 mm  (PAD_CENTER_ABOVE_CASE_M)
-------------------------------------------
total                          35.14 mm in Z   (and 14.2 mm in Y)
```

**Why this never surfaced before:** `_reanchor_to_gui()` removes exactly this
constant offset — but it requires `len(keys) >= max(2, len(offs)//2)`, so it
silently bails on **single-point runs**. Multi-grasp sweeps were always fine.

**The fix.** The true pad centre is derived from the flange pose:

```python
pad_centre = ee_world_m + R_pad[:, 2] * TOOL_OFFSET_Z
```

Because it goes through the pad's own rotation it stays exact when the pad is
rolled. Verified against both runs of 4 August: agrees with the commanded pose to
**0.03 mm** flat *and* at 20° roll. Implemented as a new preferred source
`pose_history.json [pad centre from EE+FK]`, with re-anchoring skipped for it and
the legacy sources kept as fallbacks for older runs lacking `pad_actual_R`.

**Effect on the z=65 test:** placement error went from Y −14.2 / Z +35.2 mm to
Y −0.01 / Z +0.03 mm. Confirmed correct by replot — Y spans 172–194, Z spans
1098.7–1135.7, and only the rightmost column lights up, exactly as predicted.

### 4.4 Stitcher: rotated splat (NEW)

The splat painted every 7×4 map on an axis-aligned lattice, so a rolled pad was
drawn upright. Added `load_pad_bases()`, which reads each grasp's pad axes from
`pad_actual_R` — **measured, not commanded**, so an arm that misses the requested
roll shows up honestly:

```
across (4 columns, PITCH_Y) =  R[:, 0] projected to (Y, Z)
up     (7 rows,    PITCH_Z) = -R[:, 2] projected to (Y, Z)
```

Verified: flat gives exactly (1,0)/(0,1); the 20° run gives (0.9397, 0.3420) =
(cos 20°, sin 20°), orthogonal. `_splat_one` gained a rotated branch; **the flat
separable fast path is preserved verbatim**, so upright runs paint bit-identically.

### 4.5 Stitcher: canvas sizing for rolled pads (BUG FIX, last change of session)

The canvas was still sized for the flat 22 × 37 footprint, so a rolled pad was
clipped and the coverage map came out as an octagon. Fixed to use the rotated
bounding box:

```
width  = W|cos| + H|sin|
height = W|sin| + H|cos|
```

At 0° this reduces exactly to 22 × 37, so flat runs are unchanged. At 25° the
canvas becomes 35.6 × 42.8 mm.

**THIS IS THE ONE CHANGE NOT YET CONFIRMED BY A REPLOT.** First action in the
next session: replot the −25° run and check the coverage panel shows a clean
rotated rectangle rather than an octagon.

### 4.6 APPROACH_H — do not lower it

During the 45°/35° reachability fight, `APPROACH_H` was lowered from 0.10 to
0.04. **This broke a previously-working 0° run** — cuRobo's optimiser fails on a
40 mm descent. It has been restored to 0.10. Kourosh's file has a leftover
commented line above it, so it sits at line 124 rather than 123.

### 4.7 Reachability pre-check is NOT a guarantee

Confirmed twice today. The pre-check tests IK at the **grasp** pose only; the
run then fails planning the free move to **UP** (grasp + `APPROACH_H`). A green
report followed by `[pt00:to-up] free move FAILED (Opt Fail)` is the signature.

Rolled-pad reachability outcomes so far:

| Roll | z offset | Outcome |
|---|---|---|
| 10° | 30 | Works (axis y) |
| 20° | 30 | **Works — this is the result run** |
| 35° | 50 | IK failed (EE ends up higher: drop is only 156.57·(1−cos θ)) |
| 45° | 50 | Pre-check OK, free-move to UP failed |
| 90° | 30/50/60 | cond(J) = 4978 — near-singular; no z offset helps |

Untried: negative angles (`-35`), which swing the flange to the opposite side and
give a completely different arm configuration.

### 4.8 GUI: session folders, colour scale, quality-of-life

- **SESSION FOLDER** section at the top of the Collection tab. *New session* mints
  one folder from clock + angles (`run_20260804_172021_obj0_pad20`; negatives → `m`,
  decimals → `p`). Both the reachability command and the run command receive it via
  a new `GRASP_RUN_DIR` env var, so **one test = one folder**. Also *Use existing…*
  and *Open folder* (xdg-open). The session becomes the default plot source and the
  default browse directory everywhere.
- **Pad `rotation (deg)` field ungreyed**, bound to Enter, drawn as a rotated
  polygon about the pad centre in the FRONT preview; TOP-DOWN pad shadow widened to
  `W·|cos| + H·|sin|`. Emits `GRASP_ROT_DEG` + `GRASP_ROT_AXIS="y"` only when
  non-zero, so existing commands are byte-identical.
- **Load Experiment fixed.** It only ever read a `gui_fields` key, so picking a
  plain config set nothing and *still reported success in green*. Now accepts
  experiment recipes, `gui_config.json`, and a run's `gui_config_used.json` /
  `pose_history.json`, reports how many fields loaded, and warns when a file matches
  nothing.
- **Save Config** now also writes `gui_config_used.json` + `gui_preview.png`
  straight into the session folder, so pressing it alone leaves a complete record.
  The collector copies the preview into the run folder too. *Ordering rule: the PNG
  is captured when you press Save Config, so set the preview first.*
- **Reachability** writes to `Data/reach_check/` (superseded by session folders,
  kept as the no-session fallback).
- **Grid warning:** with a non-zero roll and >1 grid point the label turns red,
  because the sweep still steps in world Y/Z while the pad's axes are rotated. See
  §7.1.

### 4.9 Colour-scale policy (shared across all plotting)

`stitching.py` is now the single source of truth (alongside `HOLD_FRAC` and
`SUBTRACT_BASELINE`). Reads `Data/plot_scale.json`; GUI control is a checkbox
plus a "fixed max" box in the Collection tab.

| checkbox | fixed max | mode |
|---|---|---|
| off | blank | per-figure auto (old heatmaps behaviour) |
| **on** | blank | **one scale across the run (default)** |
| either | e.g. `2400` | that scale in every test — for cross-run comparison |

2400 matches Paper 2's tactile-count figures. Every colorbar now states its mode.
Nothing hides behind the scale: heatmap titles gained `max <single-taxel>`,
stitched titles gained `canvas max`, temporal per-grasp panels gained `max`.

This also removed a real inconsistency: `heatmaps.py` auto-scaled per grasp while
`temporal_snapshots.py` already shared one scale across the run, so panels from
the two were never comparable.

**Demonstrated value:** with fixed 2400, the Ø26 centred grasp (max 1181) renders
as faint cyan-green while three grazing tests (max 97 / 334 / 329) are nearly
black — information the auto-scaled versions actively concealed.

### 4.10 Disk

Root filesystem hit 100 % and killed a run mid-write
(`OSError: [Errno 28] No space left on device` from the TSF-85 `dz` writer).
Recovered ~21 GB via `docker system prune -a --volumes` (20 GB) and
`journalctl --vacuum-size=200M` (1.5 GB). `~/.cache/ov` was nearly empty.
Currently 90 % used, 21 GB free. **Watch this** — the `*_mesh_state.csv` and
`*_deformations.csv` files are written every physics frame and are the bulk
consumers.

---

## 5. WHAT WAS SENT TO BERITH — AND ONE ERROR IN IT

`kourosh_pipeline_summary.py` was sent (a readable summary, not the real
collector). **No reply as of 4 August.**

**It contains one claim the handoff contradicts.** Section 1 states: *"I also
tried making the body kinematic instead — same result."* Per v7.0 §7.3 the
kinematic test has **never successfully run**: attempt 1 lost the tilt (it lived
in the joint's `LocalRot0`), attempt 2 hit the `AddXformOp` precision error,
attempt 3 was never run. This is precisely the thing Berith is most likely to
probe. **Correct it proactively.**

The summary also usefully documents, for reproduction: `scene_cylinder.usd`,
`Object_02/Cylinder` as a unit mesh scaled `(D, D, L)`, `OBJ_CENTER_M`, tilt about
the rod's centre, the FixedJoint, the extension settings, `Q_GRASP`, both driven
finger joints, and the close/hold/open sequence at 60 Hz.

---

## 6. THE SCENE (findings from today)

`/World/Object_01` and `/World/Object_03` are **empty Xforms with only a material**
— no geometry. There is no box hiding in the scene. The only mesh is
`Object_02/Cylinder`, a unit mesh whose real size is entirely its scale.

To inspect the (binary) USD you must go through a running SimulationApp — `pxr` is
not importable from `isaacsim/python.sh` directly (v7.0 §7.4). A working
`inspect_scene.py` was written and lives in `~/Paper3_Simulation/`.

**For the flat-plate test, two routes:**
1. **Squashed cylinder (cheap).** Scale is set on one line as `(d, d, L)`. Make
   the Y scale independent: keep X = 26 mm (calibration and `close_rad` stay
   valid), set Y = 200 mm. Contact radius becomes 100²/13 ≈ 769 mm, so the surface
   droops only 0.08 mm across the pad's ±11 mm versus 1.16 mm of indentation —
   effectively flat. Suggested implementation: a `GRASP_OBJ_DIAM_Y_MM` env
   override defaulting to the X diameter, so every existing run is unchanged.
   *Risk to check first: a 200 mm-wide slab may collide with the fingers or fail
   reachability.*
2. **Ask Berith for a box.** His paper used square and triangular indenters plus
   rectangular-prism test objects, so he likely has a box USD/STL ready. Don't
   block on it.

A huge-diameter cylinder does **not** work: the jaw span is 85 mm, so a Ø1000
rod cannot be grasped.

---

## 7. KNOWN-BROKEN / NOT-YET-DONE

### 7.1 Rotated grids are wrong (blocking for rolled-pad data collection)
The sweep still steps in world Y/Z while the pad's axes are rotated, so the pad
walks **diagonally across its own frame**. Fine for single-point tests, wrong for
grids. The GUI warns in red. **Must be fixed before collecting any rolled-pad
grid.**

### 7.2 `validation.py` not updated for rolled pads
`_sample_canvas()` inverts the splat cell-for-cell and still assumes an
axis-aligned footprint. **"Validate Stitch" gives wrong SSIM and TC on rolled
runs.** Correct on flat runs. Needs the same basis treatment as `_splat_one`.

### 7.3 Drawing-only: column 3 pad box, column 4 dashed frame
Both draw square regardless of pad roll. The *data* lands correctly; only the
overlay rectangles are wrong. Also, the cylinder outline in column 3 should lean
for object tilt.

### 7.4 The staircase in rolled stitched maps
Rotated taxel Voronoi cells alias against the 1 mm canvas grid. Cosmetic —
lower `res_mm` to 0.5 if it matters for a figure.

### 7.5 GSR is saturated
`validation.py` already prints a loud warning: on grasps that all trivially
succeed, GSR has zero dynamic range and `GSR_err = 0` carries **no information**.
Do not quote it as validation.

### 7.6 Calibration file has only Ø26
See §3.1. Restore from backup before any non-26 run.

---

## 8. NEXT STEPS, IN ORDER

**Immediate (finish what is in flight):**

1. **Replot the −25° run** and confirm the coverage panel is a clean rotated
   rectangle, not an octagon (§4.5). This is the only unverified change.
2. **Run the 0° control** (`GRASP_ROT_DEG=0`, same grasp). This establishes the
   blob-axis noise floor and is required before −5.97° can be quoted as a result.
3. **Restore the calibration file** from `pad_offset_calibration_BACKUP.json`.

**The science:**

4. **Flat-plate test (§9.1 of v7.0).** One grasp on a 26 mm flat plate where
   every taxel must read identically by construction. Settles the row-gain
   question *and* yields a full 7×4 per-taxel correction map — better than the
   7-vector. Turns "I think it's the CNN" into a measured artefact you can send
   Berith.
5. **Probe `mesh_state.csv`** for the z=65 run to separate physics from CNN on
   the row anomaly (§2.4).
6. **Correct the kinematic claim to Berith** (§5) and follow up.
7. Optionally: 35° or −35° pad roll at z=30 for a same-angle comparison against
   the 35° object tilt. Do not spend many runs fighting the wrist — 20° already
   carries the finding.

**Unblocking Block 3:**

8. Fix rotated grids (§7.1) *if* rolled-pad training data is wanted; otherwise
   collect straight-object data now and treat tilt as a documented limitation.
9. Fix `validation.py` for rolled pads (§7.2).
10. **Start Block 3.** `export_pair()` already writes `training_pair.npz` with
    *both* target variants (full-stitch and composite), so the format decision is
    deferrable to training time and is **not** blocking.

**For Vincent:**

11. Vincent's 28 July assignment — the training pair presented his way, i.e. the
    original 7×4 shown **inside** the symmetric extended frame (his 11×8 example),
    built from **centre frames only**, plus a plan to scale collection up. The
    4th column of the stitched figure already does this.
12. Vincent also proposed a **parallel real-data path**: teach-pendant collection
    on a fixed cylinder, stepping ~5.5 mm, two taxels each direction, as a hedge
    while Berith's pipeline is in flux (he is migrating to the Newton physics
    engine). This is under-recorded and worth raising.
13. Vincent wants a **benchmark against frontier vision models**. On 28 July he
    reported that Claude — working on the raw numeric matrix, not an image model —
    outperformed Gemini and GPT at extending a tactile frame, and was fast because
    it needed no image generation. He noted it uses no sensor priors.

---

## 9. STRATEGIC FRAMING FOR SUPERVISORS

**How serious is the tilt problem?** Serious but bounded. It does not block
Blocks 3–4 for straight objects. It *does* cap the dataset at contacts the sensor
model can represent: every tilted grasp generated is silently mislabelled, and
that is a limitation you cannot discover after training. Right move: hand the
diagnosis to Berith, build Blocks 3 and 4 on straight geometry now.

**How serious is the row non-uniformity?** Less so. A 1.85× fixed multiplicative
pattern will simply be learned as part of the sensor, and it does not move the
blob angle. It is a figures-and-limitations issue, not a data-validity one.

**What is genuinely strong right now:** Block 2 is validated and reproducible
across two independent runs; the exact figure Vincent asked for on 28 July
exists; the tilt anomaly is itself a finding with the arm, the object, the
diameter, the indentation depth and the stitcher all eliminated; and the
rolled-pad experiment is a clean second code path reaching the same conclusion.

**JP's contribution (28 July), worth revisiting:** he argued stitching should be
thought of in an **inertial reference frame** on a *deformation map*, not as 2D
image registration — and that when a new contact layer appears with no overlap
(his mobile-robot-through-a-door analogy), you **register** rather than stitch.
Vincent's counterpoint: a tactile sensor only ever sees the intersection plane,
not depth, so a convex object moving out of frame generates a contact layer that
cannot be reconciled with the first — and the gripper opening tells you when you
have changed plane. Vincent's practical conclusion: keep it 2D, and stop
recording when the data forces a plane change.

---

## 10. WORKING PREFERENCES (carried forward — please honour)

- **One verifiable step at a time.** Measure, don't guess.
- **Distances in mm.**
- **Complete files, never diffs.**
- **Concise, direct answers** — one to three lines preferred unless depth is asked for.
- **Kourosh's visual read of the simulation overrides diagnostics when they
  conflict.** This has been correct every single time.
- **Never present schematic pseudocode as a quotable line from an actual file.**
- **Verify file freshness before editing** (see §0). Twice this session an
  assistant working copy silently drifted from the delivered file; both times the
  fix was to rebuild from the user's uploaded version. If a checksum or a grep
  disagrees, trust the user's disk.

---

## 11. FILE INVENTORY

**Changed this session** (backups exist with `_before_*` suffixes):

| File | Change |
|---|---|
| `sim/collect_from_config.py` | `GRASP_ROT_DEG` enabled; pivot correction; `GRASP_RUN_DIR`; angle-tagged folder names; copies `gui_preview.png` |
| `main_gui.py` | pad rotation field; session folders; colour-scale control; Load Experiment fix; Save Config to session; Open folder |
| `viz/stitching.py` | pad centre from EE+FK; rotated splat; rotated canvas sizing; colour policy source of truth |
| `viz/heatmaps.py` | colour policy; per-panel max in titles |
| `viz/temporal_snapshots.py` | colour policy; per-panel max in titles |

**Unchanged but relevant:** `viz/blob_axis.py` (still lacks the elongation < 1.5
refusal guard — only `MIN_CELLS = 5`), `viz/validation.py` (§7.2),
`inspect_scene.py`, `kourosh_pipeline_summary.py`.

**Key data runs:**

| Run | Meaning |
|---|---|
| `run_20260803_200108` | **20° pad roll, the decisive result** — blob axis −5.97 / −1.82 |
| `run_20260804_..._obj0_pad0` (y=−16, z=65) | pad over the rod tip; row anomaly (§2.4) |
| `run_20260804_194103_obj10_padm25` | object 10° + pad −25°; used to find the canvas-clipping bug |
| Ø13 / Ø26 / Ø52 single-point series | compliance-limited contact width (§3.3) |

**People:** Berith Atemoztli De la Cruz Sánchez (TSF-85 Isaac extension and CNN
author; migrating to Newton), Vincent Duchaine (supervisor), Jean-Philippe
Roberge (co-supervisor; raised the inertial-reference-frame point).

---

*End of handoff v8.0.*
