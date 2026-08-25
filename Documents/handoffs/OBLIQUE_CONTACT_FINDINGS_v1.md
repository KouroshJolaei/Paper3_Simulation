# Oblique contact in the TSF-85 Isaac pipeline — findings as of 2026-08-06

Standing record of the pad-roll / rod-tilt campaign. Written to be read
alongside `PAPER3_HANDOFF_v8_0.md`; it does not repeat the pipeline state,
only what these tests established.

---

## 1. Why this campaign happened

Paper 3's data factory rests on Berith's TSF-85 Isaac extension producing
tactile maps that are faithful for contacts the CNN was not explicitly
trained on. The 31 July finding — the weighted-PCA blob axis reading ~1° at
20° tilt and ~16° at 35° — suggested it might not be. Before writing to
Berith, three things had to be ruled out:

1. the pad not actually being where the GUI said (ruled out: 0.007°
   orientation error vs commanded, `pad_truth_probe.json`),
2. the **expected** angle being wrong (fixed: see §2),
3. the **estimator** being the problem (quantified: see §3).

---

## 2. The expected angle had to be rebuilt (2026-08-05)

The first formula, `expected = roll − tilt`, was wrong twice:

* **Sign.** `blob_axis`'s x runs with the COLUMN INDEX, but `_taxel_centers`
  puts column 0 at the LARGEST across-coordinate. So blob_axis's x is MINUS
  the pad's across axis, and any hand-derived angle comes out negated.
* **The line is not the blob.** What the taxels see is the contact line
  CLIPPED by the 22×37 window. Push the pad off-centre and a tilted band runs
  off the edge, so the visible patch is a short stub whose principal axis is
  NOT the line's angle. At `pad_offset_y = −8 mm` with a 20° roll, a perfect
  sensor would read **−15.9°**, not −20°.

`expected` is now built geometrically: the contact band is rasterised onto
the real 7×4 taxel footprints via `stitching._taxel_centers` and run through
the SAME `blob_axis`. Nothing is transformed by hand, so the sign cannot be
wrong, and the 7×4 quantisation is automatically included.

**Band width = 15.0 mm**, fitted from an upright run's across-pad profile
(residual 0.0082, ≈ 2.4 mm indentation on the Ø26 rod). Use this for every
run so the series is comparable.

---

## 3. The estimator is not the explanation

`blob_axis.metric_selftest()` on a PERFECT straight line, no sensor and no
CNN — this is the metric's own transfer curve on a 7×4:

| true | 20° | 25° | 30° | 35° | 45° |
|---|---|---|---|---|---|
| PCA reads | 20.7 | 24.4 | 27.9 | 31.1 | 38.1 |

Essentially unbiased below ~25°, losing ~11% at 35°. **A measured angle far
below these values cannot be blamed on the metric.**

A no-contact floor was added on 2026-08-06 (`MIN_PEAK_COUNTS = 20`, raised to
2% of the run peak): the working threshold is a fraction of each map's own
range, so a noise-only map previously produced a confident-looking angle. On
the 35-point grid this silently invented angles of −8 to −12° for thirteen
grasps that never touched the rod.

---

## 4. PAD ROLL series — rod upright, pad rolled (STRONG evidence)

Coverage 57–100%, s1 and s2 agree closely. Band 15.0 mm.

| pad roll | expected | s1 measured | s1 error | s2 measured | meas elong | exp elong | cover |
|---|---|---|---|---|---|---|---|
| 0° | 0.00 | −1.25 | −1.25 | −0.12 | 3.14 | 3.31 | 100% |
| 15° | −16.4 | −4.2 | **+12.2** | — | — | — | — |
| 30° | −27.00 | −5.82 | **+21.18** | −7.20 | 1.88 | 2.94 | 86% |
| 45° | −38.38 | +3.41 | **+41.79** | +2.41 | 1.34 | 2.08 | 71% |
| 45° (repeat) | −36.39 | −1.11 | **+35.27** | −7.12 | 1.41 | 2.60 | 79% |
| 60° | −49.99 | −2.58 | **+47.40** | −2.65 | 1.18 | 2.08 | 71% |
| 75° | −67.00 | +89.81 | −23.19 | −89.95 | 2.12 | 1.75 | 57% |
| 90° | −89.8 | +89.7 | −0.5 | — | — | — | — |

**The measured angle stays near 0° through 15–60°, then snaps to ~90°
between 60° and 75°.** It never reports a diagonal. The 45° error is
repeatable (+41.8 then +35.3, both sensors, two independent runs).

Both endpoints are read correctly: 0° → −1.3°, 90° → +89.7°.

---

## 5. ROD TILT series — pad upright, rod tilted (WEAKER, needs redoing)

Same contact geometry, completely different code path — no wrist roll, no
flange swing, no tool-axis descent.

**The offsets HAD to change per angle, and this is geometry, not sloppiness.**
The rod rotates about its own centre, so a point 50 mm up its axis swings
sideways by `50·sin θ` — 35 mm at 45°. Holding `pad_offset_y = 0` would have
left the pad entirely off the rod with nothing to measure. The offsets were
therefore moved to keep the pad in contact.

The consequence still has to be carried, though: coverage fell to 32–50%
against 71–100% in §4, so the two series are not directly comparable, and
some of that is unavoidable — a diagonal 15 mm band crossing a 22×37 window
simply covers less of it than a vertical one.

| rod tilt | expected | s1 measured | s1 error | s2 measured | s2 error | meas elong | cover |
|---|---|---|---|---|---|---|---|
| 15° | +5.81 | −1.34 | −7.15 | +7.58 | +1.77 | 3.30 / 1.64 | 39% |
| 30° | +21.59 | **+22.62** | **+1.03** | +21.96 | +0.37 | 1.64 / 1.67 | 43% |
| 45° | +48.46 | −10.09 | −58.55 | +7.19 | −41.27 | 1.19 / 1.15 | 50% |
| 60° | +70.04 | +19.69 | −50.35 | +30.92 | −39.11 | 1.46 / 1.34 | 43% |
| 75° | +83.64 | −0.04 | −83.69 | −89.20 | +7.16 | 1.12 / 2.39 | 32% |

Two things to be honest about:

* **Rod 30° reads correctly** (+22.6 vs +21.6 expected, both sensors). That
  is a genuine ~22° diagonal resolved properly, and it is a real
  counter-example to "the CNN never resolves a diagonal".
* **s1 and s2 disagree badly** at 15°, 45°, 60° and 75° — by up to 66° at
  45°. Two pads squeezing the same rod should see mirror-image contacts.
  This disagreement does NOT appear in the pad-roll series, and low coverage
  alone does not obviously explain it. Unexplained.

Most of these rows have `elong` between 1.1 and 1.5, i.e. below the guard
where an axis is meaningful. **Treat this series as supporting, not primary,
evidence.** The pad-roll series in §4 is what the argument rests on.

---

## 6. The one finding robust across BOTH series

Measured blobs are systematically **rounder** than the geometry demands, and
the shortfall is worst where the true contact is most diagonal:

| series | angle | meas elong | exp elong | ratio |
|---|---|---|---|---|
| pad | 0° | 3.14 | 3.31 | 0.95 |
| pad | 30° | 1.88 | 2.94 | 0.64 |
| pad | 45° | 1.34 | 2.08 | 0.64 |
| pad | 60° | 1.18 | 2.08 | 0.57 |
| rod | 15° | 3.30 | 4.06 | 0.81 |
| rod | 30° | 1.64 | 2.04 | 0.80 |
| rod | 45° | 1.19 | 1.83 | 0.65 |
| rod | 60° | 1.46 | 2.08 | 0.70 |
| rod | 75° | 1.12 | 3.58 | 0.31 |

The failure mode is **not** a mis-angled ridge. It is the loss of the ridge:
where a diagonal band should appear, a rounder, more isotropic blob appears
instead, and PCA then reports an arbitrary near-axis-aligned direction. That
is consistent with a network whose training set contained only axis-aligned
normal indentation, where rows and columns are separable and a diagonal
ridge is outside the represented space — but it is a hypothesis, not yet a
demonstrated cause.

---

## 7. Pipeline fixes made during the campaign

All in `collect_from_config.py` unless noted.

| fix | what it was | evidence |
|---|---|---|
| rolled splat drawn/sampled correctly | column-3 box, column-4 frame, `validation._sample_canvas` and `export_pair` all assumed an upright 22×37 | old sampler: SSIM 0.53 where truth is 1.00; old outline contained 85.5% of painted cells |
| `pad_target_world_mm` | reported as `ee_z − TOOL_OFFSET_Z`, valid only upright | off by 54 mm at 20°, **221 mm at 90°** |
| `GRASP_ENABLE_GRAPH` (default 1) | `plan_free_move` used `enable_graph=False`, i.e. local optimisation only | IK probe found the 45° pose at 0.001 mm error but at `shoulder_pan +1.96` vs the arm's −0.85: reachable, 160° away, unreachable locally |
| `GRASP_APPROACH_ALONG_TOOL` (now default 1) | `plan_stitched_z` stepped along **world Z** while `CASE13_WEIGHT` + `project_pose_to_goal_frame=True` frees the **tool's** z | at 90° the two are perpendicular — a self-contradictory request; measured gap 141 mm |
| reason codes + `execution_ledger.json` | "unreachable" was one free-text string | caught the 90° false positive |
| IK probe on failed points | no way to tell "no solution" from "planner missed it" | verdict line, three probes |
| no-contact floor in `blob_axis` | noise-only maps produced confident angles | 13 phantom rows on the 35-point grid |
| `heatmaps.py` third panel | no way to see where a heatmap came from | pad pose on rod + robot visit order + skipped points |

After these, 0°, 15°, 30°, 45°, 60°, 75° and 90° pad rolls **all execute**,
with the pre-check and executor agreeing on every one.

---

## 7b. Verdict on data cleanliness

**Pose and geometry: clean.** Pad orientation error vs commanded is 0.007°,
the object does not move during close, the stitcher paints and re-samples the
rolled footprint correctly, upright runs are bit-identical to before every
fix, and pre-check and executor now agree on every angle tested.

**Tactile response: clean for axis-aligned contact, not for diagonal.** 0°
reads to ~1° and 90° to 0.5°, both with correct elongation. Between them the
angle collapses and the blob comes out ~35–45% rounder than the geometry
demands.

**Consequence for Paper 3.** Collection and training can proceed for the
axis-aligned regime — upright rod, upright pad, which is the Papers 1–2
setting and the bulk of the intended dataset. Rolled-pad and tilted-rod
grasps should NOT go into a training set until §9 is resolved: a
contact-completion model trained on maps where diagonal ridges are rendered
as round blobs would learn the renderer's failure, not the contact physics,
and no amount of downstream metric work would reveal it.

---

## 8. Still open

* **Collision is OFF** (`USE_COLLISION_WORLD = False`) in every run above.
  Nothing here is lab-certifiable. This must go on before any of this drives
  a real robot, and the rod is currently excluded from the world even when
  the flag is on.
* **The rod-tilt series stands as collected.** Fixed offsets are not
  possible (§5), so it remains supporting evidence at 32–50% coverage.
* **Where exactly the 0→90 snap happens** — 65° and 70° would pin it.
* **s1/s2 disagreement** in the rod series is unexplained.
* One run showed s1 holding **47** frames against s2's 210 — worth watching
  for a settle-timing asymmetry.

---

## 9. Questions for Berith

1. Did the 13,000-map training set contain any **obliquely oriented**
   contact? The paper describes indenters pressed straight down by a Mark-10
   with the sensor fixed, which would make every contact axis-aligned.
2. Is the CNN's input treated as separable in rows and columns anywhere in
   the architecture, or is there any augmentation by rotation?
3. Independent of angle: should a contact that *should* produce an elongated
   diagonal ridge come out ~35% rounder than the geometry implies?
4. Are there known limits on how far the model extrapolates outside its
   training distribution?

Attach: the §4 table, the §3 self-test curve, and the 45°/90° blob figures.
The 90° result matters as much as the 45° one — it shows a *horizontal*
line read to 0.5°, so this is specific to diagonals, not oblique contact in
general.
