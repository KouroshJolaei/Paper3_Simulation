# Tilted cylinder: the contact line does not lean

Hi Berith — here is the case we talked about on 31 July.

## What I see

A flat sensor pad, held **vertical**, grasping a cylinder that is **tilted
about world X**. The contact line in the tactile map should lean by the same
angle as the cylinder. It mostly does not:

| cylinder tilt | measured contact angle | fraction of geometry |
|---|---|---|
| 0°  | ~0°  | — |
| 20° | **~1°**  | 5 % |
| 35° | **~16°** | 47 % |

Measured with the weighted-PCA blob axis from our Paper 2 code
(`virtual_search.generate_eigen_align`): upsample 7×4 → 70×40 cubic,
threshold at 0.4375 of the range, pressure-weighted covariance in mm,
principal eigenvector.

## What I already ruled out

- **The arm** — pad orientation error vs commanded is **0.007°**
  (`pad_truth_probe.json` → `orientation_error_vs_commanded_deg`).
- **The object moving** — `object_moved_during_close_mm: [0.0, 0.0, 0.0]`
  on every run, including a 4× harder squeeze.
- **Diameter** — same result on Ø20, Ø26 and Ø50 (I scale
  `Object_02/Cylinder`, since it is a unit mesh).
- **Grip force** — raising `close_rad` from 0.55 to 0.62 (about 4× the
  indentation, peak 13 200 → 19 900) widened the contact but did **not**
  change the angle.
- **How the rod is held** — bolted with a `WorldFixedJoint` and kinematic
  give the same result.
- **Repeatability** — two identical runs agree to 2.4 % RMS.

60+ grasps at 20°, two runs at 35°, both sensors agree each time.

## Files

| file | what it is |
|---|---|
| `collect_from_config.py` | my collector (Isaac Sim 5.1 + cuRobo + your extension) |
| `gui_config_used.json` | the exact config for the 35° run |
| `gui_pt00_s1/s2_tactile_maps.csv` (35°) | tilted grasp, one row per physics frame |
| `gui_pt00_s1/s2_tactile_maps.csv` (0°) | straight grasp, same settings |
| `heatmap_*.png` | hold-average maps for both |

## Reading the tactile CSVs

- One row per physics frame at 60 Hz, full close → hold → open.
- `pred_0 .. pred_27` reshape to **7×4**, row-major, and **row 0 is the
  physical BOTTOM of the pad** (verified against a rod-tip grasp where only
  the pad's lowest 8.5 mm touched — rows 0–1 carried 59 % of the signal).
- My representative map is the mean of the frames at ≥ 90 % of peak sum,
  minus the pre-contact baseline (~267 counts, constant to ±1.5).

## Scene / grasp settings

```
object centre (world)  : (-0.26806, 0.199, 1.0522) m
cylinder               : Ø26 × 140 mm  -> Object_02/Cylinder scale (0.026, 0.026, 0.14)
tilt                   : 20° or 35° about world X, rotated about the centre
pad offset from centre : y = -10.9 mm, z = +30 mm
close_rad              : 0.55  (both finger_joint AND right_outer_knuckle_joint)
hold                   : 3.5 s
TOOL_OFFSET_Z          : 0.15651 m (measured for Ø26)
```

## To run it yourself

`collect_from_config.py` has absolute paths near the top (`EXAMPLES_DIR`,
`SCENES_DIR`, the calibration file). Change those to your machine and run:

```
GRASP_OUTPUT_DIR=<out> GRASP_BASENAME=gui \
  ~/isaacsim/python.sh collect_from_config.py --config gui_config_used.json
```

The config already contains the tilt, the object size and the single grid
point, so it should reproduce the 35° case directly.

## The question

Was any **obliquely oriented** contact in the 13 000 training maps? From the
paper, the dataset was made with the sensor fixed and indenters pressed
straight down by the Mark-10, so I am wondering whether a contact line that
runs diagonally across the pad is outside what the CNN has seen — and if so,
whether the 20° vs 35° difference is the network extrapolating.

Thanks for taking a look.
Kourosh
