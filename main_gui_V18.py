"""
main_gui.py — Paper 3 data-collection cockpit (STAGE A).

Pure PyCharm / Tkinter. NO Isaac needed. Lets you:
  - enter the OBJECT pose (mm) + orientation (cylinder for now)
  - enter the PAD initial pose (mm); rotation greyed out for now
  - enter a 2D GRID: n steps in X, n steps in Y, one step size (mm)
    (X,Y are along the cylinder surface, parallel to the pad face)
  - SEE a live TOP-DOWN + FRONT preview:
      TOP-DOWN: cylinder circle in middle, TWO pads facing each other along X
                (one -X side, one +X side), symmetric, pressing on the rim.
      FRONT:    pad face(s) on the cylinder surface, with the full 2D grid.

All distances shown in mm. Stage B will add the "write config + run Isaac"
bridge; Stage C the heatmap + pose-history read-back buttons.

Run in PyCharm:  python3 main_gui.py
"""

# --- FRONT (Y-Z) preview tick spacing, mm. Labelled ticks every MAJOR,
# --- faint unlabelled gridlines every MINOR. Change these to taste.
PREVIEW_TICK_MAJOR_MM = 10.0
PREVIEW_TICK_MINOR_MM = 2.0

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os, json, subprocess, threading, time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- real hardware sizes (mm) ----
PAD_W = 22.0    # pad short side (4 taxels)
PAD_H = 37.0    # pad long side  (7 taxels)
# Object size. These stay module-level because ~18 places read them, but they
# are now REFRESHED FROM THE GUI on every cfg() call (see _sync_object_size).
# The scene's Object_02/Cylinder is a UNIT mesh scaled by the collector, so a
# new diameter needs no new USD and no STL — just these numbers.
CYL_D = 26.0    # cylinder diameter (mm) — live, set by the Object size field
CYL_L = 140.0   # cylinder length   (mm) — live, set by the Object size field
GRIP_OPEN = 12.0  # half-gap of each pad from the cylinder rim before closing (mm, visual)
ROBOT_BASE_MM = np.array([20.93, -337.5, 992.75])  # robot base_link world (mm)

# ---- the gripper BODY, for the grid designer's collision check (mm) -------
# The check asks one question at every grid point: does the ROD end up inside
# the GRIPPER BODY? Two numbers describe that body.
#
# PALM_DROP_MM   how far PAST THE FLANGE the palm's underside sits, measured
#                along the tool axis. MEASURED: 10.79 (flange -> palm) + 75.9
#                (housing drop) — the same pair the FRONT preview already
#                draws its palm line with. For a rod grasped near its axis
#                this is the number that binds, always.
# PALM_RADIUS_MM how wide the body is. NOT MEASURED. Read off the outer
#                envelope of the base spheres in cuRobo's ur5e_gripper.yml
#                (+-51 x +-54 mm), which are inflated for planning and so
#                over-state the real plate. Checked 2026-08-22: the verdict
#                on a centred cylinder is IDENTICAL at 40 / 45 / 50 / 54 mm,
#                because a rod of D <= 60 never reaches the body's rim. Put
#                calipers on the real plate when convenient and correct this
#                one line; nothing else depends on it.
#
# NOT MODELLED, on purpose: the fingers and the pads. They must touch the
# object — that is their job — so any honest model of them collides with it
# at every valid grasp. cuRobo's own tool model leaves them out for exactly
# this reason (see the v4 comment in ur5e_gripper.yml). Residual exposure:
# a finger TIP alone catching the rod.
PALM_DROP_MM     = 86.69
PALM_RADIUS_MM   = 54.0
GRIPPER_CLEAR_MM = 5.0     # required air gap, rod surface to gripper body

# ---- the measured throat, which SUPERSEDES the two constants above --------
# probe_finger_throat.py drives the gripper to each calibrated close_rad and
# records, at every depth past the flange, how close the nearest piece of
# gripper comes to the tool axis. That is the same question the palm disc was
# guessing at, answered by measurement and over the WHOLE depth range rather
# than stopping at 86.69 — so when this file is present it is used instead,
# and the palm disc survives only as a fallback for machines without it.
#
# It exists because a Ø60 grasp at pad_dz +29.64 returned peak 4 counts while
# the same pose at +50.0 returned 13617: the rod's top end was jamming in the
# fingers, 15 mm BELOW where the palm disc stops looking. The probe put the
# Ø60 threshold at 112.0 mm against observations of 110.6 (dead) and 120.1
# (firm) — it predicted the failure it was written to explain, to 1.4 mm.
# The margin is BRACKETED BY OBSERVATION, not chosen for comfort:
#   Ø60 pad_dz +29.64  -0.06 mm  peak 4      DEAD
#   Ø26 pad_dz +16.31  +1.88 mm  lowest point of the six upright runs
#   Ø60 pad_dz +39.15  +3.13 mm  peak 14314  the calibration pose
# so the boundary lies between -0.06 and +1.88, and anything at or above 1.88
# would reject grasps that demonstrably worked. 1 mm sits inside that window
# and still allows for the probe reading mesh VERTICES, where the true surface
# can bulge slightly between them. Widen it only against new evidence.
THROAT_MARGIN_MM   = 1.0

# ---- paths for the config-save + Isaac-launch bridge (Stage B) ----
PROJECT   = os.path.expanduser("~/Paper3_Simulation")
CONFIG_JSON = os.path.join(PROJECT, "Data", "gui_config.json")
# measured gripper throat, written by probe_finger_throat.py (see above)
FINGER_THROAT_PATH = os.path.join(PROJECT, "Data", "finger_throat.json")
# 3-panel preview snapshot, written next to the config on every save. The
# collector copies it into the run folder so each run carries a picture of the
# grid design it was launched with (added 2026-08-04).
PREVIEW_PNG = os.path.join(PROJECT, "Data", "gui_preview.png")
# colour-scale policy shared with stitching.py / heatmaps.py / temporal_snapshots.py
PLOT_SCALE_JSON = os.path.join(PROJECT, "Data", "plot_scale.json")
# Reachability dry-runs used to land in Data/gui_run beside real data runs,
# so every test produced two folders. They now go to their own directory.
REACH_OUT_DIR = os.path.join(PROJECT, "Data", "reach_check")
CALIB_CONFIG_JSON = os.path.join(PROJECT, "Data", "gui_calib_config.json")
ISAAC_PY  = os.path.expanduser("~/isaacsim/python.sh")
COLLECT_PY = os.path.join(PROJECT, "sim", "collect_from_config.py")
EXAMPLES_DIR = os.path.expanduser("~/Paper3_Simulation/TSF-85/examples")
 
def grid_2d(nx, ny, step_mm, centered=False):
    """Return list of (dx, dy) offsets in mm for the grasp grid.
    dx runs along Z (up/down the face), dy along Y (across it).

    ANCHORED (centered=False, the original behaviour):
      offset (0,0) is the FIRST point (pt00) and the grid grows away from
      it. |n| = number of points on that axis; the SIGN picks direction:
        nx = +3 -> 0, +step, +2*step      nx = -3 -> 0, -step, -2*step

    CENTERED (centered=True):
      the entered pad offset sits in the MIDDLE and the grid mirrors both
      ways, so the initial grasp can be extrapolated in every direction.
      |n| = steps PER SIDE, so an axis holds 2|n|+1 points:
        nx = 2 -> -2, -1, 0, +1, +2  (5 points)
      Total points = (2|nx|+1) * (2|ny|+1) — 2,3 gives 35, not 12.
      Sign is ignored here: a mirrored grid has no direction.
      pt00 is still (0,0), the CENTRE: the collector starts there, and it
      is the frame the stitched training pair is anchored on.
    """
    def axis(n):
        n = int(n) if int(n) != 0 else 1
        sgn = 1.0 if n > 0 else -1.0
        return sgn * np.arange(abs(n)) * step_mm

    def axis_centered(n):
        k = abs(int(n))
        return (np.arange(-k, k + 1) * float(step_mm)) if k else np.zeros(1)

    xs = axis_centered(nx) if centered else axis(nx)
    ys = axis_centered(ny) if centered else axis(ny)
    # SERPENTINE (boustrophedon): reverse every other Y-column so the sweep
    # snakes instead of resetting to the far end of the next column.
    # Straight raster made the column wrap a hypot(Z_span, step) jump — 22.7 mm
    # for a 5-row column — and the grasp right after that jump landed ~2.2 mm
    # short of target in BOTH repeatability runs (A: pt10, B: pt06, the only
    # two 22.7 mm moves in the run). Short moves land to <0.01 mm.
    pts = []
    for j, gy in enumerate(ys):
        col = xs if (j % 2 == 0) else xs[::-1]
        for gx in col:
            pts.append((float(gx), float(gy)))
    if centered:
        # Move the centre to the front WITHOUT reordering the rest: raster
        # order keeps consecutive grasps adjacent, which the pad-to-pad
        # motion with joint-seed continuity depends on.
        for i, (gx, gy) in enumerate(pts):
            if abs(gx) < 1e-9 and abs(gy) < 1e-9:
                pts.insert(0, pts.pop(i))
                break
    return pts


def rotate_offsets(offs, roll_deg):
    """Turn a WORLD-aligned lattice into one aligned with the PAD's own edges.

    WHY (2026-08-09). grid_2d lays the pad centres out on a square lattice in
    world Y and Z. With an upright pad that is also the pad's own frame, so
    the sweep covers a clean rectangle. Roll the pad and the two stop
    agreeing: the footprints turn but the lattice does not, so the swept
    region comes out as a sheared, serrated quilt whose edges follow world Y/Z
    while every pad in it points 45 deg away. That is visible as the staircase
    envelope in the coverage panel of run_20260808_180415_obj0_pad45.

    Rotating the offsets by the same angle as the footprint puts the steps
    along the pad's width and height, so the swept region becomes a rectangle
    that is merely tilted — which is what a scan along the pad's edges means.

    SIGN. Deliberately the SAME matrix the preview uses to draw the rolled
    footprint a few hundred lines below (`_k @ [[c, s], [-s, c]]` on rows of
    [y, z]), so the lattice and the rectangles it carries can never disagree
    about which way positive roll turns.

    offs entries are (gx, gy) = (Z step, Y step) — grid_2d's order, kept.
    A roll of 0 returns the input unchanged, so upright runs are untouched."""
    if abs(float(roll_deg)) < 1e-9:
        return [(float(gx), float(gy)) for gx, gy in offs]
    c, s = np.cos(np.radians(float(roll_deg))), np.sin(np.radians(float(roll_deg)))
    out = []
    for gx, gy in offs:                      # gy along world Y, gx along world Z
        out.append((float(gy * s + gx * c),  # new Z
                    float(gy * c - gx * s)))  # new Y
    return out

def _tool_basis(roll_deg):
    """World columns of the TOOL frame at a given pad roll.

    The collector points the tool down with TOOL_DOWN_ROTVEC = [2.2214,
    2.2214, 0] — a pi turn about (1,1,0)/sqrt2 — which is the matrix
        R0 = [[0,1,0],[1,0,0],[0,0,-1]]
    i.e. tool x -> world +Y, tool y -> world +X, tool z (the APPROACH axis,
    flange towards the pads) -> world -Z. Roll is applied about the tool's
    OWN y (GRASP_ROT_AXIS="y"), which is world X, so it turns the pad inside
    its own face plane: R = R0 @ Ry(theta).

    VERIFIED against a measured number, not asserted. With this matrix the
    flange of a 20 deg rolled grasp sits 156.57*sin20 = 53.5 mm in -Y and
    156.57*(1-cos20) = 9.4 mm lower than the upright case — the two entries
    in the pivot-correction table of handoff v8 section 4.2, which were
    measured in Isaac. A sign error here would flip both.
    """
    th = np.radians(float(roll_deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([[0.0, 1.0, 0.0],
                     [c,   0.0, s  ],
                     [s,   0.0, -c ]])


def _rod_cloud(diameter_mm, length_mm, tilt_deg=0.0, tilt_axis="X",
               ax_mm=2.0, n_ang=24, cap_mm=2.0):
    """Points on the rod's SURFACE, mm, relative to the object centre.

    Sampled rather than solved: the rod and the gripper body are two
    flat-ended cylinders at an arbitrary relative angle, and the closed-form
    distance between those is a page of cases, every one of which is a place
    to be quietly wrong. A cloud at ~2 mm spacing tested against an
    analytic body is a few lines, and it is checked against the old scalar
    rule below to 0.01 mm at 0 deg.

    Both caps are included: a low grasp puts the rod's TOP CAP into the palm,
    and that is the whole point of the check."""
    R = float(diameter_mm) / 2.0
    L = float(length_mm)
    zs = np.arange(-L / 2.0, L / 2.0 + 1e-9, float(ax_mm))
    th = np.arange(int(n_ang)) * (2 * np.pi / int(n_ang))
    Z, T = np.meshgrid(zs, th, indexing="ij")
    side = np.stack([R * np.cos(T).ravel(), R * np.sin(T).ravel(),
                     Z.ravel()], axis=1)
    caps = [side]
    for zc in (-L / 2.0, L / 2.0):
        for rr in np.arange(0.0, R + 1e-9, float(cap_mm)):
            if rr <= 1e-9:
                caps.append(np.array([[0.0, 0.0, zc]])); continue
            n = max(6, int(2 * np.pi * rr / float(cap_mm)))
            a = np.arange(n) * (2 * np.pi / n)
            caps.append(np.stack([rr * np.cos(a), rr * np.sin(a),
                                  np.full(n, zc)], axis=1))
    pts = np.vstack(caps)
    t = np.radians(float(tilt_deg)); c, s = np.cos(t), np.sin(t)
    if tilt_axis == "X":
        R3 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    elif tilt_axis == "Y":
        R3 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    else:
        R3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    return pts @ R3.T


def grid_gripper_gap_mm(offs, pad_dz_mm, tool_offset_z_mm, pad_roll_deg,
                        cloud):
    """Gap between the rod's surface and the gripper body, one per grid point.

    Negative means the rod is INSIDE the body — the gripper would be driven
    into the object, which in Isaac shows up as PhysX resisting and the
    descent ending short of target, and in the data as a thin or empty map.

    offs are grid_2d/rotate_offsets entries, (gx, gy) = (world Z, world Y)
    step. The object's world position cancels: only the pad's offset from the
    object centre matters, so this works before any pose has been typed in.
    The pad's X is not a free variable — the collector centres the tool axis
    on the rod axis in X (x_fixed_centered)."""
    Rt = _tool_basis(pad_roll_deg)
    a = Rt[:, 2]
    T = float(tool_offset_z_mm)
    out = np.empty(len(offs), float)
    for i, (gx, gy) in enumerate(offs):
        pad = np.array([0.0, float(gy), float(pad_dz_mm) + float(gx)])
        ee = pad - T * a                       # flange, from the pad target
        q = (cloud - ee) @ Rt                  # rod points in the TOOL frame
        r = np.hypot(q[:, 0], q[:, 1])
        # body = {0 <= z <= PALM_DROP, r <= PALM_RADIUS}; positive = outside
        out[i] = np.maximum(np.maximum(-q[:, 2], q[:, 2] - PALM_DROP_MM),
                            r - PALM_RADIUS_MM).min()
    return out


def grid_pad_overhang_mm(offs, pad_dz_mm, pad_half_h, half_len_mm):
    """How far each grasp's pad hangs PAST the rod's end, mm. <=0 is on."""
    z = np.array([float(pad_dz_mm) + float(gx) for gx, _gy in offs])
    return np.abs(z) + float(pad_half_h) - float(half_len_mm)


def load_gripper_cloud(diameter_mm, path=None):
    """The gripper as points in the TOOL frame for one diameter, or None.

    Axes match _tool_basis exactly: x = ACROSS the pad, y = the CLOSING
    direction, z = depth past the flange. probe_finger_throat.py measures all
    three from the pads themselves, so no USD frame convention is assumed at
    either end.

    Pads are already excluded by the probe. THE FINGERS ARE EXCLUDED HERE, by
    the WORKING RADIUS: any point at or beyond half the measured pad
    separation is a holding surface, because that is the radius the jaw closes
    to on this diameter. What is left is only geometry that pokes INSIDE the
    opening — which is exactly what can stop the rod entering.

    They have to go. At a valid grasp the fingers sit a millimetre or two off
    the rod's surface, so leaving them in makes the nearest gripper point a
    holding surface at every pose and buries the jam signal. Measured on Ø26
    at the base height of the six upright runs: +3.12 mm with the fingers in,
    +13.88 mm with them out — while the Ø60 dead pose stays negative either
    way. Same reasoning as ur5e_gripper.yml's v4 comment: parts that must
    touch the object cannot be collision-checked against it.

    A depth cut was tried first and rejected: the finger's inner face runs at
    the working radius along its whole length, so no depth splits holding from
    throat.

    Why points and not the radial profile the same file also carries: the
    throat is NOT circular. It is narrow across the fingers and wide open the
    other way, so one radius per depth scores a rod stepped sideways along the
    open direction as if it had been driven into a finger. That version
    rejected a Ø60 grid at 25 deg roll which had actually grasped (peak 395),
    and shrank every upright grid to 3 points including Ø26, which worked at
    pad_dz +28.31 across six runs."""
    path = path or FINGER_THROAT_PATH
    try:
        with open(path, "r") as f:
            doc = json.load(f)
        d = doc.get("diameters", {})
        ent = d.get(f"{float(diameter_mm):.1f}") or d.get(str(float(diameter_mm)))
        if ent is None:
            return None
        pts = np.asarray(ent.get("cloud_xyz_mm", []), float)
        if pts.ndim != 2 or not len(pts):
            return None
        sep = float(ent.get("pad_separation_mm", 0.0))
        if sep > 0.0:
            lat = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[lat < sep / 2.0]
        return pts if len(pts) else None
    except Exception:
        return None


def grid_gripper_cloud_gap_mm(offs, pad_dz_mm, tool_offset_z_mm, pad_roll_deg,
                              gcloud, diameter_mm, length_mm,
                              obj_tilt_deg=0.0, obj_tilt_axis="X"):
    """Smallest distance from the GRIPPER to the ROD, one value per grid point.

    Negative means a piece of gripper is inside the object — the rod would jam
    against the fingers and hold the pads off, which reads as a dead map
    rather than as a crash.

    The rod is solved analytically (exact for a finite cylinder, tilt
    included) and the gripper is sampled, which is the right way round: the
    gripper is the awkward shape and the rod is the simple one. Roll is
    handled for free, because the whole cloud rides the tool frame."""
    Rt = _tool_basis(pad_roll_deg)
    a_tool = Rt[:, 2]
    T = float(tool_offset_z_mm)
    G = gcloud @ Rt.T                          # gripper in the object frame
    pads = np.array([[0.0, float(gy), float(pad_dz_mm) + float(gx)]
                     for gx, gy in offs], float)
    ee = pads - T * a_tool                     # (M,3) flange per grid point
    P = ee[:, None, :] + G[None, :, :]         # (M,N,3)

    th = np.radians(float(obj_tilt_deg))
    c, s = np.cos(th), np.sin(th)
    if obj_tilt_axis == "X":
        ax = np.array([0.0, -s, c])
    elif obj_tilt_axis == "Y":
        ax = np.array([s, 0.0, c])
    else:
        ax = np.array([0.0, 0.0, 1.0])

    R, H = float(diameter_mm) / 2.0, float(length_mm) / 2.0
    t = P @ ax                                 # axial coord along the rod
    radial = np.linalg.norm(P - t[..., None] * ax, axis=2)
    # SIGNED, and the sign is the whole point. Clamping both terms at zero —
    # as the first version did — reports a point INSIDE the rod as 0.00 mm
    # rather than as interference, so the Ø60 dead pose (exactly one gripper
    # point inside the object) scored the same as a graze and passed.
    outside = np.sqrt(np.maximum(radial - R, 0.0) ** 2 +
                      np.maximum(np.abs(t) - H, 0.0) ** 2)
    inside = np.minimum(R - radial, H - np.abs(t))
    d = np.where(outside > 0.0, outside, -np.maximum(inside, 0.0))
    return d.min(axis=1)


def design_grid(diameter_mm, length_mm, tool_offset_z_mm, step_mm=6.0,
                pad_roll_deg=0.0, indent_mm=2.4, across_margin_mm=3.0,
                palm_clear_mm=5.0, canvas_mm=96.0, obj_tilt_deg=0.0,
                obj_tilt_axis="X"):
    """Choose the two grid counts and the base pad height from the OBJECT.

    Returns (ok, result_dict, [reason lines]). Every number it picks is
    derived from a stated limit, and every limit is reported, so the choice
    can be read off the screen instead of trusted.

    IT RETURNS n_across AND n_along, NOT nx AND ny (2026-08-22). Those two
    names meant opposite things at the two ends of this handover and nobody
    could see it: this function derived one count from the ACROSS band and
    one from the ALONG window, while grid_2d's nx steps along world Z (UP
    the rod) and its ny across world Y. do_design_grid passed them straight
    through, so every automatic grid ran transposed. Proved on
    run_20260821_165542 (D42, 25 deg): the observed vertical excursion of
    29.358 mm equals 24*cos25 + 18*sin25 to 0.001 mm, which only holds if
    the count derived from the across band was driving the vertical steps.
    Costs, both real: the vertical sweep overran its own window (palm), and
    on the upright runs the across sweep reached +-18 mm against a designed
    +-15.53, leaving 0.53 mm of pad inside the contact band at the outermost
    column — near-empty maps. The names now say which is which and the
    caller must map them deliberately.

    WHY THIS EXISTS. nx, ny, step and pad_dz were typed by hand, and a wrong
    guess is expensive: on run_20260808_234035 the top grasps sat at
    pad_dz = +56 mm with a 45 deg pad whose half-height is 20.86 mm, putting
    the pad's top edge at 76.9 mm against a rod that ends at 70 mm — the pad
    was hanging ~7 mm off the end, which is part of why those grasps came back
    thin. Nothing in the pipeline could have caught that.

    THE FOUR LIMITS.

    ACROSS the rod. The pads close to a calibrated gap, so contact exists only
    where the cylinder's surface has not receded further than the indentation.
    A surface point at offset y sits back by (D/2 - sqrt((D/2)^2 - y^2)), so
    contact needs that <= indent, giving a band of half-width
        band = sqrt(indent * (D - indent))
    which for D = 26 and indent = 2.4 mm gives 7.53 mm — i.e. a 15.06 mm band,
    matching the 15.0 mm width fitted from the upright run's across-pad
    profile. The pad centre may travel until its near edge is about to leave
    that band: PAD_W/2 + band, less a margin so the last grasp is not the
    marginal one.

    ALONG the rod, three limits, whichever binds first:
      rod end  the pad must stay ON the object: |dz| + pad_half_h <= L/2
      palm     the palm sits (TOOL_OFFSET_Z - 86.69) mm above the pad centre,
               so grasping low on a tall rod drives it into the top of the
               object. This is why every run so far used pad_dz ~ +40.
      canvas   the pair canvas is pinned: pad_h_eff + 2*ny*step <= 96 mm.

    A rolled pad is handled by using its BOUNDING half-height, the same
    quantity stitching.pad_half_extents computes, so the designer and the
    stitcher agree about how much room a rolled pad needs.

    THE FIFTH LIMIT: EVERY POINT IS CHECKED, NOT THE WINDOW (2026-08-22).
    The four limits above are scalars applied to the base height, and two
    things slip through them once the pad is rolled. First, the palm limit
    measured the lever arm as if the tool were still vertical; a rolled tool
    holds its palm only (TOOL_OFFSET_Z - 86.69)*cos(roll) above the pad, so
    the clearance was over-stated by 6.5 mm at 25 deg and 20 mm at 45 deg.
    Second, when the grid steps in the PAD's frame the ACROSS steps carry a
    vertical component of n_across*step*sin(roll) that the along window
    never saw — another 10.1 mm on that D42 run. Together with the transpose
    above, 7 of its 63 points put the rod INSIDE the gripper body, worst
    -3.15 mm, and 14 sat inside the 5 mm margin.

    So the grid is now verified point by point against the real gripper
    body: the rod's surface versus a cylinder of PALM_RADIUS_MM reaching
    PALM_DROP_MM past the flange, at each point's own rolled tool pose. If a
    grid fails, the height is re-centred first and the counts are shrunk
    only if that is not enough — and if nothing works it REFUSES, because a
    grid that collides quietly is worse than no grid. At 0 deg roll this
    reproduces the old scalar palm rule to 0.01 mm, so upright designs are
    unchanged.

    STEP IS NOT CHOSEN. It is pinned by the caller (6 mm) so that coverage
    density is identical on every object; only the counts and the height
    adapt.
    """
    D, L = float(diameter_mm), float(length_mm)
    step = float(step_mm)
    why = []

    if step <= 0:
        return False, {}, ["step must be > 0"]
    if D <= 2 * indent_mm:
        return False, {}, [f"object diameter {D:.1f} mm is too small to "
                           f"indent {indent_mm:.1f} mm"]

    # ---- pad footprint, upright or rolled (same rule as the stitcher) ------
    c, s = (abs(np.cos(np.radians(pad_roll_deg))),
            abs(np.sin(np.radians(pad_roll_deg))))
    pad_hw = (PAD_W * c + PAD_H * s) / 2.0        # half-width  of the bbox
    pad_hh = (PAD_W * s + PAD_H * c) / 2.0        # half-height of the bbox
    if abs(pad_roll_deg) > 1e-6:
        why.append(f"pad rolled {pad_roll_deg:+.1f} deg -> footprint bbox "
                   f"{2*pad_hw:.1f} x {2*pad_hh:.1f} mm "
                   f"(upright would be {PAD_W:.0f} x {PAD_H:.0f})")

    # ---- ACROSS -----------------------------------------------------------
    band = float(np.sqrt(indent_mm * (D - indent_mm)))
    across_max = pad_hw + band - across_margin_mm
    n_across = int(np.floor(max(0.0, across_max) / step))
    why.append(f"ACROSS: contact band half-width = sqrt({indent_mm:.1f} x "
               f"({D:.1f} - {indent_mm:.1f})) = {band:.2f} mm "
               f"(a {2*band:.1f} mm band)")
    why.append(f"        pad centre may reach {pad_hw:.1f} + {band:.2f} - "
               f"{across_margin_mm:.1f} = {across_max:.2f} mm "
               f"-> n_across = floor({across_max:.2f}/{step:.1f}) "
               f"= {n_across}")

    # ---- ALONG: the feasible window for the pad centre ---------------------
    # palm_z = pad_z + (TOOL_OFFSET_Z - 86.69); it must clear the rod top.
    palm_above_pad = float(tool_offset_z_mm) - 86.69
    dz_lo = (L / 2.0) - palm_above_pad + palm_clear_mm   # lowest allowed pad_dz
    dz_hi = (L / 2.0) - pad_hh                           # highest allowed
    why.append(f"ALONG : palm sits {palm_above_pad:.2f} mm above the pad "
               f"(TOOL_OFFSET_Z {tool_offset_z_mm:.2f} - 86.69)")
    why.append(f"        palm clearance >= {palm_clear_mm:.1f} mm -> "
               f"pad_dz >= {dz_lo:+.2f} mm")
    why.append(f"        pad must stay on the rod -> "
               f"pad_dz <= {L/2:.1f} - {pad_hh:.2f} = {dz_hi:+.2f} mm")

    if dz_lo > dz_hi:
        return False, {}, why + [
            f"NO VALID HEIGHT: the palm needs pad_dz >= {dz_lo:+.1f} mm but "
            f"the rod end needs pad_dz <= {dz_hi:+.1f} mm. This object is too "
            f"long for the gripper to reach without the palm hitting its top "
            f"(the window closes when L/2 > {palm_above_pad - palm_clear_mm + pad_hh:.1f} mm)."]

    span = dz_hi - dz_lo                       # total travel available, mm
    nl_travel = int(np.floor((span / 2.0) / step))
    nl_canvas = int(np.floor((canvas_mm - 2 * pad_hh) / (2 * step)))
    n_along = max(0, min(nl_travel, nl_canvas))
    bound = "rod end / palm" if nl_travel <= nl_canvas else "pair canvas"
    why.append(f"        window is {span:.2f} mm wide -> n_along <= "
               f"{nl_travel} by travel, <= {nl_canvas} by the "
               f"{canvas_mm:.0f} mm canvas -> n_along = {n_along} "
               f"(bound by {bound})")

    pad_dz = (dz_lo + dz_hi) / 2.0             # centre the sweep in the window
    why.append(f"        base pad height = midpoint of the window = "
               f"{pad_dz:+.2f} mm")

    # ---- VERIFY EVERY POINT against the real gripper body ------------------
    # Everything above is a scalar rule about the base height. This is the
    # part that actually looks at the grid that will be run.
    cloud = _rod_cloud(D, L, obj_tilt_deg, obj_tilt_axis)
    gcloud = load_gripper_cloud(D)
    if gcloud is not None:
        clear_mm = THROAT_MARGIN_MM
        why.append(f"CHECK : every point tested against the MEASURED gripper "
                   f"({len(gcloud)} points, finger_throat.json), margin "
                   f"{clear_mm:.1f} mm")
    else:
        clear_mm = palm_clear_mm
        why.append(f"CHECK : finger_throat.json NOT FOUND — falling back to "
                   f"the palm disc, which stops at {PALM_DROP_MM:.1f} mm and "
                   f"CANNOT see a rod jammed in the fingers. Run "
                   f"probe_finger_throat.py.")

    def _offsets(n_across_, n_along_):
        # THE HANDOVER, stated once and in one place: grid_2d's FIRST count
        # steps along world Z (up the rod), its SECOND across world Y.
        offs = grid_2d(n_along_, n_across_, step, centered=True)
        return rotate_offsets(offs, pad_roll_deg)

    def _gaps(offs, dz_):
        if gcloud is not None:
            return grid_gripper_cloud_gap_mm(
                offs, dz_, tool_offset_z_mm, pad_roll_deg, gcloud, D, L,
                obj_tilt_deg, obj_tilt_axis)
        return grid_gripper_gap_mm(offs, dz_, tool_offset_z_mm,
                                   pad_roll_deg, cloud)

    def _worst(n_across_, n_along_, dz_):
        offs = _offsets(n_across_, n_along_)
        gap = _gaps(offs, dz_)
        over = grid_pad_overhang_mm(offs, dz_, pad_hh, L / 2.0)
        return float(gap.min()), float(over.max()), gap, over

    def _lowest_clear_dz(n_across_, n_along_):
        """Smallest base height at which no point touches the gripper.
        Raising the tool always carries the rod's top DEEPER into free space
        (by cos(roll) per mm), so the gap rises monotonically with dz and a
        bisection is exact to the tolerance. Returns None if even the top of
        the search range collides."""
        lo, hi = dz_lo - 60.0, dz_hi + 60.0
        if _worst(n_across_, n_along_, hi)[0] < clear_mm:
            return None
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _worst(n_across_, n_along_, mid)[0] >= clear_mm:
                hi = mid
            else:
                lo = mid
            if hi - lo < 0.01:
                break
        return hi

    # Shrink the LESS informative axis first. Below 45 deg the along steps
    # dominate the vertical excursion (cos > sin) and across coverage is the
    # direction the completion model actually learns from, so along goes
    # first; past 45 deg the roles swap and so does the order.
    _c, _s = (abs(np.cos(np.radians(pad_roll_deg))),
              abs(np.sin(np.radians(pad_roll_deg))))
    shrink_first_along = (_c >= _s)

    trimmed = []
    dz_mid = pad_dz                     # what the four scalar limits wanted
    while True:
        need = _lowest_clear_dz(n_across, n_along)
        if need is not None:
            # The window's midpoint stays the preference — it is what keeps
            # the sweep centred on the rod. Only if it fails is the height
            # moved, and then only to the nearest value that works: up to
            # `need` if the palm is the problem, down to the highest height
            # that keeps every pad on the rod if the end is.
            offs = _offsets(n_across, n_along)
            gx_hi = max(gx for gx, _gy in offs)
            # 0.01 mm off the rim, not on it: clipping to the exact boundary
            # lands the last pad's bounding box on the rod's end face and
            # floating point then decides whether that counts as "off". The
            # back-off is 3x below the pipeline's own +-0.03 mm placement
            # accuracy, so it costs nothing real and makes the answer stable.
            dz_end = (L / 2.0) - pad_hh - gx_hi - 0.01
            if need <= dz_end + 1e-9:
                chosen = float(np.clip(dz_mid, need, dz_end))
                g, o, _, _ = _worst(n_across, n_along, chosen)
                if g >= clear_mm and o <= 1e-6:
                    if abs(chosen - dz_mid) > 0.01:
                        why.append(
                            f"        base height moved {dz_mid:+.2f} -> "
                            f"{chosen:+.2f} mm: the midpoint put the rod "
                            + ("into the gripper" if chosen > dz_mid
                               else "a pad off the rod end"))
                    pad_dz = chosen
                    break
        # cannot be fixed by height alone -> drop a step from one axis
        if shrink_first_along and n_along > 0:
            n_along -= 1; trimmed.append("n_along")
        elif n_across > 0:
            n_across -= 1; trimmed.append("n_across")
        elif n_along > 0:
            n_along -= 1; trimmed.append("n_along")
        else:
            g, o, _, _ = _worst(0, 0, pad_dz)
            return False, {}, why + [
                "REFUSED: even a single grasp at this height cannot be made "
                f"safe (gripper gap {g:+.1f} mm, pad overhang {o:+.1f} mm).",
                "The rod is too long for the gripper to reach this low, or "
                "the roll swings the palm into it. Shorten the object, "
                "reduce the roll, or grasp higher."]

    gap, over = _worst(n_across, n_along, pad_dz)[2:]
    if trimmed:
        why.append(f"        POINT CHECK trimmed {len(trimmed)} step(s) "
                   f"({', '.join(sorted(set(trimmed)))}): the four limits "
                   f"above are scalars, and the rolled grid's corners reach "
                   f"further than any of them describe")
    why.append(f"        worst rod-to-gripper gap {gap.min():+.2f} mm "
               f"(need {clear_mm:.1f}); worst pad overhang past the rod "
               f"end {over.max():+.2f} mm (need <= 0)")

    # ---- report what the choice actually costs -----------------------------
    used_y = 2 * pad_hw + 2 * n_across * step
    used_z = 2 * pad_hh + 2 * n_along * step
    n_pts = (2 * n_across + 1) * (2 * n_along + 1)
    cov_y = (2 * pad_hw) / step
    cov_z = (2 * pad_hh) / step
    why.append(f"RESULT: {2*n_across+1} across x {2*n_along+1} along = "
               f"{n_pts} points at {step:.1f} mm, ~{n_pts*2.0:.0f} min")
    why.append(f"        swept area {used_y:.1f} x {used_z:.1f} mm of the "
               f"{canvas_mm:.0f} mm canvas; density ~{cov_y:.1f} x "
               f"{cov_z:.1f} grasps per cell")

    bad = []
    if used_y > canvas_mm or used_z > canvas_mm:
        bad.append(f"swept area {used_y:.1f} x {used_z:.1f} mm exceeds the "
                   f"{canvas_mm:.0f} mm pair canvas")
    if n_across == 0 and n_along == 0:
        bad.append("no room to move in either direction — a single grasp is "
                   "not a sweep")
    if bad:
        return False, {}, why + bad

    return True, {"n_across": n_across, "n_along": n_along,
                  "step_mm": step, "pad_dz_mm": pad_dz,
                  "n_points": n_pts, "band_mm": 2 * band,
                  "swept_mm": (used_y, used_z), "bound_by": bound,
                  "dz_window": (dz_lo, dz_hi),
                  "gap_mm": float(gap.min()),
                  "overhang_mm": float(over.max()),
                  "trimmed": trimmed}, why


class CockpitGUI:
    def __init__(self, root):
        self.root = root
        root.title("Paper 3 — Collection Cockpit (Stage A)")
        self._show_measured = False   # green 'measured pad' only after Load result

        # defaults in mm (from our proven scene)
        self.vars = {
            "obj_x": tk.StringVar(value="-268.06"),
            "obj_y": tk.StringVar(value="199.0"),
            "obj_z": tk.StringVar(value="1052.2"),
            "obj_diam": tk.StringVar(value="26.0"),         # object diameter mm
            "obj_len": tk.StringVar(value="140.0"),         # object length   mm
            "obj_tilt_deg": tk.StringVar(value="0.0"),      # 0 = standing
            "obj_tilt_axis": tk.StringVar(value="X"),       # tilt about this axis
            # pad pose = offset from OBJECT CENTER (mm). X is fixed (centered grasp).
            "pad_dy": tk.StringVar(value="0.0"),
            "pad_dz": tk.StringVar(value="0.0"),
            # In-plane ROLL of the pad about its own face normal (deg).
            # Emitted as GRASP_ROT_DEG; the collector applies it about the
            # TOOL-LOCAL Y axis (verified 2026-08-03: Y rolls the pad in its
            # face plane; X tips the gripper off the rod).
            "pad_rot": tk.StringVar(value="0.0"),
            "grid_nx":  tk.StringVar(value="2"),
            "grid_ny":  tk.StringVar(value="3"),
            "grid_step": tk.StringVar(value="8.0"),   # mm
            # grid mirrors around the entered pad offset instead of starting there
            "grid_centered": tk.BooleanVar(value=False),
            "grid_pad_frame": tk.BooleanVar(value=True),
            "export_anchors": tk.BooleanVar(value=True),
            "headless": tk.BooleanVar(value=False),   # False = show Isaac window
            "calib_headless": tk.BooleanVar(value=False),  # Calibrate tab headless toggle
            "calib_dz": tk.StringVar(value="0.0"),  # Calibrate pad Z offset (Y stays centered)
            # finger-joint angle to squeeze to during calibration.
            # blank = let the collector decide (stored value, else 26 mm value)
            "calib_close_rad": tk.StringVar(value=""),
            "stitch_want_gsr": tk.BooleanVar(value=False),
            # Stitch-tab: append the blob-axis metric self-test to the report
            "blob_selftest": tk.BooleanVar(value=False),
            # Pre-check every grid point before moving. ON by default: it costs
            # a plan per point but stops the arm attempting poses it cannot
            # reach. Turn OFF while iterating on a config you have already
            # proven, to save that time. Emitted as GRASP_REACH_CHECK.
            "reach_check": tk.BooleanVar(value=True),
            # Contact-aware closing. OFF by default everywhere: the nine
            # hand-tuned calibration entries are known good, and a new method
            # must be compared against them before it replaces them.
            "contact_close": tk.BooleanVar(value=False),   # kept: run cmds read it
            # fixed | contact | both. "both" measures the SAME diameter twice
            # in one sitting, one grasp per mode, into two separate files.
            "calib_mode": tk.StringVar(value="fixed angle"),
            # WHICH calibration file the grid designer reads AND the collector
            # runs with. They must agree: the offsets differ by 0.26 mm on D26
            # between fixed and contact, and a mismatch puts every pad that far
            # off with nothing to show it.
            "calib_source": tk.StringVar(value="main"),
            # Per-frame mesh CSVs: ~98 MB per grasp, and nothing downstream
            # reads them. ON here (single runs are where you might want the
            # diagnostic), OFF in the batch tab.
            "log_mesh": tk.BooleanVar(value=True),
            "batch_log_mesh": tk.BooleanVar(value=False),
            "contact_signal": tk.StringVar(value="deformation (mm)"),
            "contact_target": tk.StringVar(value="1.00"),
            # Emitted as GRASP_TOOL_COLLISION. Default ON: the collector's own
            # log says that without it "a plan that sweeps the FINGERS or the
            # PADS through the object will still look safe", and every run so
            # far was made with it off.
            "tool_collision": tk.BooleanVar(value=True),
            # Stitch-tab: fit the contact band width from the run's own maps
            "blob_fit_width": tk.BooleanVar(value=False),
            # ---- colour-scale policy (2026-08-04) ----
            # Written to Data/plot_scale.json and read by stitching.py,
            # heatmaps.py and temporal_snapshots.py, so every figure in the
            # project is scaled the same way.
            "scale_shared": tk.BooleanVar(value=True),
            "scale_fixed":  tk.StringVar(value=""),   # blank = not fixed  # Stitch-tab: include GSR in validation
        }

        # ---- Notebook: tab 1 = collection cockpit, tab 2 = stitching ----
        self.nb = ttk.Notebook(root)
        self.nb.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.tab_collect = ttk.Frame(self.nb)
        self.tab_batch = ttk.Frame(self.nb)
        self.tab_stitch = ttk.Frame(self.nb)
        self.tab_calib = ttk.Frame(self.nb)
        self.nb.add(self.tab_collect, text="Collection")
        self.nb.add(self.tab_calib, text="Calibrate")
        self.nb.add(self.tab_batch, text="Batch")
        self.nb.add(self.tab_stitch, text="Stitching (Block 2)")

        self._build_inputs()
        self._build_preview()
        self._build_batch_tab()
        self._build_stitch_tab()
        self._build_calib_tab()
        self.refresh()

    def _build_inputs(self):
        # The left panel grew past the window height and had no way to reach
        # the buttons at the bottom. Put it inside a Canvas with a vertical
        # scrollbar: the panel keeps its natural width, only scrolls in Y.
        _outer = ttk.Frame(self.tab_collect)
        _outer.grid(row=0, column=0, sticky="nsew")
        _outer.rowconfigure(0, weight=1)
        _outer.columnconfigure(0, weight=1)

        _cv = tk.Canvas(_outer, highlightthickness=0, borderwidth=0)
        _sb = ttk.Scrollbar(_outer, orient="vertical", command=_cv.yview)
        _cv.configure(yscrollcommand=_sb.set)
        _cv.grid(row=0, column=0, sticky="nsew")
        _sb.grid(row=0, column=1, sticky="ns")

        frm = ttk.Frame(_cv, padding=10)
        _cv.create_window((0, 0), window=frm, anchor="nw")

        def _fit(_e=None):
            _cv.configure(scrollregion=_cv.bbox("all"))
            # never clip horizontally: canvas matches the panel's needed width
            _cv.configure(width=frm.winfo_reqwidth())
        frm.bind("<Configure>", _fit)

        def _wheel(e):
            if getattr(e, "num", None) == 4 or getattr(e, "delta", 0) > 0:
                _cv.yview_scroll(-1, "units")
            elif getattr(e, "num", None) == 5 or getattr(e, "delta", 0) < 0:
                _cv.yview_scroll(1, "units")

        def _grab(_e=None):        # only while the pointer is over this panel,
            for _b in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                _cv.bind_all(_b, _wheel)

        def _release(_e=None):     # so the plot area keeps its own wheel
            for _b in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                _cv.unbind_all(_b)

        for _w in (_cv, frm):
            _w.bind("<Enter>", _grab)
            _w.bind("<Leave>", _release)

        self._inputs_canvas = _cv          # kept for later resizing
        r = 0

        # ---------------- SESSION FOLDER (added 2026-08-04) ----------------
        # One folder per test. Before this, the reachability dry-run and the
        # real run each minted their own timestamp, so a single test produced
        # two folders minutes apart. Now the GUI decides the name ONCE and
        # passes it to both via GRASP_RUN_DIR, and that folder is also the
        # default for loading configs and for re-plotting heatmaps/stitches.
        ttk.Label(frm, text="SESSION FOLDER (one per test)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2,
                                              sticky="w"); r += 1
        self.session_lbl = ttk.Label(frm, text="(none)", foreground="#06a",
                                     wraplength=430, justify="left")
        self.session_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        _sf = ttk.Frame(frm); _sf.grid(row=r, column=0, columnspan=2,
                                       sticky="ew", pady=(2, 6)); r += 1
        ttk.Button(_sf, text="New session (stamp + angles)",
                   command=self.new_session).pack(side="left")
        ttk.Button(_sf, text="Use existing...",
                   command=self.pick_session).pack(side="left", padx=4)
        ttk.Button(_sf, text="Open folder",
                   command=self.open_session_folder).pack(side="left")
        ttk.Button(_sf, text="Data folder...",
                   command=self.set_run_root).pack(side="left", padx=4)
        self.run_root_lbl = ttk.Label(frm, text="", foreground="#555",
                                      wraplength=430, justify="left")
        self.run_root_lbl.grid(row=r, column=0, columnspan=2,
                               sticky="w"); r += 1
        ttk.Label(frm, text="(the folder name is fixed when you press New "
                            "session; press it again after changing angles)",
                  foreground="#888", wraplength=430,
                  justify="left").grid(row=r, column=0, columnspan=2,
                                       sticky="w"); r += 1
        self._run_root = self._load_run_root()
        self._refresh_root_label()

        ttk.Label(frm, text="OBJECT pose (world, mm)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        for key, lab in [("obj_x", "x"), ("obj_y", "y"), ("obj_z", "z")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="tilt (deg)").grid(row=r, column=0, sticky="e")
        for key, lab in [("obj_diam", "diameter (mm)"), ("obj_len", "length (mm)")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w")
            e.bind("<Return>", lambda ev: self.refresh()); r += 1
        self.obj_cal_lbl = ttk.Label(frm, text="", foreground="#555",
                                     wraplength=250)
        self.obj_cal_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        e = ttk.Entry(frm, textvariable=self.vars["obj_tilt_deg"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="tilt axis").grid(row=r, column=0, sticky="e")
        ob = ttk.Combobox(frm, textvariable=self.vars["obj_tilt_axis"], width=12,
                          values=["X", "Y", "Z"])
        ob.grid(row=r, column=1, sticky="w"); ob.bind("<<ComboboxSelected>>", lambda ev: self.refresh()); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

        ttk.Label(frm, text="PAD offset from object center (mm)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, text="x").grid(row=r, column=0, sticky="e")
        xe = ttk.Entry(frm, width=12, state="disabled")
        xe.grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(frm, text="(x fixed: centered grasp)", foreground="#888").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        for key, lab in [("pad_dy", "y"), ("pad_dz", "z")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="rotation (deg)").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["pad_rot"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="(pad roll in its own face plane; tool-local Y)",
                  foreground="#888").grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

        ttk.Label(frm, text="GRID (anchored at pad offset; sign = direction)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, text="n steps X").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["grid_nx"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="n steps Y").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["grid_ny"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="step (mm)").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["grid_step"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Checkbutton(frm, text="centered grid (mirror both sides; n = steps PER SIDE)",
                        variable=self.vars["grid_centered"],
                        command=self.refresh).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Checkbutton(frm, text="step along PAD axes (rolled pad sweeps a "
                                  "tilted rectangle, not a sheared quilt)",
                        variable=self.vars["grid_pad_frame"],
                        command=self.refresh).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Design grid from object geometry",
                   command=self.do_design_grid).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(4, 2)); r += 1
        self.design_lbl = ttk.Label(frm, text="", foreground="#555",
                                    wraplength=250, justify="left")
        self.design_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.grid_count = ttk.Label(frm, text="", foreground="#555")
        self.grid_count.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Button(frm, text="Update Preview", command=self.refresh).grid(
            row=r, column=0, columnspan=2, pady=(12, 4), sticky="ew"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Checkbutton(frm, text="Run headless (no Isaac window)",
                        variable=self.vars["headless"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Checkbutton(frm, text="Pre-check reachability before moving "
                                  "(slower, skips points the arm cannot reach)",
                        variable=self.vars["reach_check"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        _cf = ttk.Frame(frm)
        _cf.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(_cf, text="calibration file:").grid(row=0, column=0, sticky="w")
        _cs = ttk.Combobox(_cf, textvariable=self.vars["calib_source"],
                           values=["main", "fixed", "contact"],
                           state="readonly", width=10)
        _cs.grid(row=0, column=1, sticky="w", padx=(4, 6))
        self.calsrc_lbl = ttk.Label(_cf, text="", foreground="#555")
        self.calsrc_lbl.grid(row=0, column=2, sticky="w")
        _cs.bind("<<ComboboxSelected>>", lambda _e: self._sync_cal_source())

        ttk.Checkbutton(frm, text="Save per-frame mesh CSVs (~98 MB/grasp; "
                                  "nothing downstream reads them)",
                        variable=self.vars["log_mesh"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Checkbutton(frm, text="Model the gripper BODY in collision "
                                  "(ur5e_gripper.yml — fingers and pads are "
                                  "never modelled, they must touch)",
                        variable=self.vars["tool_collision"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Save Config", command=self.save_config).grid(
            row=r, column=0, columnspan=2, pady=4, sticky="ew"); r += 1
        ttk.Button(frm, text="Check Reachability (no motion)",
                   command=self.save_and_show_reach_cmd).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
        ttk.Button(frm, text="Load reachability result",
                   command=self.load_reachability).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
        ttk.Button(frm, text="Save + Show Run Command", command=self.save_and_show_cmd).grid(
            row=r, column=0, columnspan=2, pady=4, sticky="ew"); r += 1

        # ---- Save / Load a named EXPERIMENT (full recipe: pose, tilt, grid) ----
        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(frm, text="EXPERIMENT recipe:", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Save Experiment As…", command=self.save_experiment).grid(
            row=r, column=0, columnspan=2, pady=2, sticky="ew"); r += 1
        ttk.Button(frm, text="Load Experiment…", command=self.load_experiment).grid(
            row=r, column=0, columnspan=2, pady=2, sticky="ew"); r += 1

        self.status = ttk.Label(frm, text="", foreground="#06a", wraplength=190)
        self.status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 0)); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Label(frm, text="AFTER the run:", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        # plot source: newest run (default) OR a saved folder you pick
        # ---------------- COLOUR SCALE (added 2026-08-04) ----------------
        ttk.Label(frm, text="COLOUR SCALE (all heatmaps / stitches / temporal)",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2,
                                             sticky="w", pady=(8, 0)); r += 1
        ttk.Checkbutton(frm, text="one scale across the whole run",
                        variable=self.vars["scale_shared"],
                        command=self.save_plot_scale).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        _cs = ttk.Frame(frm); _cs.grid(row=r, column=0, columnspan=2,
                                       sticky="w"); r += 1
        ttk.Label(_cs, text="fixed max (blank = off):").pack(side="left")
        _ce = ttk.Entry(_cs, textvariable=self.vars["scale_fixed"], width=8)
        _ce.pack(side="left", padx=4)
        _ce.bind("<Return>", lambda ev: self.save_plot_scale())
        ttk.Button(_cs, text="Apply", command=self.save_plot_scale).pack(side="left")
        ttk.Label(frm, text="(fixed max makes DIFFERENT tests comparable; "
                            "2400 matches Paper 2's tactile counts)",
                  foreground="#888", wraplength=430,
                  justify="left").grid(row=r, column=0, columnspan=2,
                                       sticky="w"); r += 1

        ttk.Button(frm, text="Plot from folder…", command=self.choose_plot_folder).grid(
            row=r, column=0, pady=2, sticky="ew")
        ttk.Button(frm, text="Use newest", command=self.use_newest_run).grid(
            row=r, column=1, pady=2, sticky="ew"); r += 1
        self.plot_src_lbl = ttk.Label(frm, text="plot source: newest run (auto)",
                                      foreground="#666", wraplength=190)
        self.plot_src_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Show Heatmaps (s1+s2)", command=self.show_heatmaps).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Show Pose History", command=self.show_pose_history).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Make Verification Plots", command=self.make_verifications).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Show Temporal Snapshots (4-step)", command=self.show_temporal).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Check Pad Truth (safety)", command=self.show_pad_truth).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1

        self.info = ttk.Label(frm, text="", foreground="#0a6", wraplength=190)
        self.info.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

    def _build_preview(self):
        self.fig = Figure(figsize=(11.0, 4.4), dpi=100)
        self.ax_top   = self.fig.add_subplot(1, 3, 1)
        self.ax_front = self.fig.add_subplot(1, 3, 2)
        self.ax_3d    = self.fig.add_subplot(1, 3, 3, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_collect)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.tab_collect.columnconfigure(1, weight=1)
        self.tab_collect.rowconfigure(0, weight=1)

    def _sync_cal_source(self):
        """Say which file is in force, and refresh the design that depends on it."""
        p = self._cal_path()
        want = self.vars["calib_source"].get()
        got = os.path.basename(p)
        fell_back = (want != "main" and not got.endswith(f"_{want}.json"))
        self.calsrc_lbl.config(
            text=(f"{got}" + ("  (MISSING — using main)" if fell_back else "")),
            foreground="#a00" if fell_back else "#555")
        try:
            self.refresh()
        except Exception:
            pass

    def _cal_path(self):
        """The calibration file currently selected, main if it is missing."""
        base = os.path.join(PROJECT, "Data", "pad_offset_calibration.json")
        sfx = {"main": "", "fixed": "_fixed",
               "contact": "_contact"}.get(self.vars["calib_source"].get(), "")
        if sfx:
            p = base.replace(".json", f"{sfx}.json")
            if os.path.isfile(p):
                return p
        return base

    def _cal_entry(self):
        """Calibration entry for the current diameter, or None."""
        try:
            with open(self._cal_path()) as f:
                cal = json.load(f)
            return cal.get(f"{CYL_D:.1f}")
        except Exception:
            return None

    def _newest_probe(self):
        """Newest pad_truth_probe.json (a real measured grasp), or None."""
        import glob
        try:
            cands = glob.glob(os.path.join(PROJECT, "Data", "gui_run", "*",
                                           "pad_truth_probe.json"))
            if not cands:
                return None
            with open(max(cands, key=os.path.getmtime)) as f:
                return json.load(f)
        except Exception:
            return None

    def _read(self):
        def fnum(key, default=0.0):
            try:
                return float(str(self.vars[key].get()).strip())
            except Exception:
                return default
        def inum(key, default=1):
            # signed integer: |n| = number of points, sign = grid direction
            try:
                v = int(float(str(self.vars[key].get()).strip()))
                return v if v != 0 else 1
            except Exception:
                return default
        try:
            self._sync_object_size(fnum)
            obj = np.array([fnum("obj_x"), fnum("obj_y"), fnum("obj_z")])
            pad = obj + np.array([0.0, fnum("pad_dy"), fnum("pad_dz")])
            return {
                "obj": obj,
                "tilt_deg": fnum("obj_tilt_deg"),
                "tilt_axis": self.vars["obj_tilt_axis"].get(),
                "pad": pad,
                "pad_dy": fnum("pad_dy"),
                "pad_dz": fnum("pad_dz"),
                "pad_rot": fnum("pad_rot"),
                "nx": inum("grid_nx"),
                "ny": inum("grid_ny"),
                "step": fnum("grid_step", 1.0),
                "centered": bool(self.vars["grid_centered"].get()),
                "pad_frame": bool(self.vars["grid_pad_frame"].get()),
            }
        except Exception:
            return None

    def refresh(self):
        cfg = self._read()
        if cfg is None:
            self.info.config(text="check numeric inputs", foreground="red"); return

        obj, pad = cfg["obj"], cfg["pad"]
        offs = grid_2d(cfg["nx"], cfg["ny"], cfg["step"],
                       centered=cfg["centered"])            # (dx,dy) mm
        _rolled = abs(cfg["pad_rot"]) > 1e-6
        _pad_frame = bool(cfg["pad_frame"])
        if _pad_frame:
            offs = rotate_offsets(offs, cfg["pad_rot"])
        if hasattr(self, "grid_count"):
            _j = [float(np.hypot(offs[i+1][0]-offs[i][0],
                                 offs[i+1][1]-offs[i][1]))
                  for i in range(len(offs)-1)]
            _mx = max(_j) if _j else 0.0
            # The rolled-pad warning now fires only when the steps are NOT
            # following the pad, which is the case that actually shears the
            # swept region.
            _warn = _rolled and (not _pad_frame) and len(offs) > 1
            self.grid_count.config(
                text=f"{len(offs)} grasp points"
                     + ("  (centered: n = per side)" if cfg["centered"]
                        else "  (anchored at pad offset)")
                     + f"   ~{len(offs) * 2.5:.0f} min"
                     + f"   max jump {_mx:.1f} mm"
                     + ("   ⚠ pad rolled: grid still steps in world Y/Z"
                        if _warn else
                        (f"   steps follow the pad ({cfg['pad_rot']:+.0f}°)"
                         if (_rolled and _pad_frame and len(offs) > 1) else "")),
                foreground=("#b00" if (_mx > 15.0 or _warn) else "#555"))

        # ---------- TOP-DOWN (X-Y): two pads squeezing the cylinder along X ----------
        ax = self.ax_top; ax.clear()
        ax.set_title("TOP-DOWN (X-Y)\ntwo pads squeeze along X")
        ax.set_xlabel("world X (mm)"); ax.set_ylabel("world Y (mm)")
        # cylinder circle at object centre
        th = np.linspace(0, 2*np.pi, 80)
        ax.plot(obj[0] + (CYL_D/2)*np.cos(th), obj[1] + (CYL_D/2)*np.sin(th),
                color="steelblue", linewidth=2, label="cylinder")
        ax.scatter(obj[0], obj[1], color="steelblue", s=15)
        # two pads on -X and +X sides, tangent to the rim (+ small opening gap).
        # In top-down the pad's SHORT side (width, across Y) is visible as a line.
        rim = CYL_D/2 + GRIP_OPEN
        # A pad rolled by theta about its own face normal still lies in the same
        # plane, but its shadow along Y grows: W*|cos| + H*|sin|.
        _pr = np.radians(cfg["pad_rot"])
        half_y = (PAD_W*abs(np.cos(_pr)) + PAD_H*abs(np.sin(_pr))) / 2.0
        for gx, gy in offs:
            cy = pad[1] + gy    # the grid's Y offset shifts pads along Y
            # -X pad
            ax.plot([obj[0]-rim, obj[0]-rim], [cy-half_y, cy+half_y],
                    color="crimson", linewidth=3)
            # +X pad
            ax.plot([obj[0]+rim, obj[0]+rim], [cy-half_y, cy+half_y],
                    color="darkorange", linewidth=3)
        ax.plot([], [], color="crimson", linewidth=3, label="pad -X (s1)")
        ax.plot([], [], color="darkorange", linewidth=3, label="pad +X (s2)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

        # ---------- FRONT (Y-Z): pad face on the cylinder surface, 2D grid ----------
        ax = self.ax_front; ax.clear()
        ax.set_title("FRONT (Y-Z)\npad face on cylinder, grid")
        ax.set_xlabel("world Y (mm)"); ax.set_ylabel("world Z (mm)")
        # Cylinder drawn as a rectangle (length x diameter) tilted by tilt_deg.
        # In the FRONT (Y-Z) view we show tilt about X as an in-plane rotation
        # (standing=0 -> vertical bar; 90 -> horizontal). Tilt about Y/Z tips it
        # out of / within this plane; we approximate by rotating in-plane for X,
        # and note the axis in the title.
        tilt = np.radians(cfg["tilt_deg"])
        # rectangle corners centred at object (in Y-Z), long axis = Z when standing
        L, D = CYL_L, CYL_D
        corners = np.array([[-D/2, -L/2], [ D/2, -L/2], [ D/2, L/2], [-D/2, L/2], [-D/2, -L/2]])
        c, s = np.cos(tilt), np.sin(tilt)
        Rt = np.array([[c, -s], [s, c]])
        rot = corners @ Rt.T
        ax.plot(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue",
                linewidth=2, label=f"cylinder (tilt {cfg['tilt_deg']:.0f}° about {cfg['tilt_axis']})")
        ax.fill(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue", alpha=0.35)
        # grid of pad footprints (pad short side across Y, long side up Z)
        # colour by reachability if a report has been loaded:
        #   green = reachable, red = unreachable, crimson = not checked yet
        _rm = getattr(self, "reach_map", None) or {}
        _lbl_done = {"ok": False, "bad": False, "raw": False, "first": False}
        for i, (gx, gy) in enumerate(offs):
            py = pad[1] + gy - PAD_W/2
            pz = pad[2] + gx - PAD_H/2   # use X-grid as the up/down (Z) sweep on the face
            if i in _rm:
                _ok = bool(_rm[i])
                col = "#0a9d3a" if _ok else "#d81b1b"
                key = "ok" if _ok else "bad"
                lab = ("pad (reachable)" if _ok else "pad (UNREACHABLE)") \
                      if not _lbl_done[key] else None
                _lbl_done[key] = True
                ls = "-" if _ok else "--"
            else:
                col, ls = "crimson", "-"
                lab = "pad" if not _lbl_done["raw"] else None
                _lbl_done["raw"] = True
            # pt00 IS THE ANCHOR: it is the pose every other grasp is
            # extrapolated from, and the one the stitcher uses as the initial
            # frame, so it should not look like any other point in the grid.
            # Drawn purple and heavier. Reachability stays readable through
            # the LINESTYLE (dashed = unreachable), so no information is lost.
            _lw = 1.5
            if i == 0:
                col = "#7b2fbe"
                _lw = 2.8
                lab = ("pt00 (initial pad pose)" if not _lbl_done["first"]
                       else None)
                _lbl_done["first"] = True
            if abs(cfg["pad_rot"]) < 1e-6:
                ax.add_patch(mpatches.Rectangle((py, pz), PAD_W, PAD_H,
                                                fill=False, edgecolor=col,
                                                linewidth=_lw, linestyle=ls,
                                                label=lab))
            else:
                # Rotate the footprint about the PAD CENTRE, not a corner.
                # Drawn as a Polygon so this works on any matplotlib version
                # (Rectangle's rotation_point= needs >= 3.6).
                _cy, _cz = pad[1] + gy, pad[2] + gx
                _k = np.array([[-PAD_W/2, -PAD_H/2], [ PAD_W/2, -PAD_H/2],
                               [ PAD_W/2,  PAD_H/2], [-PAD_W/2,  PAD_H/2]])
                _c, _s = np.cos(_pr), np.sin(_pr)
                _kr = _k @ np.array([[_c, _s], [-_s, _c]])
                ax.add_patch(mpatches.Polygon(_kr + [_cy, _cz], closed=True,
                                              fill=False, edgecolor=col,
                                              linewidth=_lw, linestyle=ls,
                                              label=lab))
            ax.scatter(pad[1]+gy, pad[2]+gx, color=col, s=(22 if i == 0 else 10),
                       zorder=(6 if i == 0 else 3))
        # visit path: the exact order the collector executes (pt00 -> last)
        if len(offs) > 1:
            px = [pad[1] + gy for gx, gy in offs]
            pz = [pad[2] + gx for gx, gy in offs]
            ax.plot(px, pz, color="dimgray", linestyle="--", linewidth=1.2,
                    alpha=0.9, zorder=4, label="visit path")
            # start/last markers are HOLLOW rings so they never hide the
            # green/red reachability colour of the dot underneath them.
            ax.scatter(px[0], pz[0], facecolors="none", edgecolors="green",
                       s=90, linewidths=1.8, zorder=5, label="pt00 (start = base)")
            ax.scatter(px[-1], pz[-1], facecolors="none", edgecolors="red",
                       marker="s", s=90, linewidths=1.8, zorder=5, label="last pt")
            if len(offs) <= 24:
                for i, (x_, z_) in enumerate(zip(px, pz)):
                    ax.annotate(str(i), (x_, z_), textcoords="offset points",
                                xytext=(3, 3), fontsize=6, color="dimgray")
        ax.set_aspect("equal", adjustable="datalim")

        # ---------- flange / palm / pad-centre overlay (calibration made visible) --
        # Shows, at the anchor pad (pt00), where the EE flange and gripper palm sit
        # above the pad, using the CALIBRATED offset for this diameter. Makes the
        # ~157mm EE->pad and the palm-vs-rod clearance visible before you run.
        try:
            cal = self._cal_entry()
            if cal is not None:
                off_mm   = float(cal["TOOL_OFFSET_Z"]) * 1000.0      # pad centre -> EE
                palm_mm  = 10.79                                     # EE -> palm (measured)
                anchor_y = pad[1] + offs[0][1]
                anchor_z = pad[2] + offs[0][0]                       # pad-centre target
                ee_z     = anchor_z + off_mm

                PALM_HOUSING_DROP_MM = 75.9
                palm_z = ee_z - 10.79 - PALM_HOUSING_DROP_MM



                # palm_z   = ee_z - palm_mm
                rod_top  = obj[2] + CYL_L / 2
                ax.scatter([anchor_y], [ee_z], marker="v", s=70, color="#333",
                           zorder=6, label="EE (flange)")
                ax.scatter([anchor_y], [palm_z], marker="_", s=200, color="#8a6d3b",
                           zorder=6, label="palm")
                ax.plot([anchor_y, anchor_y], [palm_z, anchor_z], color="#999",
                        lw=1, ls=":", zorder=3)
                clr = palm_z - rod_top
                ax.annotate(f"EE↕pad {off_mm:.0f}mm\npalm↕rodtop {clr:+.0f}mm",
                            (anchor_y, ee_z), textcoords="offset points",
                            xytext=(6, -2), fontsize=6, color="#333")
        except Exception:
            pass

        ax.legend(fontsize=6, loc="upper right")
        # Fine ticks: the grid steps are ~5 mm, so 20/50 mm ticks were far too
        # coarse to read a pad position off the plot.
        ax.xaxis.set_major_locator(MultipleLocator(PREVIEW_TICK_MAJOR_MM))
        ax.yaxis.set_major_locator(MultipleLocator(PREVIEW_TICK_MAJOR_MM))
        ax.xaxis.set_minor_locator(MultipleLocator(PREVIEW_TICK_MINOR_MM))
        ax.yaxis.set_minor_locator(MultipleLocator(PREVIEW_TICK_MINOR_MM))
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="both", which="minor", length=2)
        ax.grid(which="major", alpha=0.35, linewidth=0.7)
        ax.grid(which="minor", alpha=0.15, linewidth=0.4)

        # ---------- 3D scene: cylinder + two pads + base + grid (real size) ----------
        ax = self.ax_3d; ax.clear()
        ax.set_title("3D scene (real size)")
        # cylinder as a 3D surface, tilted by tilt_deg about the chosen axis
        zc = np.linspace(-CYL_L/2, CYL_L/2, 20)
        th = np.linspace(0, 2*np.pi, 24)
        TH, ZC = np.meshgrid(th, zc)
        XC = (CYL_D/2)*np.cos(TH)
        YC = (CYL_D/2)*np.sin(TH)
        pts = np.stack([XC.ravel(), YC.ravel(), ZC.ravel()], axis=1)
        tilt = np.radians(cfg["tilt_deg"]); axis = cfg["tilt_axis"]
        c, s = np.cos(tilt), np.sin(tilt)
        if axis == "X":
            R3 = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == "Y":
            R3 = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            R3 = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        pts = pts @ R3.T + obj
        Xc = pts[:, 0].reshape(XC.shape); Yc = pts[:, 1].reshape(YC.shape); Zc = pts[:, 2].reshape(ZC.shape)
        ax.plot_surface(Xc, Yc, Zc, color="steelblue", alpha=0.4, linewidth=0)

        # two pads on -X and +X of the object, at each grid point (real rectangles)
        rim = CYL_D/2 + GRIP_OPEN
        def pad_rect(center, normal_x_sign):
            # pad face spans Y (width) and Z (height); positioned at +/-X from center
            cx = obj[0] + normal_x_sign*rim
            cy, cz = center[1], center[2]
            ys = np.array([cy-PAD_W/2, cy+PAD_W/2, cy+PAD_W/2, cy-PAD_W/2, cy-PAD_W/2])
            zs = np.array([cz-PAD_H/2, cz-PAD_H/2, cz+PAD_H/2, cz+PAD_H/2, cz-PAD_H/2])
            xs = np.full_like(ys, cx)
            return xs, ys, zs
        for gx, gy in offs:
            cen = np.array([obj[0], pad[1]+gy, pad[2]+gx])
            xs, ys, zs = pad_rect(cen, -1); ax.plot(xs, ys, zs, color="crimson", linewidth=1.5)
            xs, ys, zs = pad_rect(cen, +1); ax.plot(xs, ys, zs, color="darkorange", linewidth=1.5)

        # robot base marker
        ax.scatter(*ROBOT_BASE_MM, color="black", s=40, marker="s", label="robot base")
        ax.plot([], [], color="crimson", label="pad -X (s1)")
        ax.plot([], [], color="darkorange", label="pad +X (s2)")
        ax.plot([], [], color="steelblue", label="cylinder")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.legend(fontsize=6, loc="upper left")
        # equalize aspect roughly around the object
        rng = 120
        ax.set_xlim(obj[0]-rng, obj[0]+rng)
        ax.set_ylim(obj[1]-rng, obj[1]+rng)
        ax.set_zlim(obj[2]-rng, obj[2]+rng)

        n = abs(cfg["nx"] * cfg["ny"])
        self.info.config(
            text=f"pad offset from center: y={cfg['pad_dy']:.1f}, z={cfg['pad_dz']:.1f} mm\n"
                 f"grid: {cfg['nx']}x{cfg['ny']} = {n} grasp poses (anchored: pt00 = base,\n"
                 f"sign = direction), step {cfg['step']:.1f} mm "
                 f"[upd #{getattr(self, '_refresh_count', 0)+1}]",
            foreground="#0a6")

        # actually redraw the canvas (this was missing -> preview never updated)
        self.fig.tight_layout()
        self.canvas.draw()
        try:
            self.canvas.flush_events()
        except Exception:
            pass
        self._refresh_count = getattr(self, "_refresh_count", 0) + 1


    # ---------- Stage B: build config, save, launch Isaac ----------
    def build_config(self):
        """Assemble the full config dict: object pose+tilt, pad offset, and the
        list of grasp points (each = pad Y-Z offset from object center, mm).
        All poses in mm; the sim converts to metres."""
        cfg = self._read()
        if cfg is None:
            return None
        offs = grid_2d(cfg["nx"], cfg["ny"], cfg["step"],
                       centered=cfg["centered"])   # (dx,dy) mm on the face
        # Steps follow the PAD's own edges unless switched off. The rotation
        # happens HERE, once, so pad_offset_y/z_mm in the file are already the
        # world offsets every consumer needs — the collector reads them
        # straight (GRID_POINTS), heatmaps draws `planned` from them, and the
        # stitcher never sees them at all. Nothing downstream has to know.
        if cfg["pad_frame"]:
            offs = rotate_offsets(offs, cfg["pad_rot"])
        # each grasp point = pad offset from object centre (y,z), plus the base pad offset
        points = []
        for k, (gx, gy) in enumerate(offs):
            points.append({
                "index": k,
                "pad_offset_y_mm": cfg["pad_dy"] + gy,   # Y across the face
                "pad_offset_z_mm": cfg["pad_dz"] + gx,   # Z up/down the face
            })
        return {
            "object": {
                "center_world_mm": cfg["obj"].tolist(),
                "tilt_deg": cfg["tilt_deg"],
                "tilt_axis": cfg["tilt_axis"],
                "shape": "cylinder",
                "diameter_mm": CYL_D,
                "length_mm": CYL_L,
            },
            "pad": {
                "base_offset_y_mm": cfg["pad_dy"],
                "base_offset_z_mm": cfg["pad_dz"],
                "x_fixed_centered": True,
                # Documentation only — the collector reads the roll from
                # GRASP_ROT_DEG / GRASP_ROT_AXIS, not from this file.
                "rotation_deg": cfg["pad_rot"],
                "rotation_axis": "y",
            },
            "grid": {
                "nx": cfg["nx"], "ny": cfg["ny"], "step_mm": cfg["step"],
                "centered": cfg["centered"],
                # provenance: were pad_offset_y/z_mm stepped along world Y/Z,
                # or along the pad's own edges? Reading a run months later,
                # this is the only thing that says which.
                "step_frame": "pad" if cfg["pad_frame"] else "world",
                "step_roll_deg": (float(cfg["pad_rot"]) if cfg["pad_frame"]
                                  else 0.0),
                "n_points": len(points),
            },
            "points": points,
        }

    def save_config(self):
        cfg = self.build_config()
        if cfg is None:
            messagebox.showerror("Config", "Check numeric inputs."); return None
        os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
        with open(CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        # Snapshot the three preview plots next to the config. The collector
        # copies BOTH into the run folder, so no manual screenshots are needed.
        try:
            self.fig.savefig(PREVIEW_PNG, dpi=110, bbox_inches="tight")
            _png_note = "\npreview image saved"
        except Exception as e:
            _png_note = f"\n(preview image NOT saved: {e})"

        # ALSO drop both straight into the session folder, so pressing Save
        # Config is enough — you no longer have to run the grid before the
        # folder has a record of what you set up (fixed 2026-08-04).
        _sess_note = ""
        _sess = self._session_or_none()
        if _sess:
            try:
                os.makedirs(_sess, exist_ok=True)
                with open(os.path.join(_sess, "gui_config_used.json"), "w") as f:
                    json.dump(cfg, f, indent=2)
                try:
                    self.fig.savefig(os.path.join(_sess, "gui_preview.png"),
                                     dpi=110, bbox_inches="tight")
                except Exception:
                    pass
                _sess_note = "\nalso saved into session folder"
            except Exception as e:
                _sess_note = f"\n(session copy failed: {e})"

        self.status.config(
            text=f"saved config: {cfg['grid']['n_points']} points\n{CONFIG_JSON}"
                 + _png_note + _sess_note,
            foreground="#0a6")
        return cfg

    def save_experiment(self):
        """Save the full experiment recipe to a named JSON so it can be
        reloaded later and reproduced exactly. Stores both the GUI field
        values (for exact reload) and the built config (for reference)."""
        from tkinter import filedialog
        cfg = self.build_config()
        if cfg is None:
            messagebox.showerror("Experiment", "Check numeric inputs."); return
        fields = {k: self.vars[k].get() for k in self.vars
                  if k != "headless"}
        recipe = {"gui_fields": fields, "config": cfg,
                  "note": "Paper3 experiment recipe — reload in the cockpit"}
        exp_dir = os.path.join(PROJECT, "Data", "experiments")
        os.makedirs(exp_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=exp_dir, defaultextension=".json",
            filetypes=[("Experiment JSON", "*.json")],
            title="Save experiment recipe as")
        if not path:
            return
        with open(path, "w") as f:
            json.dump(recipe, f, indent=2)
        self.status.config(text="experiment saved:\n" + os.path.basename(path),
                           foreground="#0a6")

    def load_experiment(self):
        """Load a saved recipe OR a plain config back into the GUI fields.

        THREE file shapes are accepted, because all three describe a run:
          1. an experiment recipe from "Save Experiment As..."   (gui_fields)
          2. Data/gui_config.json                                (built config)
          3. a run folder's gui_config_used.json / pose_history.json
        Before 2026-08-04 only (1) worked: picking a config set NOTHING and
        still reported success in green, which is why the button looked broken.
        Now (2) and (3) are mapped back onto the fields, and a file that
        matches nothing says so instead of pretending.
        """
        from tkinter import filedialog
        exp_dir = os.path.join(PROJECT, "Data", "experiments")
        path = filedialog.askopenfilename(
            initialdir=(self._session_or_none() or
                        (exp_dir if os.path.isdir(exp_dir)
                         else os.path.join(PROJECT, "Data"))),
            filetypes=[("Experiment or config JSON", "*.json"), ("All", "*.*")],
            title="Load experiment recipe or config")
        if not path:
            return
        try:
            with open(path) as f:
                doc = json.load(f)

            # pose_history.json nests the config one level down
            if "gui_fields" not in doc and isinstance(doc.get("config"), dict):
                doc = doc["config"]

            fields, src = {}, ""
            if isinstance(doc.get("gui_fields"), dict):
                fields, src = dict(doc["gui_fields"]), "experiment recipe"
            elif "object" in doc and "grid" in doc:
                o = doc.get("object", {}) or {}
                p = doc.get("pad", {}) or {}
                g = doc.get("grid", {}) or {}
                c = list(o.get("center_world_mm", [None, None, None]))
                fields = {
                    "obj_x": c[0], "obj_y": c[1], "obj_z": c[2],
                    "obj_diam": o.get("diameter_mm"),
                    "obj_len": o.get("length_mm"),
                    "obj_tilt_deg": o.get("tilt_deg"),
                    "obj_tilt_axis": o.get("tilt_axis"),
                    "pad_dy": p.get("base_offset_y_mm"),
                    "pad_dz": p.get("base_offset_z_mm"),
                    "pad_rot": p.get("rotation_deg", 0.0),
                    "grid_nx": g.get("nx"), "grid_ny": g.get("ny"),
                    "grid_step": g.get("step_mm"),
                }
                fields = {k: v for k, v in fields.items() if v is not None}
                src = "config"

            n = 0
            for k, v in fields.items():
                if k in self.vars:
                    self.vars[k].set(f"{v:g}" if isinstance(v, float) else v)
                    n += 1
            if src == "config" and "grid_centered" in self.vars:
                self.vars["grid_centered"].set(
                    bool((doc.get("grid") or {}).get("centered", False)))
                n += 1
            if src == "config" and "grid_pad_frame" in self.vars:
                # Files written before 2026-08-09 have no step_frame key and
                # were, by definition, stepped in world Y/Z — so default to
                # "world" here rather than to the new behaviour, or reloading
                # an old config would silently redesign its grid.
                self.vars["grid_pad_frame"].set(
                    str((doc.get("grid") or {}).get("step_frame", "world"))
                    == "pad")
                n += 1

            if n == 0:
                messagebox.showwarning(
                    "Load",
                    "Nothing loaded — that file has no GUI fields and no "
                    "object/grid blocks I can read.\n\nPick an experiment recipe "
                    "from Data/experiments, Data/gui_config.json, or a run's "
                    "gui_config_used.json.")
                self.status.config(text="nothing loaded from that file",
                                   foreground="#b00")
                return

            self.refresh()
            self.status.config(
                text=f"{src} loaded ({n} fields):\n{os.path.basename(path)}",
                foreground="#0a6")
        except Exception as e:
            messagebox.showerror("Experiment", "Could not load:\n%s" % e)

    def save_and_show_cmd(self):
        cfg = self.save_config()
        if cfg is None:
            return
        # the exact, proven terminal command (this is what worked for you)
        headless = "1" if self.vars["headless"].get() else "0"
        _rot = cfg.get("pad", {}).get("rotation_deg", 0.0)
        _sess = self._session_or_none()
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            + (f'GRASP_RUN_DIR="{_sess}" \\\n' if _sess else
               f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n')
            + f'GRASP_BASENAME="gui" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            + (f'GRASP_ROT_DEG="{_rot:g}" \\\n'
               f'GRASP_ROT_AXIS="y" \\\n' if abs(_rot) > 1e-6 else "")
            + ("" if self.vars["reach_check"].get() else
               'GRASP_REACH_CHECK="0" \\\n')
            + ('GRASP_TOOL_COLLISION="1" \\\n'
               if self.vars["tool_collision"].get() else "")
            + self._cal_read_env()
            + ("" if self.vars["log_mesh"].get() else 'GRASP_LOG_MESH="0" \\\n')
            + f"{ISAAC_PY} {COLLECT_PY} \\\n"
            f"  --config {CONFIG_JSON}"
        )
        # pop a window with the command, selectable + a Copy button
        win = tk.Toplevel(self.root)
        win.title("Run command — copy into a terminal")
        tk.Label(win, text="Config saved. Copy this into a terminal and run it:\n"
                           "(watch the Isaac window; come back and press Show Heatmaps when done)",
                 justify="left").pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=80, height=7, wrap="none")
        txt.insert("1.0", cmd); txt.configure(state="normal")
        txt.pack(padx=10, pady=4)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
            self.status.config(text="command copied to clipboard.", foreground="#0a6")
        tk.Button(win, text="Copy to clipboard", command=_copy).pack(pady=(4, 10))
        self.status.config(text=f"config saved: {cfg['grid']['n_points']} points.\n"
                                f"copy the command to run.", foreground="#0a6")

    # ================= Reachability =================
    def save_and_show_reach_cmd(self):
        """Dry-run every grid point in Isaac (IK + Paper-2 limit gates). No motion."""
        cfg = self.save_config()
        if cfg is None:
            return
        headless = "1" if self.vars["headless"].get() else "0"
        _rot = cfg.get("pad", {}).get("rotation_deg", 0.0)
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            + (f'GRASP_RUN_DIR="{self._session_or_none()}" \\\n'
               if self._session_or_none() else
               f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/reach_check" \\\n')
            + f'GRASP_BASENAME="reach" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            f'GRASP_REACH_ONLY="1" \\\n'
            + ('GRASP_TOOL_COLLISION="1" \\\n'
               if self.vars["tool_collision"].get() else "")
            + self._cal_read_env()
            + ("" if self.vars["log_mesh"].get() else 'GRASP_LOG_MESH="0" \\\n')
            + (f'GRASP_ROT_DEG="{_rot:g}" \\\n'
               f'GRASP_ROT_AXIS="y" \\\n' if abs(_rot) > 1e-6 else "")
            + f"{ISAAC_PY} {COLLECT_PY} \\\n"
            f"  --config {CONFIG_JSON}"
        )
        win = tk.Toplevel(self.root)
        win.title("Reachability check — copy into a terminal")
        tk.Label(win, justify="left",
                 text=("Config saved. This checks EVERY grid point and writes\n"
                       "reachability_report.json. The robot does NOT move.\n"
                       "Then press 'Load reachability result' to colour the grid.")
                 ).pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=82, height=8, wrap="none")
        txt.insert("1.0", cmd); txt.pack(padx=10, pady=4)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
            self.status.config(text="reachability command copied.", foreground="#0a6")
        tk.Button(win, text="Copy to clipboard", command=_copy).pack(pady=(4, 10))
        self.status.config(text="config saved. copy the reachability command.",
                           foreground="#0a6")

    def _find_reach_report(self):
        """Newest reachability_report.json: next to the config, else newest run dir."""
        cands = []
        side = os.path.join(os.path.dirname(CONFIG_JSON), "reachability_report.json")
        if os.path.exists(side):
            cands.append(side)
        run_root = os.path.join(PROJECT, "Data", "gui_run")
        if os.path.isdir(run_root):
            for d in os.listdir(run_root):
                p = os.path.join(run_root, d, "reachability_report.json")
                if os.path.exists(p):
                    cands.append(p)
        if not cands:
            return None
        return max(cands, key=os.path.getmtime)

    def load_reachability(self):
        """Load the newest report and colour the grid green/red."""
        path = self._find_reach_report()
        if path is None:
            messagebox.showinfo("Reachability",
                                "No reachability_report.json found yet.\n\n"
                                "Press 'Check Reachability (no motion)' first and run "
                                "the command it shows.")
            return
        try:
            with open(path) as f:
                rep = json.load(f)
        except Exception as e:
            messagebox.showerror("Reachability", f"Could not read:\n{path}\n\n{e}")
            return
        self.reach_map = {int(p["index"]): bool(p.get("reachable", False))
                          for p in rep.get("points", [])}
        n_ok = sum(1 for v in self.reach_map.values() if v)
        n_all = len(self.reach_map)
        bad = [f"pt{i:02d}" for i, v in sorted(self.reach_map.items()) if not v]
        self.refresh()
        msg = f"{n_ok}/{n_all} points reachable."
        if bad:
            reasons = {p["index"]: p.get("reason", "") for p in rep.get("points", [])}
            detail = "\n".join(f"  pt{i:02d}: {reasons.get(i,'')}"
                               for i, v in sorted(self.reach_map.items()) if not v)
            msg += f"\n\nUNREACHABLE (will be skipped):\n{detail}"
        self.status.config(
            text=f"reachability: {n_ok}/{n_all} OK" + (f", {len(bad)} skipped" if bad else ""),
            foreground="#0a6" if not bad else "#c60")
        messagebox.showinfo("Reachability", f"{msg}\n\nreport:\n{path}")

    # ================= Calibrate tab =================
    def _sync_object_size(self, fnum):
        """Copy the GUI's diameter/length into the module-level CYL_D/CYL_L.

        Everything downstream (preview, palm-clearance line, 3D scene,
        reachability, the written config) already reads those two names, so
        this one call makes the whole GUI follow the entry fields. Also warns
        when the chosen diameter has no calibration yet -- the collector will
        refuse to run in that case, and it is better to see it here."""
        global CYL_D, CYL_L
        d = fnum("obj_diam", 26.0)
        L = fnum("obj_len", 140.0)
        if d > 0:
            CYL_D = d
        if L > 0:
            CYL_L = L
        if not hasattr(self, "obj_cal_lbl"):
            return
        # READ THE FILE THE DROPDOWN SELECTED, and name it (2026-08-25).
        # This used to read the main file on a hardcoded path and print a bare
        # "NOT calibrated", which could contradict what the run actually did:
        # a batch reading pad_offset_calibration_contact.json collected D26 and
        # D32 perfectly while this label called both uncalibrated. A status
        # line that can disagree with the run is worse than none, so it now
        # follows the same _cal_path() everything else uses and says which file
        # it looked in and what it found there.
        _cp = self._cal_path()
        _err = None
        try:
            with open(_cp) as f:
                cal = json.load(f)
        except FileNotFoundError:
            cal, _err = {}, "file not found"
        except Exception as e:
            cal, _err = {}, f"unreadable ({type(e).__name__})"
        ent = cal.get(f"{CYL_D:.1f}")
        _fn = os.path.basename(_cp)
        if ent:
            self.obj_cal_lbl.config(
                text=f"\u00d8{CYL_D:.1f} calibrated  "
                     f"(TOOL_OFFSET_Z {ent.get('TOOL_OFFSET_Z')}, "
                     f"close_rad {ent.get('close_rad')})   [{_fn}]",
                foreground="#070")
        else:
            est = max(0.05, (85.0 - CYL_D) / 106.0)
            why = (_err if _err else
                   (f"{_fn} has " +
                    (", ".join(sorted(cal, key=float)) if cal else "nothing")))
            self.obj_cal_lbl.config(
                text=f"\u00d8{CYL_D:.1f} NOT in {_fn} - collection will "
                     f"refuse. ({why}). Calibrate tab first; try "
                     f"GRASP_CLOSE_RAD={est:.3f}",
                foreground="#b00")

    def _build_calib_tab(self):
        """Calibrate the pad Z-offset for the CURRENT object diameter. Closes the
        gripper on the object (centered) once, measures TOOL_OFFSET_Z, stores it.
        Same copy-run-command + headless-toggle UX as the Collection tab."""
        # SCROLLABLE (2026-08-25), same treatment the Stitch tab got: the
        # closing-mode block pushed the buttons and the result box below the
        # window edge with no way to reach them. Everything below still grids
        # into `frm`; `frm` simply lives inside a Canvas now.
        _outer = ttk.Frame(self.tab_calib)
        _outer.grid(row=0, column=0, sticky="nsew")
        self.tab_calib.columnconfigure(1, weight=1)
        self.tab_calib.rowconfigure(0, weight=1)
        _outer.rowconfigure(0, weight=1)

        # Width is set from the CONTENT once it is built (see the end of this
        # method): a fixed width clipped the buttons and the wrapped help text
        # off the right-hand edge, with no horizontal scrollbar to reach them.
        _cv = tk.Canvas(_outer, highlightthickness=0)
        _cv.grid(row=0, column=0, sticky="nsew")
        _sb = ttk.Scrollbar(_outer, orient="vertical", command=_cv.yview)
        _sb.grid(row=0, column=1, sticky="ns")
        _cv.configure(yscrollcommand=_sb.set)

        frm = ttk.Frame(_cv, padding=10)
        _win = _cv.create_window((0, 0), window=frm, anchor="nw")
        frm.bind("<Configure>",
                 lambda e: _cv.configure(scrollregion=_cv.bbox("all")))
        _cv.bind("<Configure>",
                 lambda e: _cv.itemconfigure(_win, width=e.width))

        def _cwheel(ev):
            if getattr(ev, "num", None) == 4:
                _cv.yview_scroll(-1, "units")
            elif getattr(ev, "num", None) == 5:
                _cv.yview_scroll(1, "units")
            else:
                _cv.yview_scroll(int(-1 * (ev.delta / 120)), "units")
        for _w in (_cv, frm):
            _w.bind("<Enter>", lambda e: (_cv.bind_all("<MouseWheel>", _cwheel),
                                          _cv.bind_all("<Button-4>", _cwheel),
                                          _cv.bind_all("<Button-5>", _cwheel)))
            _w.bind("<Leave>", lambda e: (_cv.unbind_all("<MouseWheel>"),
                                          _cv.unbind_all("<Button-4>"),
                                          _cv.unbind_all("<Button-5>")))

        r = 0
        ttk.Label(frm, text="CALIBRATE pad Z-offset",
                  font=("", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, wraplength=250, foreground="#444", justify="left",
                  text=("Closes the gripper on the CURRENT object (centered) and "
                        "measures the exact TOOL_OFFSET_Z for its diameter, then "
                        "stores it. Collection refuses an object until it is "
                        "calibrated.")).grid(row=r, column=0, columnspan=2, sticky="w",
                                             pady=(2, 8)); r += 1

        self.calib_obj_lbl = ttk.Label(frm, text="", foreground="#06a", wraplength=250)
        self.calib_obj_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="PAD placement (mm)",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, text="Z offset (up/down)").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["calib_dz"], width=10)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda _e: self.refresh_calib()); r += 1

        ttk.Label(frm, text="close_rad").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["calib_close_rad"], width=10)
        e.grid(row=r, column=1, sticky="w")
        e.bind("<Return>", lambda _e: self.refresh_calib()); r += 1
        self.calib_rad_lbl = ttk.Label(frm, text="", foreground="#555",
                                       wraplength=250)
        self.calib_rad_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Use estimate for this diameter",
                   command=self._fill_close_rad_estimate).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(0, 6)); r += 1

        ttk.Label(frm, text="Y locked to center (needed for a valid diameter grip)",
                  foreground="#888", wraplength=250).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        # ---- CONTACT-AWARE CLOSING ------------------------------------
        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="CLOSING MODE",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2,
                                             sticky="w"); r += 1
        ttk.Label(frm, text="mode").grid(row=r, column=0, sticky="e")
        _md = ttk.Combobox(frm, textvariable=self.vars["calib_mode"],
                           values=["fixed angle", "contact-aware",
                                   "both (compare)"],
                           state="readonly", width=18)
        _md.grid(row=r, column=1, sticky="w")
        _md.bind("<<ComboboxSelected>>", lambda _e: self._sync_calib_mode()); r += 1
        ttk.Label(frm, text="signal").grid(row=r, column=0, sticky="e")
        _sig = ttk.Combobox(frm, textvariable=self.vars["contact_signal"],
                            values=["deformation (mm)", "tactile (counts)"],
                            state="readonly", width=18)
        _sig.grid(row=r, column=1, sticky="w")
        _sig.bind("<<ComboboxSelected>>",
                  lambda _e: self._update_contact_hint()); r += 1
        self.contact_tgt_lbl = ttk.Label(frm, text="target (mm)")
        self.contact_tgt_lbl.grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["contact_target"], width=10)
        e.grid(row=r, column=1, sticky="w")
        e.bind("<Return>", lambda _e: self._update_contact_hint()); r += 1
        self.contact_lbl = ttk.Label(frm, text="", foreground="#555",
                                     wraplength=250, justify="left")
        self.contact_lbl.grid(row=r, column=0, columnspan=2, sticky="w",
                              pady=(2, 0)); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Checkbutton(frm, text="Run headless (no Isaac window)",
                        variable=self.vars["calib_headless"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Update Preview", command=self.refresh_calib).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(6, 2)); r += 1
        ttk.Button(frm, text="Save + Show Calibrate Command",
                   command=self.save_and_show_calib_cmd).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
        self.calib_status = ttk.Label(frm, text="", foreground="#06a", wraplength=250)
        self.calib_status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0)); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="AFTER the calibrate run:",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Load calibration result",
                   command=self.show_calibration).grid(
            row=r, column=0, sticky="ew", pady=2)
        ttk.Button(frm, text="Reset view",
                   command=self.reset_calibration_view).grid(
            row=r, column=1, sticky="ew", pady=2); r += 1
        self.calib_result = tk.Text(frm, width=34, height=10, wrap="word")
        self.calib_result.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0)); r += 1

        # preview (FRONT Y-Z only) — object + centered pad
        self.fig_calib = Figure(figsize=(4.6, 5.2), dpi=100)
        self.ax_calib = self.fig_calib.add_subplot(1, 1, 1)
        self.canvas_calib = FigureCanvasTkAgg(self.fig_calib, master=self.tab_calib)
        self.canvas_calib.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self._update_contact_hint()
        # size the scroll panel to whatever the widgets actually need
        frm.update_idletasks()
        _cv.configure(width=frm.winfo_reqwidth() + 6)
        self.refresh_calib()

    def refresh_calib(self):
        cfg = self._read()
        self._update_close_rad_hint()
        ax = self.ax_calib; ax.clear()
        if cfg is None:
            ax.set_title("check numeric inputs"); self.canvas_calib.draw(); return
        obj = cfg["obj"]
        try:
            cdz = float(str(self.vars["calib_dz"].get()).strip())
        except Exception:
            cdz = 0.0
        pad_z = obj[2] + cdz
        # is the pad still on the cylinder body (within +/- length/2)?
        on_body = abs(cdz) + PAD_H/2 <= CYL_L/2 + 1e-6
        flag = "" if on_body else "   [WARNING: pad off the object end]"
        self.calib_obj_lbl.config(
            text=(f"object center (mm): {obj.round(1).tolist()}\n"
                  f"diameter: {CYL_D} mm    (Y centered)\n"
                  f"pad Z: {pad_z:.1f} mm  (offset {cdz:+.1f}){flag}"))
        ax.set_title("CALIBRATE — pad on object (FRONT Y-Z)")
        ax.set_xlabel("world Y (mm)"); ax.set_ylabel("world Z (mm)")
        tilt = np.radians(cfg["tilt_deg"])
        L, D = CYL_L, CYL_D
        corners = np.array([[-D/2, -L/2], [D/2, -L/2], [D/2, L/2], [-D/2, L/2], [-D/2, -L/2]])
        c, s = np.cos(tilt), np.sin(tilt); Rt = np.array([[c, -s], [s, c]]); rot = corners @ Rt.T
        ax.plot(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue",
                linewidth=2, label="cylinder")
        ax.fill(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue", alpha=0.35)
        pad_color = "crimson" if on_body else "red"
        ax.add_patch(mpatches.Rectangle((obj[1]-PAD_W/2, pad_z-PAD_H/2), PAD_W, PAD_H,
                     fill=False, edgecolor=pad_color, linewidth=1.8,
                     linestyle="-" if on_body else "--", label="pad (Y centered)"))
        ax.scatter(obj[1], pad_z, color=pad_color, s=12)

        # ---- overlay the MEASURED pad centre from the last real grasp (#2) ----
        # Only shown AFTER 'Load calibration result' is pressed (self._show_measured),
        # so opening the GUI shows a clean target-only plot, not a stale run.
        try:
            if getattr(self, "_show_measured", False):
                pr = self._newest_probe()
                cal = self._cal_entry()
                if pr and cal and ("TSF_right_CASE_closed_world_mm" in pr):
                    r = np.array(pr["TSF_right_CASE_closed_world_mm"])
                    l = np.array(pr["TSF_left_CASE_closed_world_mm"])
                    case_mid = 0.5 * (r + l)                 # Case origin (mm, world)
                    shift = float(cal.get("pad_center_above_case_m", 0.0)) * 1000.0
                    meas_y = case_mid[1]
                    meas_z = case_mid[2] - shift             # Case origin -> pad centre
                    ax.add_patch(mpatches.Rectangle((meas_y-PAD_W/2, meas_z-PAD_H/2),
                                 PAD_W, PAD_H, fill=False, edgecolor="#0a9d3a",
                                 linewidth=1.6, linestyle="--", label="pad measured"))
                    ax.scatter(meas_y, meas_z, color="#0a9d3a", s=12)
                    ax.annotate(f"Δz {meas_z-pad_z:+.1f} mm", (meas_y, meas_z),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=7, color="#0a7d2a")
        except Exception:
            pass

        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        self.canvas_calib.draw()

    def build_calib_config(self):
        """Config for a calibrate grasp: ONE point, Y centered, Z = calib_dz.
        Y is intentionally locked to center so the pads meet the full diameter."""
        cfg = self._read()
        if cfg is None:
            return None
        try:
            cdz = float(str(self.vars["calib_dz"].get()).strip())
        except Exception:
            cdz = 0.0
        return {
            "object": {
                "center_world_mm": cfg["obj"].tolist(),
                "tilt_deg": cfg["tilt_deg"],
                "tilt_axis": cfg["tilt_axis"],
                "shape": "cylinder",
                "diameter_mm": CYL_D,
                "length_mm": CYL_L,
            },
            "pad": {"base_offset_y_mm": 0.0, "base_offset_z_mm": cdz,
                    "x_fixed_centered": True},
            "grid": {"nx": 1, "ny": 1, "step_mm": 8.0, "n_points": 1},
            "points": [{"index": 0, "pad_offset_y_mm": 0.0, "pad_offset_z_mm": cdz}],
            "calibrate": True,
        }

    def close_rad_estimate(self, d_mm=None):
        """Finger-joint angle that closes the 2F-85 to `d_mm`.

        The jaw span is close to linear in the joint angle: fully open is
        ~85 mm at ~0 rad, and the VERIFIED 26 mm grasp sits at 0.55 rad, so
        span ~= 85 - 106*rad. This is a STARTING POINT, not a calibration —
        the calibrate run checks the tactile peak against the 26 mm
        reference and refuses to store anything outside a sane band."""
        d = CYL_D if d_mm is None else float(d_mm)
        return max(0.05, (85.0 - d) / 106.0)

    def _fill_close_rad_estimate(self):
        self.vars["calib_close_rad"].set(f"{self.close_rad_estimate():.3f}")
        self.refresh_calib()

    def _update_close_rad_hint(self):
        """Show where close_rad will come from, and what it means in mm."""
        if not hasattr(self, "calib_rad_lbl"):
            return
        txt = self.vars["calib_close_rad"].get().strip()
        ent = self._cal_entry()
        est = self.close_rad_estimate()
        if txt:
            try:
                rad = float(txt)
            except ValueError:
                self.calib_rad_lbl.config(
                    text="close_rad must be a number (or blank)",
                    foreground="#b00")
                return
            span = 85.0 - 106.0 * rad
            self.calib_rad_lbl.config(
                text=f"squeezes to ~{span:.1f} mm span vs \u00d8{CYL_D:.1f} "
                     f"object  ({CYL_D - span:+.1f} mm of squeeze). "
                     f"Estimate for this diameter: {est:.3f}",
                foreground=("#070" if 0.0 < (CYL_D - span) < 8.0 else "#b00"))
        elif ent and ent.get("close_rad") is not None:
            self.calib_rad_lbl.config(
                text=f"blank -> collector uses the stored "
                     f"{float(ent['close_rad']):.3f} rad for this diameter",
                foreground="#555")
        else:
            self.calib_rad_lbl.config(
                text=f"blank -> collector falls back to the 26 mm value "
                     f"0.550 rad, which will MISS a \u00d8{CYL_D:.1f} object. "
                     f"Use {est:.3f}",
                foreground="#b00")

    def _sync_calib_mode(self):
        """Keep the legacy contact_close flag in step with the 3-way mode."""
        m = self.vars["calib_mode"].get()
        self.vars["contact_close"].set(m.startswith("contact"))
        self._update_contact_hint()

    def _update_contact_hint(self):
        """Say plainly what the chosen mode will do, and what it will not."""
        mode = self.vars["calib_mode"].get()
        sig = self.vars["contact_signal"].get()
        if hasattr(self, "contact_tgt_lbl"):
            self.contact_tgt_lbl.config(
                text="target (counts)" if sig.startswith("tactile")
                else "target (mm)")
        if mode.startswith("both"):
            self.contact_lbl.config(
                text=("Two grasps on this diameter, back to back: one at the "
                      "fixed close_rad above, one closing to the target. "
                      "Writes pad_offset_calibration_fixed.json and "
                      "_contact.json.\n\nThe existing hand-tuned file is not "
                      "touched — and it should not be the comparison anyway: "
                      "it was measured 3-20 August, so differences against it "
                      "mix 'the method changed' with 'three weeks passed'. "
                      "Two grasps taken today separate the two.\n\nThe "
                      "command box will show TWO commands; run them in order."),
                foreground="#06a")
            return
        on = mode.startswith("contact")
        if not on:
            self.contact_lbl.config(
                text=("Fixed angle: closes to close_rad above and stores the "
                      "offset it lands at. This is how all nine current "
                      "entries were made."),
                foreground="#555")
            return
        try:
            tgt = float(self.vars["contact_target"].get())
        except ValueError:
            self.contact_lbl.config(text="target must be a number.",
                                    foreground="#a00"); return
        if sig.startswith("tactile"):
            self.contact_lbl.config(
                text=(f"Closes until the tactile sum rises {tgt:.0f} counts "
                      f"above rest — the same quantity tactile_peak_sum in the "
                      f"calibration file is the maximum of, so compare it "
                      f"against the 8600-15400 those entries record. Baseline "
                      f"with the pads open is about 250.\n\nRead live from "
                      f"the extension in memory (it runs in the same process), "
                      f"not from the CSV, which only flushes every 50 frames "
                      f"against a 60-frame close.\n\nCAVEAT: inference is an "
                      f"async loop that drops queued frames, so the prediction "
                      f"can lag or go stale. The run refuses to trust a "
                      f"reading whose frame never advanced and says so in the "
                      f"log — check for TACTILE WENT STALE.\n\nclose_rad "
                      f"stays a hard ceiling."),
                foreground="#06a")
            return
        self.contact_lbl.config(
            text=(f"Closes until the pad is squashed by {tgt:.2f} mm, then "
                  f"stores THAT angle and offset. Measured after removing the "
                  f"pad's rigid motion, so it reads 0 at rest and only rises "
                  f"on real contact.\n\nMEASURED on \u00d826 at close_rad "
                  f"0.557 (tactile peak 15367, firm): 0.000 mm at 0.46 rad, "
                  f"0.3 mm at 0.51, 1.4 mm at full close. So 1.0 mm is a firm "
                  f"grasp that stops slightly early — which is the point, "
                  f"since the same indentation on every diameter is what makes "
                  f"the squeeze consistent.\n\nclose_rad above stays a hard "
                  f"ceiling: it can only stop earlier, never squeeze harder. "
                  f"Writes pad_offset_calibration_contact.json; the hand-tuned "
                  f"file is untouched."),
            foreground="#06a")

    def _cal_read_env(self):
        """Tell the collector to read the SAME calibration file the grid was
        designed with. Silence here would mean the designer used one offset
        and the run used another."""
        sfx = {"fixed": "_fixed", "contact": "_contact"}.get(
            self.vars["calib_source"].get(), "")
        return f'GRASP_CAL_READ="{sfx}" \\\n' if sfx else ""

    def _contact_signal_env(self):
        """The signal + target lines alone, with no CONTACT_CLOSE switch.

        CALIB_BOTH already implies contact closing for its second grasp, so the
        switch is not wanted there — but the signal and target very much are."""
        sig = self.vars["contact_signal"].get()
        try:
            tgt = float(self.vars["contact_target"].get())
        except ValueError:
            tgt = 6000.0 if sig.startswith("tactile") else 1.0
        if sig.startswith("tactile"):
            return (f'GRASP_CONTACT_SIGNAL="tactile" \\\n'
                    f'GRASP_CONTACT_TARGET="{tgt:g}" \\\n')
        return (f'GRASP_CONTACT_SIGNAL="deformation" \\\n'
                f'GRASP_CONTACT_TARGET="{tgt/1000.0:g}" \\\n')

    def _contact_env(self):
        """The env lines for contact-aware closing, or '' when it is off."""
        if not self.vars["contact_close"].get():
            return ""
        sig = self.vars["contact_signal"].get()
        try:
            tgt = float(self.vars["contact_target"].get())
        except ValueError:
            return ""
        if sig.startswith("tactile"):
            # counts are counts; no unit conversion
            return (f'GRASP_CONTACT_CLOSE="1" \\\n'
                    f'GRASP_CONTACT_SIGNAL="tactile" \\\n'
                    f'GRASP_CONTACT_TARGET="{tgt:g}" \\\n')
        # the GUI field is in mm; the collector works in stage units (m)
        return (f'GRASP_CONTACT_CLOSE="1" \\\n'
                f'GRASP_CONTACT_SIGNAL="deformation" \\\n'
                f'GRASP_CONTACT_TARGET="{tgt/1000.0:g}" \\\n')

    def save_and_show_calib_cmd(self):
        cfg = self.build_calib_config()
        if cfg is None:
            messagebox.showerror("Calibrate", "Check numeric inputs."); return
        os.makedirs(os.path.dirname(CALIB_CONFIG_JSON), exist_ok=True)
        with open(CALIB_CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        headless = "1" if self.vars["calib_headless"].get() else "0"
        _crad = self.vars["calib_close_rad"].get().strip()
        if _crad:
            try:
                float(_crad)
            except ValueError:
                messagebox.showerror("Calibrate",
                                     "close_rad must be a number, or blank.")
                return
        def _one(contact, suffix):
            """One calibrate invocation. contact=None keeps the mode's own
            setting; suffix='' writes the main file."""
            # ONE place decides the signal and its units (2026-08-25). This
            # used to hardcode deformation, so picking tactile in the dropdown
            # produced a deformation command with the tactile target divided by
            # 1000 — a nonsense number that would still have run.
            env = ('GRASP_CONTACT_CLOSE="1" \\\n' + self._contact_signal_env()
                   if contact else "")
            return (
                f"cd {EXAMPLES_DIR} && \\\n"
                f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n'
                f'GRASP_BASENAME="calib" \\\n'
                f'GRASP_HEADLESS="{headless}" \\\n'
                f'GRASP_CALIBRATE="1" \\\n'
                + env
                + (f'GRASP_CAL_SUFFIX="{suffix}" \\\n' if suffix else "")
                + (f'GRASP_CLOSE_RAD="{_crad}" \\\n' if _crad else "")
                + f"{ISAAC_PY} {COLLECT_PY} \\\n"
                f"  --config {CALIB_CONFIG_JSON}")

        _mode = self.vars["calib_mode"].get()
        if _mode.startswith("both"):
            # ONE launch, ONE descent, TWO closes (2026-08-25). The second
            # point is a pad-to-pad move of zero distance, so the arm never
            # lifts between them: pt00 closes to the fixed angle, pt01 closes
            # to the contact target, at the same pose in the same session.
            # Everything except the closing is held fixed, which is the whole
            # point of comparing them.
            cmd = (
                f"cd {EXAMPLES_DIR} && \\\n"
                f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n'
                f'GRASP_BASENAME="calib" \\\n'
                f'GRASP_HEADLESS="{headless}" \\\n'
                f'GRASP_CALIBRATE="1" \\\n'
                f'GRASP_CALIB_BOTH="1" \\\n'
                # Built directly, NOT via _contact_env(): that returns nothing
                # unless contact_close is ticked, and "both" deliberately
                # leaves it unticked. Calling it here silently dropped the
                # signal and target, so a run asked to use TACTILE fell back to
                # the collector's deformation defaults and recorded
                # signal=deformation, target=0.001 (2026-08-25).
                + self._contact_signal_env()
                + (f'GRASP_CLOSE_RAD="{_crad}" \\\n' if _crad else "")
                + f"{ISAAC_PY} {COLLECT_PY} \\\n"
                f"  --config {CALIB_CONFIG_JSON}")
        else:
            cmd = _one(_mode.startswith("contact"), "")
        win = tk.Toplevel(self.root)
        win.title("Calibrate command — copy into a terminal")
        tk.Label(win, justify="left",
                 text=("Config saved. Copy this into a terminal and run it.\n"
                       "It closes on the object ONCE (Y centered, at your Z) and stores "
                       "the offset for this diameter.\nThen press 'Load calibration result'."
                       )).pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=82, height=8, wrap="none")
        txt.insert("1.0", cmd); txt.pack(padx=10, pady=4)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
            self.calib_status.config(text="calibrate command copied.", foreground="#0a6")
        tk.Button(win, text="Copy to clipboard", command=_copy).pack(pady=(4, 10))
        self.calib_status.config(text="calib config saved. copy the command to run.",
                                 foreground="#0a6")

    def show_calibration(self):
        cal_path = self._cal_path()   # follow the dropdown, not a fixed name
        self.calib_result.delete("1.0", "end")
        if not os.path.exists(cal_path):
            self.calib_result.insert("end", f"No calibration file yet:\n{cal_path}\n\n"
                                            f"Run the calibrate command first.")
            return
        try:
            with open(cal_path) as f:
                cal = json.load(f)
        except Exception as e:
            self.calib_result.insert("end", f"Could not read:\n{e}"); return
        if not cal:
            self.calib_result.insert("end", "Calibration file is empty."); return
        lines = ["Calibrated offsets by diameter:", ""]
        for k, v in sorted(cal.items(), key=lambda kv: float(kv[0])):
            lines.append(f"  \u00d8{v.get('diameter_mm', k)} mm  ->  "
                         f"TOOL_OFFSET_Z = {v.get('TOOL_OFFSET_Z')}")
        key = f"{CYL_D:.1f}"
        if key in cal:
            v = cal[key]
            lines += ["", f"Current object (\u00d8{CYL_D}):  [calibrated]",
                      f"  TOOL_OFFSET_Z   = {v.get('TOOL_OFFSET_Z')} m",
                      f"  method          = {v.get('method','?')}"]
            if "TOOL_OFFSET_Z_case_origin" in v:
                lines += [f"  case-origin      = {v['TOOL_OFFSET_Z_case_origin']}",
                          f"  + pad-centre     = {v.get('pad_center_above_case_m')}"]
            # left/right pad symmetry (from the calibration grasp)
            try:
                pr = np.array(v["pad_right_closed_world_m"]) * 1000
                pl = np.array(v["pad_left_closed_world_m"]) * 1000
                lines += ["", "Pad symmetry (L vs R):",
                          f"  Z apart : {abs(pr[2]-pl[2]):.2f} mm  (should be ~0)",
                          f"  X gap   : {abs(pr[0]-pl[0]):.2f} mm  (pad-origin to origin;"
                          f" not the contact gap)"]
            except Exception:
                pass
            # landing error from the newest real grasp
            try:
                probe = self._newest_probe()
                if probe and "TSF_right_CASE_closed_world_mm" in probe:
                    r = np.array(probe["TSF_right_CASE_closed_world_mm"])
                    l = np.array(probe["TSF_left_CASE_closed_world_mm"])
                    shift = float(v.get("pad_center_above_case_m", 0.0)) * 1000
                    meas_z = 0.5*(r[2]+l[2]) - shift
                    tgt_z = probe.get("gui_target_m", [0,0,0])[2]*1000
                    lines += ["", "Last grasp landing:",
                              f"  pad centre Z : {meas_z:.1f} mm",
                              f"  target Z     : {tgt_z:.1f} mm",
                              f"  error        : {meas_z-tgt_z:+.1f} mm"]
            except Exception:
                pass
        else:
            lines.append(f"\nCurrent object (\u00d8{CYL_D}): NOT in "
                         f"{os.path.basename(cal_path)}")
        self.calib_result.insert("end", "\n".join(lines))
        self.calib_status.config(text="calibration loaded.", foreground="#0a6")
        self._show_measured = True    # now the green measured pad may draw
        self.refresh_calib()   # refresh the preview so the measured pad shows

    def reset_calibration_view(self):
        """Clear the measured overlay + text -> a fresh target-only plot."""
        self._show_measured = False
        try:
            self.calib_result.delete("1.0", "end")
        except Exception:
            pass
        self.calib_status.config(text="view reset - target only.", foreground="#06a")
        self.refresh_calib()

    # ---------- Stage C: read-back heatmaps + pose history ----------
    def _run_dir(self):
        # Priority: an explicit "Plot from folder..." override, then the active
        # SESSION folder, then the newest run on disk.
        forced = getattr(self, "_forced_run_dir", None)
        if forced:
            return forced
        sess = getattr(self, "_session_dir", None)
        if sess and os.path.isdir(sess):
            return sess
        import glob
        base = os.path.join(PROJECT, "Data", "gui_run")
        runs = sorted(glob.glob(os.path.join(base, "run_*")))
        if runs:
            return runs[-1]          # newest run
        return base                   # fallback (old flat layout)

    def save_plot_scale(self):
        """Write Data/plot_scale.json — read by stitching / heatmaps /
        temporal_snapshots so the GUI and standalone runs agree."""
        raw = self.vars["scale_fixed"].get().strip()
        fixed = None
        if raw:
            try:
                fixed = float(raw)
                if fixed <= 0:
                    fixed = None
            except ValueError:
                messagebox.showerror("Colour scale",
                                     "Fixed max must be a number (or blank).")
                return
        doc = {"shared": bool(self.vars["scale_shared"].get()),
               "fixed_vmax": fixed}
        try:
            os.makedirs(os.path.dirname(PLOT_SCALE_JSON), exist_ok=True)
            with open(PLOT_SCALE_JSON, "w") as f:
                json.dump(doc, f, indent=2)
        except Exception as e:
            messagebox.showerror("Colour scale", "Could not save:\n%s" % e)
            return
        mode = (f"fixed {fixed:g}" if fixed is not None
                else ("shared across run" if doc["shared"] else "auto per figure"))
        self.status.config(text="colour scale: " + mode + "\n(replot to apply)",
                           foreground="#0a6")

    # ================= Session folder =================
    @staticmethod
    def _ang_tag(v):
        """Angle -> filename-safe token. -10 -> 'm10', 35.5 -> '35p5'."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        s = f"{abs(v):g}".replace(".", "p")
        return ("m" + s) if v < 0 else s

    def _session_name(self):
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        obj = self._ang_tag(self.vars["obj_tilt_deg"].get())
        pad = self._ang_tag(self.vars["pad_rot"].get())
        return f"run_{stamp}_obj{obj}_pad{pad}"

    # ---- where new sessions are created --------------------------------
    # "Use existing..." points the session at ONE run folder, for re-plotting
    # or re-stitching. It is not a parent directory, so it could never send
    # NEW runs somewhere else — new_session hardcoded Data/gui_run and
    # silently ignored whatever had been picked. This is the missing piece:
    # a root that new_session actually uses, so a batch can be written
    # straight to an external drive.
    SETTINGS_PATH = os.path.join(PROJECT, "Data", "gui_settings.json")

    def _default_run_root(self):
        return os.path.join(PROJECT, "Data", "gui_run", "SIM")

    def _load_run_root(self):
        """Remembered across restarts: an overnight batch should not depend
        on someone re-picking the drive each morning."""
        try:
            with open(self.SETTINGS_PATH) as f:
                d = json.load(f)
            p = d.get("run_root")
            if p and os.path.isdir(p):
                return p
            if p:
                print(f"[gui] saved data folder is not present: {p}")
        except Exception:
            pass
        return self._default_run_root()

    def _save_run_root(self, path):
        try:
            d = {}
            if os.path.exists(self.SETTINGS_PATH):
                with open(self.SETTINGS_PATH) as f:
                    d = json.load(f)
            d["run_root"] = path
            os.makedirs(os.path.dirname(self.SETTINGS_PATH), exist_ok=True)
            with open(self.SETTINGS_PATH, "w") as f:
                json.dump(d, f, indent=2)
        except Exception as e:
            print(f"[gui] could not save the data folder ({e})")

    def _run_root_free_gb(self, path):
        try:
            st = os.statvfs(path)
            return st.f_bavail * st.f_frsize / 1e9
        except Exception:
            return None

    def _refresh_root_label(self):
        p = getattr(self, "_run_root", None) or self._default_run_root()
        free = self._run_root_free_gb(p)
        ext = not p.startswith(os.path.expanduser("~"))
        txt = f"data folder: {p}"
        if free is not None:
            txt += f"   ({free:.0f} GB free)"
        if hasattr(self, "run_root_lbl"):
            self.run_root_lbl.config(
                text=txt,
                foreground=("#b00" if (free is not None and free < 5)
                            else ("#7b2fbe" if ext else "#555")))

    def set_run_root(self):
        """Choose the PARENT folder that new sessions are created in."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=(getattr(self, "_run_root", None)
                        or self._default_run_root()),
            title="Pick the folder that NEW run sessions will be created in")
        if not d:
            return
        # Writability is checked here, not at run time: discovering a
        # read-only or unmounted drive after a 45-minute grasp has already
        # started is the expensive way to find out.
        try:
            t = os.path.join(d, ".gui_write_test")
            with open(t, "w") as f:
                f.write("x")
            os.remove(t)
        except Exception as e:
            messagebox.showerror("Data folder",
                f"Cannot write to:\n{d}\n\n{e}\n\n"
                "Nothing was changed.")
            return
        self._run_root = d
        self._save_run_root(d)
        self._refresh_root_label()
        free = self._run_root_free_gb(d)
        messagebox.showinfo("Data folder",
            f"New sessions will be created in:\n{d}\n\n"
            + (f"{free:.0f} GB free.\n\n" if free is not None else "")
            + "Runs already open are unaffected. This is remembered between "
              "restarts.")

    def new_session(self):
        """Mint a new session folder from the clock and the current angles.
        Everything after this — reachability, the run, heatmaps, stitching —
        lands in this one folder."""
        root = getattr(self, "_run_root", None) or self._default_run_root()
        if not os.path.isdir(root):
            try:
                os.makedirs(root, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Session",
                    f"The data folder is not available:\n{root}\n\n{e}\n\n"
                    "If it is an external drive, is it still plugged in?")
                return
        path = os.path.join(root, self._session_name())
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Session", "Could not create folder:\n%s" % e)
            return
        self._session_dir = path
        self._forced_run_dir = None      # session takes over as the plot source
        self._stitch_dir = None
        self.session_lbl.config(text=os.path.basename(path), foreground="#06a")
        if hasattr(self, "plot_src_lbl"):
            self.plot_src_lbl.config(text="plot source: session folder",
                                     foreground="#06a")
        if hasattr(self, "stitch_run_lbl"):
            self.stitch_run_lbl.config(text=os.path.basename(path))
        self.status.config(text="new session:\n" + path, foreground="#0a6")

    def pick_session(self):
        """Point the session at an existing run folder (to re-plot or re-stitch)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=os.path.join(PROJECT, "Data", "gui_run"),
            title="Pick an existing run folder to use as the session")
        if not d:
            return
        self._session_dir = d
        self._forced_run_dir = None
        self._stitch_dir = None
        self.session_lbl.config(text=os.path.basename(d), foreground="#06a")
        if hasattr(self, "plot_src_lbl"):
            self.plot_src_lbl.config(text="plot source: session folder",
                                     foreground="#06a")
        if hasattr(self, "stitch_run_lbl"):
            self.stitch_run_lbl.config(text=os.path.basename(d))
        self.status.config(text="session set to:\n" + d, foreground="#0a6")

    def open_session_folder(self):
        """Open the active plot/session folder in the desktop file manager.
        Falls back to the folder the plot buttons are pointing at, so this
        works even before a session has been minted."""
        import subprocess, sys
        path = self._session_or_none() or self._run_dir()
        if not path or not os.path.isdir(path):
            messagebox.showinfo(
                "Open folder",
                "No folder to open yet.\n\nPress 'New session' first, or run a "
                "grid so there is something on disk.")
            return
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)                        # noqa: S606
            else:
                subprocess.Popen(["xdg-open", path])
            self.status.config(text="opened:\n" + path, foreground="#0a6")
        except Exception as e:
            messagebox.showerror("Open folder",
                                 f"Could not open:\n{path}\n\n{e}")

    def _session_or_none(self):
        return getattr(self, "_session_dir", None)

    def choose_plot_folder(self):
        """Pick a saved run folder to (re)generate plots from. All four plot
        buttons then read from — and save back into — this folder."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=(self._session_or_none()
                        or os.path.join(PROJECT, "Data", "gui_run")),
            title="Pick a run folder to plot from")
        if d:
            self._forced_run_dir = d
            self.plot_src_lbl.config(text="plot source: " + os.path.basename(d),
                                     foreground="#06a")

    def use_newest_run(self):
        """Clear the folder override — plot from the newest run again."""
        self._forced_run_dir = None
        self.plot_src_lbl.config(text="plot source: newest run (auto)",
                                 foreground="#666")

    def show_heatmaps(self):
        """Hold-average heatmap per grasp, s1 | s2 side by side, one window
        per grid point. PNGs are saved to <run>/Heatmaps/ by viz/heatmaps.py."""
        import glob, traceback, importlib.util
        run = self._run_dir()
        if not glob.glob(os.path.join(run, "*_s1_tactile_maps.csv")):
            messagebox.showinfo("Heatmaps",
                f"No tactile files found in:\n{run}\n\nRun the simulation first.")
            return
        try:
            hpath = None
            for cand in (os.path.join(PROJECT, "viz", "heatmaps.py"),
                         os.path.join(PROJECT, "heatmaps.py"),
                         os.path.join(PROJECT, "sim", "heatmaps.py")):
                if os.path.exists(cand):
                    hpath = cand; break
            if hpath is None:
                messagebox.showerror("Heatmaps",
                    "heatmaps.py not found (expected in viz/).")
                return
            spec = importlib.util.spec_from_file_location("heatmaps", hpath)
            hm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hm)

            made = hm.plot_run(run)          # saves Heatmaps/heatmap_<tag>.png
            if not made:
                messagebox.showinfo("Heatmaps", "No heatmaps produced.")
                return

            # SAVE ONLY (2026-08-22). This used to open one matplotlib
            # window per grid point, which on a 63-point sweep buries the
            # screen in 63 windows and makes the cockpit unusable until they
            # are all closed. The PNGs are already on disk in Heatmaps/ and a
            # file browser shows them side by side far better than Tk can.
            self.status.config(
                text=(f"heatmaps: {len(made)} grasps (s1|s2) saved to "
                      f"Heatmaps/ — not opened"),
                foreground="#0a6")
        except Exception:
            messagebox.showerror("Heatmaps",
                "Heatmaps failed:\n\n" + traceback.format_exc())

    def show_pose_history(self):
        import json as _json
        run = self._run_dir()
        ph = os.path.join(run, "pose_history.json")
        if not os.path.exists(ph):
            messagebox.showinfo("Pose History",
                f"No pose_history.json in:\n{run}\n\nRun the simulation first.")
            return
        with open(ph) as f:
            data = _json.load(f)
        lines = ["Real pad/EE pose reached at each grasp (world, m):\n"]
        for p in data.get("points", []):
            ee = p["ee_world_m"]
            lines.append(f"  {p['tag']}:  x={ee[0]:+.4f}  y={ee[1]:+.4f}  z={ee[2]:+.4f}")
        win = tk.Toplevel(self.root)
        win.title("Pose History")
        txt = tk.Text(win, width=60, height=max(6, len(lines)+2), wrap="none")
        txt.insert("1.0", "\n".join(lines))
        txt.pack(padx=10, pady=10)
        self.status.config(text=f"showing {len(data.get('points', []))} poses.",
                           foreground="#0a6")

    def make_verifications(self):
        """Generate one Paper-2-style desired-vs-actual plot per grasp into
        <run>/Individual_Verifications/, then open the folder's first image."""
        run = self._run_dir()
        ph = os.path.join(run, "pose_history.json")
        if not os.path.exists(ph):
            messagebox.showinfo("Verification",
                f"No pose_history.json in:\n{run}\n\nRun the simulation first.")
            return
        try:
            import importlib.util
            vpath = os.path.join(PROJECT, "viz", "individual_verifications.py")
            if not os.path.exists(vpath):
                # fall back to sim/ or project root
                for alt in (os.path.join(PROJECT, "individual_verifications.py"),
                            os.path.join(PROJECT, "sim", "individual_verifications.py")):
                    if os.path.exists(alt):
                        vpath = alt; break
            spec = importlib.util.spec_from_file_location("indiv_verif", vpath)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            made = mod.plot_all(run)
        except Exception as e:
            messagebox.showerror("Verification", f"Error making plots:\n{e}")
            return
        if not made:
            messagebox.showinfo("Verification",
                "No plots made (pose history may lack pad poses — re-run the collector).")
            return
        # show each verification plot in its OWN separate window
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        for png in made:
            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mpimg.imread(png)); ax.axis("off")
            ax.set_title(os.path.basename(png))
            fig.tight_layout()
        plt.show()
        self.status.config(
            text=f"made {len(made)} verification plots (separate windows) in\n"
                 f"Individual_Verifications/",
            foreground="#0a6")


    def show_temporal(self):
        """Extract the paper's 4 temporal snapshots (5/50/95% + 3s) from this
        run's tactile CSVs and display them (rows=grasps, cols=squeeze stages)."""
        run = self._run_dir()
        import glob
        if not glob.glob(os.path.join(run, "*_s1_tactile_maps.csv")):
            messagebox.showinfo("Temporal Snapshots",
                f"No tactile files in:\n{run}\n\nRun the simulation first.")
            return
        try:
            import importlib.util
            tpath = os.path.join(PROJECT, "viz", "temporal_snapshots.py")
            for alt in (tpath, os.path.join(PROJECT, "temporal_snapshots.py"),
                        os.path.join(PROJECT, "sim", "temporal_snapshots.py")):
                if os.path.exists(alt):
                    tpath = alt; break
            spec = importlib.util.spec_from_file_location("temporal_snapshots", tpath)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            made = mod.plot_run(run)
        except Exception as e:
            messagebox.showerror("Temporal Snapshots", f"Error:\n{e}")
            return
        if not made:
            messagebox.showinfo("Temporal Snapshots", "No snapshots produced.")
            return
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        for png in made:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mpimg.imread(png)); ax.axis("off")
            ax.set_title(os.path.basename(png))
            fig.tight_layout()
        plt.show()
        self.status.config(
            text=f"temporal snapshots saved + shown ({len(made)} sensor plots).",
            foreground="#0a6")


    def show_pad_truth(self):
        """Safety report: GUI-designed pad pose vs pose reached in Isaac.
        Red banner if any grasp exceeds the threshold (once measured exists);
        amber 'FK-only' banner until the physics measurement is wired in."""
        run = self._run_dir()
        if not os.path.exists(os.path.join(run, "pose_history.json")):
            messagebox.showinfo("Pad Truth", f"No pose_history.json in:\n{run}"); return
        try:
            import importlib.util
            ppath = os.path.join(PROJECT, "viz", "pad_truth.py")
            spec = importlib.util.spec_from_file_location("pad_truth", ppath)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            png, any_bad, rows = mod.check_run(run)
        except Exception:
            import traceback; messagebox.showerror("Pad Truth", traceback.format_exc()); return
        if png is None:
            messagebox.showinfo("Pad Truth", "Nothing to check."); return
        import matplotlib.pyplot as plt, matplotlib.image as mpimg
        fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mpimg.imread(png)); ax.axis("off"); fig.tight_layout(); plt.show()
        self.status.config(
            text=("PAD MISMATCH — see report (red)" if any_bad else "pad truth checked"),
            foreground=("#c00" if any_bad else "#0a6"))


    # ---------- Batch tab (unattended queue) ----------
    def _build_batch_tab(self):
        """A queue of SESSION FOLDERS, run one after another, unattended.

        It queues folders the GUI already wrote rather than inventing a batch
        format, so every item has already passed design_grid's throat, palm,
        canvas, contact-band and per-point gripper checks before it can be in
        the list. Nothing is re-validated here and nothing new can be
        misconfigured here.

        It writes a queue file and a runner script; the run itself happens in a
        terminal, exactly like every other command this GUI produces. That also
        means closing the GUI does not kill a 20-hour batch."""
        outer = ttk.Frame(self.tab_batch, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.tab_batch.rowconfigure(0, weight=1)
        self.tab_batch.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        outer.columnconfigure(0, weight=1)
        r = 0

        ttk.Label(outer, text="BATCH — run saved configs unattended",
                  font=("", 11, "bold")).grid(row=r, column=0, columnspan=2,
                                              sticky="w"); r += 1
        ttk.Label(outer, wraplength=560, foreground="#555", justify="left",
                  text=("Set a test up on the Collection tab, press New session, "
                        "then Save Config — that writes gui_config_used.json into "
                        "the session folder. Add those folders here, in the order "
                        "you want them run.\n\nPut the runs you care about most "
                        "first: a queue that gets cut short then leaves you a "
                        "usable dataset instead of half of everything.")
                  ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 8))
        r += 1

        lf = ttk.Frame(outer)
        lf.grid(row=r, column=0, columnspan=2, sticky="nsew"); r += 1
        lf.rowconfigure(0, weight=1); lf.columnconfigure(0, weight=1)
        self.batch_list = tk.Listbox(lf, height=12, selectmode="extended",
                                     activestyle="none")
        self.batch_list.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, orient="vertical",
                           command=self.batch_list.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.batch_list.configure(yscrollcommand=sb.set)
        self._batch_items = []

        bf = ttk.Frame(outer)
        bf.grid(row=r, column=0, columnspan=2, sticky="w", pady=6); r += 1
        for txt, cmd in (("Add folder(s)", self.batch_add),
                         ("Add ALL in a parent folder", self.batch_add_parent),
                         ("Remove", self.batch_remove),
                         ("Up", lambda: self.batch_move(-1)),
                         ("Down", lambda: self.batch_move(1)),
                         ("Clear", self.batch_clear)):
            ttk.Button(bf, text=txt, command=cmd).pack(side="left", padx=2)

        ttk.Separator(outer, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(outer, text="SETTINGS — applied to EVERY run in the queue",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2,
                                             sticky="w"); r += 1
        sf = ttk.Frame(outer)
        sf.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(sf, text="calibration file:").grid(row=0, column=0, sticky="e")
        ttk.Combobox(sf, textvariable=self.vars["calib_source"],
                     values=["main", "fixed", "contact"], state="readonly",
                     width=10).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(sf, foreground="#555", wraplength=380, justify="left",
                  text=("one method for the whole dataset — mixing them inside "
                        "one dataset is exactly what ruins a week")
                  ).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(outer, text="Model the gripper BODY in collision",
                        variable=self.vars["tool_collision"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Checkbutton(outer, text="Save per-frame mesh CSVs  (OFF saves "
                                    "~98 MB/grasp — 59 GB over 20 hours)",
                        variable=self.vars["batch_log_mesh"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(outer, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Button(outer, text="Write batch + show command",
                   command=self.batch_write).grid(row=r, column=0, sticky="w")
        r += 1
        self.batch_lbl = ttk.Label(outer, text="", foreground="#06a",
                                   wraplength=560, justify="left")
        self.batch_lbl.grid(row=r, column=0, columnspan=2, sticky="w",
                            pady=(6, 0))

    # ---- batch helpers ----
    def _batch_label(self, folder):
        """Read the config so the list says what the run IS, not where it lives."""
        try:
            with open(os.path.join(folder, "gui_config_used.json")) as f:
                c = json.load(f)
            o, p, g = c.get("object", {}), c.get("pad", {}), c.get("grid", {})
            return (f"\u00d8{o.get('diameter_mm', '?'):g}x{o.get('length_mm', '?'):g}"
                    f"  roll {p.get('rotation_deg', 0):g}\u00b0"
                    f"  tilt {o.get('tilt_deg', 0):g}\u00b0"
                    f"  {g.get('n_points', '?')} pts")
        except Exception:
            return "(no gui_config_used.json — will FAIL)"

    def _batch_refresh(self):
        self.batch_list.delete(0, "end")
        for i, f in enumerate(self._batch_items, 1):
            self.batch_list.insert("end",
                                   f"{i:3d}.  {self._batch_label(f)}    "
                                   f"[{os.path.basename(f)}]")
        self.batch_lbl.config(text=f"{len(self._batch_items)} run(s) queued.",
                              foreground="#06a")

    def batch_add(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            title="Pick a session folder (contains gui_config_used.json)",
            initialdir=(getattr(self, "_run_root", None)
                        or self._default_run_root()))
        if d and d not in self._batch_items:
            self._batch_items.append(d)
            self._batch_refresh()

    def batch_add_parent(self):
        """Add every immediate subfolder that has a config, sorted by name."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            title="Pick the PARENT folder holding several session folders",
            initialdir=(getattr(self, "_run_root", None)
                        or self._default_run_root()))
        if not d:
            return
        found = sorted(
            os.path.join(d, x) for x in os.listdir(d)
            if os.path.isfile(os.path.join(d, x, "gui_config_used.json")))
        new = [f for f in found if f not in self._batch_items]
        self._batch_items.extend(new)
        self._batch_refresh()
        self.batch_lbl.config(
            text=f"added {len(new)} folder(s); {len(self._batch_items)} queued.",
            foreground="#06a")

    def batch_remove(self):
        for i in sorted(self.batch_list.curselection(), reverse=True):
            del self._batch_items[i]
        self._batch_refresh()

    def batch_move(self, d):
        sel = list(self.batch_list.curselection())
        if not sel:
            return
        for i in (sel if d < 0 else reversed(sel)):
            j = i + d
            if 0 <= j < len(self._batch_items):
                self._batch_items[i], self._batch_items[j] = \
                    self._batch_items[j], self._batch_items[i]
        self._batch_refresh()
        self.batch_list.selection_clear(0, "end")
        for i in sel:
            j = i + d
            if 0 <= j < len(self._batch_items):
                self.batch_list.selection_set(j)

    def batch_clear(self):
        self._batch_items = []
        self._batch_refresh()

    def batch_write(self):
        """Write batch_queue.json next to run_batch.py and show the command."""
        if not self._batch_items:
            messagebox.showwarning("Batch", "Nothing queued.")
            return
        sfx = {"fixed": "_fixed", "contact": "_contact"}.get(
            self.vars["calib_source"].get(), "")
        queue = {
            "generated": time.strftime("%Y%m%d_%H%M%S"),
            "settings": {
                "isaac_py": ISAAC_PY, "collect_py": COLLECT_PY,
                "examples_dir": EXAMPLES_DIR,
                "cal_read": sfx,
                "tool_collision": bool(self.vars["tool_collision"].get()),
                "log_mesh": bool(self.vars["batch_log_mesh"].get())},
            "items": [{"config": f, "label": self._batch_label(f)}
                      for f in self._batch_items]}
        qpath = os.path.join(EXAMPLES_DIR, "batch_queue.json")
        try:
            with open(qpath, "w") as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            messagebox.showerror("Batch", f"Could not write the queue:\n{e}")
            return

        missing = [f for f in self._batch_items
                   if not os.path.isfile(os.path.join(f, "gui_config_used.json"))]
        warn = (f"\n\n{len(missing)} folder(s) have NO gui_config_used.json "
                f"and will fail — press Save Config in each first."
                if missing else "")
        state = os.path.join(EXAMPLES_DIR, "batch_state.json")
        old = ("\n\nA batch_state.json already exists: re-running RESUMES and "
               "skips what is already done. Delete it to start over."
               if os.path.isfile(state) else "")

        cmd = f"cd {EXAMPLES_DIR} && python3 run_batch.py"
        try:
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
        except Exception:
            pass
        self.batch_lbl.config(
            text=(f"{len(self._batch_items)} run(s) written to "
                  f"batch_queue.json.\ncommand copied:  {cmd}"
                  f"\n\nrun_batch.py must be in {EXAMPLES_DIR}."
                  f"{warn}{old}"),
            foreground="#a00" if missing else "#0a6")
        win = tk.Toplevel(self.root)
        win.title("Batch — copy into a terminal")
        tk.Label(win, justify="left",
                 text=("The queue is written. Copy this into a terminal:\n"
                       "  python3 run_batch.py                 run / resume\n"
                       "  python3 run_batch.py --status        see progress\n"
                       "  python3 run_batch.py --retry-failed  retry failures\n"
                       "\nIt survives closing this GUI. Ctrl-C is safe: the "
                       "state is on disk, so re-running resumes.")
                 ).pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=80, height=3, wrap="none")
        txt.insert("1.0", cmd)
        txt.pack(padx=10, pady=4)
        tk.Button(win, text="Close", command=win.destroy).pack(pady=(2, 10))

    # ---------- Stitching tab (Block 2) ----------
    def _build_stitch_tab(self):
        # SCROLLABLE (2026-08-22). The tab grew past the window: the blob-axis
        # block and the status line sat below the bottom edge with no way to
        # reach them. Everything below is unchanged and still grids into
        # `frm` — `frm` now simply lives inside a Canvas that can scroll.
        _outer = ttk.Frame(self.tab_stitch)
        _outer.grid(row=0, column=0, sticky="nsew")
        self.tab_stitch.rowconfigure(0, weight=1)
        self.tab_stitch.columnconfigure(0, weight=1)
        _outer.rowconfigure(0, weight=1)
        _outer.columnconfigure(0, weight=1)

        _cv = tk.Canvas(_outer, highlightthickness=0)
        _cv.grid(row=0, column=0, sticky="nsew")
        _sb = ttk.Scrollbar(_outer, orient="vertical", command=_cv.yview)
        _sb.grid(row=0, column=1, sticky="ns")
        _cv.configure(yscrollcommand=_sb.set)

        frm = ttk.Frame(_cv, padding=14)
        _win = _cv.create_window((0, 0), window=frm, anchor="nw")
        frm.bind("<Configure>",
                 lambda e: _cv.configure(scrollregion=_cv.bbox("all")))
        # keep the content the width of the viewport so wraplength labels
        # behave exactly as they did before the canvas existed
        _cv.bind("<Configure>",
                 lambda e: _cv.itemconfigure(_win, width=e.width))

        # Wheel scrolling, bound to the canvas rather than globally so the
        # other tabs and any entry field keep their own behaviour. Linux
        # sends Button-4/5, not MouseWheel.
        def _wheel(ev):
            if getattr(ev, "num", None) == 4:
                _cv.yview_scroll(-1, "units")
            elif getattr(ev, "num", None) == 5:
                _cv.yview_scroll(1, "units")
            else:
                _cv.yview_scroll(int(-1 * (ev.delta / 120)), "units")
        for _w in (_cv, frm):
            _w.bind("<Enter>", lambda e: (_cv.bind_all("<MouseWheel>", _wheel),
                                          _cv.bind_all("<Button-4>", _wheel),
                                          _cv.bind_all("<Button-5>", _wheel)))
            _w.bind("<Leave>", lambda e: (_cv.unbind_all("<MouseWheel>"),
                                          _cv.unbind_all("<Button-4>"),
                                          _cv.unbind_all("<Button-5>")))
        r = 0
        ttk.Label(frm, text="BLOCK 2 — stitch per-grasp maps into ONE extended contact map",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(frm, justify="left", foreground="#555", text=(
            "Every grasp's HOLD-AVERAGE map is projected at its recorded pad offset\n"
            "(pose_history.json) onto one mm canvas; overlapping cells are averaged.\n"
            "Outputs land in <run>/Stitched/:  stitched_s1/s2 .png + .npy (+ mask),\n"
            "and training_pair.npz = INPUT (center grasp) -> TARGET (extended map).")
                  ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 10)); r += 1

        ttk.Label(frm, text="run folder").grid(row=r, column=0, sticky="e")
        self.stitch_run_lbl = ttk.Label(frm, text="(newest run — auto)", foreground="#06a")
        self.stitch_run_lbl.grid(row=r, column=1, sticky="w", padx=4)
        ttk.Button(frm, text="Browse…", command=self._stitch_browse).grid(
            row=r, column=2, sticky="ew"); r += 1
        ttk.Button(frm, text="Use newest run", command=self._stitch_use_newest).grid(
            row=r, column=2, sticky="ew", pady=2); r += 1

        ttk.Label(frm, text="canvas resolution (mm/cell)").grid(row=r, column=0, sticky="e")
        self.vars["stitch_res"] = tk.StringVar(value="1.0")
        ttk.Entry(frm, textvariable=self.vars["stitch_res"], width=8).grid(
            row=r, column=1, sticky="w", padx=4); r += 1

        ttk.Button(frm, text="Build Stitched Maps (s1 + s2)",
                   command=self.do_stitch).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(10, 3)); r += 1
        ttk.Button(frm, text="Export Training Pair (center -> extended)",
                   command=self.do_export_pair).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1
        ttk.Checkbutton(
            frm,
            text="also export ANCHORED pairs (one per grasp, into "
                 "Stitched/pairs/)",
            variable=self.vars["export_anchors"]).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(frm, justify="left", foreground="#555", wraplength=430, text=(
            "Re-centres the canvas on each grasp instead of only pt00, so one "
            "sweep yields many pairs at no robot cost. training_pair.npz is "
            "untouched. Measured on 6 rolled-pad runs: tactile-centroid error "
            "fell 5.96 -> 3.58 mm (interior) -> 3.00 mm (with edge anchors), "
            "crossing Paper 1's 4.42 mm safe-zone threshold; L1 and SSIM were "
            "flat. They are AUGMENTATION, not new objects -- train.py decides "
            "which to use via --anchors, and always validates on pt00 only.")
        ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 4)); r += 1
        ttk.Button(frm, text="Grid Accuracy (designed vs actual movement)",
                   command=self.do_grid_accuracy).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1

        # ---- Block 2 validation: stitch round-trip fidelity (SSIM / TC / GSR) ----
        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(10, 6)); r += 1
        ttk.Label(frm, text="VALIDATE stitch fidelity (round-trip)",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(frm, justify="left", foreground="#555", wraplength=430, text=(
            "Re-samples the stitched canvas at each grasp's own taxel positions and\n"
            "compares recovered-vs-original with SSIM, Tactile-Centroid error, and\n"
            "GSR. This checks the STITCHER as a container (high SSIM / low TC is\n"
            "expected, esp. the center grasp) — it is NOT model completion.\n"
            "GSR needs TensorFlow; it prints 'disabled' and skips if unavailable.")
                  ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 4)); r += 1
        ttk.Checkbutton(frm, text="include GSR (needs TensorFlow + model)",
                        variable=self.vars["stitch_want_gsr"]).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Button(frm, text="Validate Stitch (SSIM / TC / GSR)",
                   command=self.do_validate).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(4, 3)); r += 1

        # ---- contact-blob orientation: measured axis vs the grid design ----
        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(10, 6)); r += 1
        ttk.Label(frm, text="BLOB AXIS — contact angle vs design",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=3,
                                             sticky="w"); r += 1
        ttk.Label(frm, justify="left", foreground="#555", wraplength=430, text=(
            "Per grasp, the weighted-PCA principal axis of the contact blob\n"
            "(Paper 2's own method), against the angle the GEOMETRY implies:\n"
            "the contact band clipped by the pad window, put through the same\n"
            "7x4 estimator. Offsets matter — a tilted band pushed off-centre\n"
            "is only partly visible, and a short piece of a tilted line reads\n"
            "much straighter than the line itself, so 'expected' is often far\n"
            "from the rod tilt. Everything comes from this run's own folder.\n"
            "Per grasp, not on the stitched map — on a 1xN sweep the stitched\n"
            "blob's shape is set by the stepping direction, not the contact.")
                  ).grid(row=r, column=0, columnspan=3, sticky="w",
                         pady=(0, 4)); r += 1
        ttk.Label(frm, text="contact band width (mm)").grid(row=r, column=0,
                                                            sticky="e")
        self.vars["blob_band_mm"] = tk.StringVar(value="8.0")
        ttk.Entry(frm, textvariable=self.vars["blob_band_mm"], width=8).grid(
            row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Checkbutton(frm, text="fit band width from this run's own maps",
                        variable=self.vars["blob_fit_width"]).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(frm, text="also show the metric self-test "
                                  "(ideal line contact -> what PCA reads)",
                        variable=self.vars["blob_selftest"]).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Button(frm, text="Blob Axis (measured vs expected)",
                   command=self.do_blob_axis).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(4, 3)); r += 1

        self.stitch_status = ttk.Label(frm, text="", foreground="#0a6",
                                       wraplength=430, justify="left")
        self.stitch_status.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1

    def _stitch_target_dir(self):
        return self._stitch_dir if getattr(self, "_stitch_dir", None) else self._run_dir()

    def _stitch_browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=(self._session_or_none()
                        or os.path.join(PROJECT, "Data", "gui_run")),
            title="Pick a run folder")
        if d:
            self._stitch_dir = d
            self.stitch_run_lbl.config(text=os.path.basename(d))

    def _stitch_use_newest(self):
        self._stitch_dir = None
        self.stitch_run_lbl.config(text="(newest run — auto)")

    def _load_stitching_module(self):
        import importlib.util
        for cand in (os.path.join(PROJECT, "viz", "stitching.py"),
                     os.path.join(PROJECT, "stitching.py"),
                     os.path.join(PROJECT, "sim", "stitching.py")):
            if os.path.exists(cand):
                spec = importlib.util.spec_from_file_location("stitching", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def do_design_grid(self):
        """Fill nx / ny / step / pad z from the object's geometry.

        Writes into the SAME variables you type into, then calls refresh(),
        so every other button behaves exactly as before and any number can
        still be overridden by hand afterwards."""
        import traceback
        try:
            # _read() refreshes CYL_D / CYL_L from the entry fields as a side
            # effect (via _sync_object_size), so the designer sees whatever
            # object is currently typed in rather than a stale global.
            cfg = self._read()
            if cfg is None:
                messagebox.showerror("Design grid",
                    "Could not read the object fields.")
                return
            cal = self._cal_entry()
            if cal is None:
                messagebox.showwarning("Design grid",
                    f"No calibration entry for \u00d8{CYL_D:.1f} mm.\n\n"
                    "The palm-clearance limit needs TOOL_OFFSET_Z for THIS "
                    "diameter, so calibrate this object first — guessing it "
                    "would put the palm into the top of the rod.")
                return
            roll = float(cfg.get("pad_rot", 0.0))
            step = float(cfg.get("step", 6.0))
            if step <= 0:
                step = 6.0

            ok, res, why = design_grid(
                CYL_D, CYL_L, float(cal["TOOL_OFFSET_Z"]) * 1000.0,
                step_mm=step, pad_roll_deg=roll,
                obj_tilt_deg=float(cfg.get("tilt_deg", 0.0)),
                obj_tilt_axis=str(cfg.get("tilt_axis", "X")))
            text = "\n".join(why)
            if not ok:
                self.design_lbl.config(text="design REFUSED — see dialog",
                                       foreground="#b00")
                messagebox.showerror("Design grid — refused",
                    text + "\n\nNothing was changed. Adjust the step, the "
                    "object, or the pad roll and try again.")
                return

            # THE TRANSPOSITION, FIXED HERE AND ONLY HERE (2026-08-22).
            # grid_2d's "n steps X" box steps along world Z — UP THE ROD —
            # and its "n steps Y" box steps across world Y. The designer
            # derives one count from the along window and one from the
            # across band. Until today they were handed over in the order
            # they were derived, which transposed every automatic grid; see
            # design_grid's docstring for the proof from run_20260821_165542.
            # grid_2d is deliberately NOT changed: its convention is what the
            # collector, the stitcher, the serpentine order and every run
            # already on disk speak.
            self.vars["grid_nx"].set(str(res["n_along"]))    # X box = up the rod
            self.vars["grid_ny"].set(str(res["n_across"]))   # Y box = across it
            self.vars["grid_step"].set(f"{res['step_mm']:.1f}")
            self.vars["pad_dz"].set(f"{res['pad_dz_mm']:.2f}")
            self.vars["pad_dy"].set("0.0")
            self.vars["grid_centered"].set(True)
            # The point check tested the grid AS THE PAD STEPS IT. Leaving
            # this off would run a different grid from the one verified.
            self.vars["grid_pad_frame"].set(True)
            self.refresh()

            self.design_lbl.config(
                text=(f"\u00d8{CYL_D:.0f}x{CYL_L:.0f}: "
                      f"{2*res['n_across']+1} across x "
                      f"{2*res['n_along']+1} along = {res['n_points']} pts "
                      f"at {res['step_mm']:.1f} mm, pad z "
                      f"{res['pad_dz_mm']:+.1f} mm\n"
                      f"bound by {res['bound_by']}; gripper gap "
                      f"{res['gap_mm']:+.1f} mm"),
                foreground="#0a6")
            messagebox.showinfo("Design grid", text)
        except Exception:
            messagebox.showerror("Design grid",
                "Failed:\n\n" + traceback.format_exc())

    def _load_grid_accuracy_module(self):
        """Load grid_accuracy.py. Like validation.py it imports stitching
        itself, so its directory goes on sys.path first."""
        import importlib.util, sys
        for cand in (os.path.join(PROJECT, "viz", "grid_accuracy.py"),
                     os.path.join(PROJECT, "grid_accuracy.py"),
                     os.path.join(PROJECT, "sim", "grid_accuracy.py")):
            if os.path.exists(cand):
                d = os.path.dirname(cand)
                if d not in sys.path:
                    sys.path.insert(0, d)
                spec = importlib.util.spec_from_file_location(
                    "grid_accuracy", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def do_grid_accuracy(self):
        """Commanded grid vs where the pad actually landed."""
        import traceback
        try:
            mod = self._load_grid_accuracy_module()
            if mod is None:
                messagebox.showerror("Grid accuracy",
                    "grid_accuracy.py not found (expected in viz/).")
                return
            if self._load_stitching_module() is None:
                messagebox.showerror("Grid accuracy",
                    "stitching.py not found (expected in viz/); "
                    "grid_accuracy needs it for the measured positions.")
                return
            run = self._stitch_target_dir()
            png, st = mod.plot_run(run)
            # bias and scatter are reported apart on purpose: a constant
            # offset shifts the whole stitch together and does not blur it,
            # while per-point scatter is what actually smears neighbouring
            # grasps into each other.
            msg = (f"compared {st['n_compared']} of {st['n_commanded']} "
                   f"points\n\n"
                   f"BIAS (systematic)\n"
                   f"   dY {st['bias_y_mm']:+.3f}   dZ {st['bias_z_mm']:+.3f} mm\n\n"
                   f"SCATTER (per-point)\n"
                   f"   dY {st['std_y_mm']:.3f}   dZ {st['std_z_mm']:.3f} mm\n\n"
                   f"miss: mean {st['mean_miss_mm']:.3f}, "
                   f"p95 {st['p95_miss_mm']:.3f}, "
                   f"max {st['max_miss_mm']:.3f} mm at {st['max_miss_at']}\n"
                   f"(one taxel is 5.5 x 5.29 mm)\n\n"
                   f"source: {st['offset_source']}")
            if st["missing"]:
                msg += (f"\n\nNOT REACHED ({len(st['missing'])}): "
                        + ", ".join(st["missing"]))
            self.stitch_status.config(
                text=f"grid accuracy: bias ({st['bias_y_mm']:+.2f}, "
                     f"{st['bias_z_mm']:+.2f}) mm, scatter "
                     f"({st['std_y_mm']:.2f}, {st['std_z_mm']:.2f}) mm\n{png}",
                foreground=("#b00" if st["missing"] else "#0a6"))
            messagebox.showinfo("Grid accuracy", msg)
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            fig = plt.figure(figsize=(13.0, 3.8))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mpimg.imread(png)); ax.axis("off")
            ax.set_title(os.path.basename(png))
            fig.tight_layout()
            plt.show()
        except Exception:
            messagebox.showerror("Grid accuracy",
                "Failed:\n\n" + traceback.format_exc())

    def _load_validation_module(self):
        """Load viz/validation.py. It imports stitching.py itself, so make sure
        the viz/ dir is on sys.path first."""
        import importlib.util, sys
        for cand in (os.path.join(PROJECT, "viz", "validation.py"),
                     os.path.join(PROJECT, "validation.py"),
                     os.path.join(PROJECT, "sim", "validation.py")):
            if os.path.exists(cand):
                d = os.path.dirname(cand)
                if d not in sys.path:
                    sys.path.insert(0, d)
                spec = importlib.util.spec_from_file_location("validation", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def _load_blob_module(self):
        """Load viz/blob_axis.py. Like validation.py it imports stitching.py
        itself, so the viz/ dir has to be on sys.path first."""
        import importlib.util, sys
        for cand in (os.path.join(PROJECT, "viz", "blob_axis.py"),
                     os.path.join(PROJECT, "blob_axis.py"),
                     os.path.join(PROJECT, "sim", "blob_axis.py")):
            if os.path.exists(cand):
                d = os.path.dirname(cand)
                if d not in sys.path:
                    sys.path.insert(0, d)
                spec = importlib.util.spec_from_file_location("blob_axis", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def do_stitch(self):
        import traceback
        try:
            mod = self._load_stitching_module()
            if mod is None:
                messagebox.showerror("Stitching",
                    "stitching.py not found (expected in viz/).")
                return
            run = self._stitch_target_dir()
            try:
                res = float(self.vars["stitch_res"].get())
            except Exception:
                res = 1.0
            made = mod.stitch_run(run, res)
            if not made:
                messagebox.showinfo("Stitching",
                    f"No stitched maps produced in:\n{run}\n\n"
                    "Needs tactile CSVs + pose_history.json (run the sim first).")
                return
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            for png in made:
                fig = plt.figure(figsize=(10.5, 4.8))
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(mpimg.imread(png)); ax.axis("off")
                ax.set_title(os.path.basename(png))
                fig.tight_layout()
            plt.show()
            self.stitch_status.config(
                text=f"stitched {len(made)} sensor map(s) -> "
                     f"{os.path.basename(run)}/Stitched/", foreground="#0a6")
        except Exception:
            messagebox.showerror("Stitching",
                "Stitching failed:\n\n" + traceback.format_exc())

    def do_export_pair(self):
        import traceback, glob
        try:
            mod = self._load_stitching_module()
            if mod is None:
                messagebox.showerror("Stitching",
                    "stitching.py not found (expected in viz/).")
                return
            run = self._stitch_target_dir()
            # The res box drives the FIGURES. The training pair has its own
            # pinned canvas (1.0 mm, 96 mm square, pad frame) so every run
            # exports one tensor shape — passing the box value here would
            # undo that, so it is deliberately not passed.
            npz = mod.export_pair(run)

            # export_pair returns None when it REFUSES — the run's designed
            # initial grasp was never collected, so the pair would be built on
            # a substitute. This used to report success in green with the word
            # "None" as the filename, which is the exact silent-substitution
            # failure the stitcher fix removed; do not reintroduce it here.
            if not npz:
                want = getattr(mod, "INITIAL_GRASP", "pt00")
                # use stitching's OWN key parser, so the point names quoted
                # here cannot drift from the ones it just complained about
                _key = getattr(mod, "_pt_key", None)
                have = sorted({k for k in
                               (_key(os.path.basename(p)) if _key else None
                                for p in glob.glob(os.path.join(
                                    run, "*_pt*_s1_tactile_maps.csv")))
                               if k})
                span = (f"{have[0]}..{have[-1]} ({len(have)} points)"
                        if have else "none found")
                self.stitch_status.config(
                    text=f"NO training pair written — designed initial grasp "
                         f"{want} is missing from this run",
                    foreground="#b00")
                messagebox.showwarning("Training pair REFUSED",
                    f"No training_pair.npz was written for:\n{run}\n\n"
                    f"The designed initial grasp {want} is not in this run.\n"
                    f"Collected: {span}\n\n"
                    "The stitched maps and figures are unaffected and still "
                    "valid — only the training pair is refused, because its "
                    "input frame would be a substitute grasp and nothing in "
                    "the file would say so.\n\n"
                    "Either re-collect the missing point (check "
                    "execution_ledger.json for why it failed), or set "
                    "STITCH_ALLOW_FALLBACK=1 to export with the substitution "
                    "recorded in the meta.")
                return

            msg = f"training pair exported:\n{npz}"

            # ANCHORED PAIRS, only after the pt00 pair succeeded. If pt00 was
            # refused the run is not fit to train on at all, so writing 40
            # anchored pairs from it would be manufacturing volume from data
            # the pipeline just rejected.
            if self.vars["export_anchors"].get():
                try:
                    made, skipped = mod.export_anchor_pairs(
                        run, include_edge=True, verbose=True)
                    n_i = sum(1 for _t, k in made if k == "interior")
                    n_e = len(made) - n_i
                    msg += (f"\n\nanchored pairs: {len(made)} written "
                            f"({n_i} interior, {n_e} edge)")
                    if skipped:
                        msg += f", {len(skipped)} skipped"
                    self.stitch_status.config(text=msg, foreground="#0a6")
                    messagebox.showinfo("Training pairs",
                        f"{os.path.basename(npz)} written, plus "
                        f"{len(made)} anchored pairs in Stitched/pairs/\n"
                        f"   {n_i} interior (measured target on all sides)\n"
                        f"   {n_e} edge (target lopsided — one side unswept)\n\n"
                        "These are extra VIEWS of this one sweep, not extra "
                        "objects. train.py chooses which to use with "
                        "--anchors none|interior|all, and validates on pt00 "
                        "pairs only so the settings stay comparable.")
                except Exception as e:
                    # The pt00 pair is already on disk and valid; the anchored
                    # ones are a bonus, so a failure here reports and stops
                    # rather than discarding what succeeded.
                    self.stitch_status.config(
                        text=msg + f"\n(anchored pairs failed: {e})",
                        foreground="#b00")
                    messagebox.showwarning("Anchored pairs",
                        f"training_pair.npz was written and is valid.\n\n"
                        f"The anchored pairs failed:\n{e}")
                    return

            self.stitch_status.config(text=msg, foreground="#0a6")
        except Exception:
            messagebox.showerror("Stitching",
                "Export failed:\n\n" + traceback.format_exc())

    def do_validate(self):
        """Round-trip validate the stitch (SSIM / TC / GSR) and show the report
        in a scrollable window. Runs in a thread because GSR/TensorFlow can take
        a few seconds to import the first time."""
        import traceback, threading
        run = self._stitch_target_dir()
        try:
            res = float(self.vars["stitch_res"].get())
        except Exception:
            res = 1.0
        want_gsr = bool(self.vars["stitch_want_gsr"].get())
        self.stitch_status.config(
            text="validating stitch (this can take a few seconds"
                 + (", loading TensorFlow for GSR…" if want_gsr else "") + ")",
            foreground="#06a")

        def _worker():
            try:
                mod = self._load_validation_module()
                if mod is None:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Validate", "validation.py not found (expected in viz/)."))
                    return
                results, report = mod.validate_and_save(run, res, want_gsr=want_gsr)
                self.root.after(0, lambda: self._show_validation_report(run, report))
            except Exception:
                tb = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror(
                    "Validate", "Validation failed:\n\n" + tb))

        threading.Thread(target=_worker, daemon=True).start()

    def do_blob_axis(self):
        """Per-grasp contact-blob orientation vs the angle the grid design
        implies. Threaded because it re-reads every grasp's CSV."""
        import traceback
        run = self._stitch_target_dir()
        selftest = bool(self.vars["blob_selftest"].get())
        fit_w = bool(self.vars["blob_fit_width"].get())
        try:
            band = float(self.vars["blob_band_mm"].get())
        except Exception:
            band = 8.0
        self.stitch_status.config(text="measuring blob axis…",
                                  foreground="#06a")

        def _worker():
            try:
                mod = self._load_blob_module()
                if mod is None:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Blob Axis",
                        "blob_axis.py not found (expected in viz/)."))
                    return
                report, pngs = mod.blob_and_save(
                    run, band_width_mm=band, fit_width=fit_w)
                if selftest:
                    report = report + "\n\n" + mod.metric_selftest()
                self.root.after(0, lambda: self._show_blob_report(
                    run, report, pngs))
            except Exception:
                tb = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror(
                    "Blob Axis", "Blob axis failed:\n\n" + tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_blob_report(self, run, report, pngs):
        self._show_report_window("Blob axis — " + os.path.basename(run),
                                 report)
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            for png in pngs:
                fig = plt.figure(figsize=(12.5, 4.4))
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(mpimg.imread(png)); ax.axis("off")
                ax.set_title(os.path.basename(png))
                fig.tight_layout()
            if pngs:
                plt.show()
        except Exception:
            pass
        self.stitch_status.config(
            text="blob axis done → " + os.path.basename(run)
                 + "/Stitched/blob_axis_report.txt", foreground="#0a6")

    def _show_report_window(self, title, report):
        """Scrollable, copyable text window. Shared by the blob-axis report
        and anything else that wants to show a plain-text result."""
        win = tk.Toplevel(self.root)
        win.title(title)
        txt = tk.Text(win, width=86, height=30, wrap="none",
                      font=("TkFixedFont", 10))
        txt.insert("1.0", report)
        txt.configure(state="normal")
        yscroll = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=yscroll.set)
        txt.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        yscroll.grid(row=0, column=1, sticky="ns", pady=10)
        win.columnconfigure(0, weight=1); win.rowconfigure(0, weight=1)

        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(report)
        ttk.Button(win, text="Copy report", command=_copy).grid(
            row=1, column=0, columnspan=2, pady=(0, 10))
        return win

    def _show_validation_report(self, run, report):
        win = tk.Toplevel(self.root)
        win.title("Stitch round-trip validation — " + os.path.basename(run))
        txt = tk.Text(win, width=78, height=28, wrap="none",
                      font=("TkFixedFont", 10))
        txt.insert("1.0", report)
        txt.configure(state="normal")
        yscroll = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=yscroll.set)
        txt.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        yscroll.grid(row=0, column=1, sticky="ns", pady=10)
        win.columnconfigure(0, weight=1); win.rowconfigure(0, weight=1)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(report)
        ttk.Button(win, text="Copy report", command=_copy).grid(
            row=1, column=0, columnspan=2, pady=(0, 10))
        self.stitch_status.config(
            text="validation done → saved to " + os.path.basename(run)
                 + "/Stitched/validation_report.txt", foreground="#0a6")

if __name__ == "__main__":
    root = tk.Tk()
    app = CockpitGUI(root)
    root.mainloop()
