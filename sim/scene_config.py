"""
scene_config.py — ONE place to set the cylinder pose and the tactile-pad poses,
                  and PREVIEW how the pad(s) meet the cylinder before you run
                  the heavy simulator.

RUNS IN PYCHARM (normal Python: numpy + matplotlib). No Isaac Sim here.

WHAT THIS DOES
--------------
1. Holds the CYLINDER pose (currently fixed in the scene — shown for reference).
2. Holds the PAD grid: where the tactile pad touches, stepping across the
   cylinder, with optional in-plane rotation.
3. Draws a TO-SCALE preview (front view + side view) so you can SEE the pad
   over the cylinder for every grid point — then decide if it's a good grid
   BEFORE collecting data.

WORKFLOW
--------
  Edit the CONFIG below  ->  run this file in PyCharm  ->  look at the preview
  ->  happy? then run the real collection (run_grid.sh) with the same numbers.

NOTE ON FRAMES
--------------
All positions are WORLD coordinates in METERS, measured from the scene origin.
The cylinder pose is read-only for now (baked in the scene USD). You control
the PAD grid freely.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import csv as _csv

# ============================================================
# CYLINDER  (currently FIXED in the scene — shown for reference)
# Read from the scene earlier: center, size, axis.
# ============================================================
CYL_CENTER = np.array([-0.26806, 0.199, 1.0522])  # world center of the cylinder (m)
CYL_DIAMETER = 0.026     # 26 mm
CYL_LENGTH   = 0.140     # 140 mm
CYL_AXIS     = "Z"       # long axis points up (world Z) = standing cylinder

# ============================================================
# OBJECT POSE — YOU EDIT THIS  (step 2)
# Where the object sits in the scene, and how it's oriented. The collector
# (grasp_one_grid_v2.py) bolts the object at this exact pose with a fixed
# joint, so it stays put AND the sensor still reads contact.
#
# OBJECT_POS = world position of the object CENTER [x, y, z] in meters.
# OBJECT_ORIENT = one of:
#   "standing"          long axis vertical (world Z) — a standing can
#   "horizontal_x"      lying down, long axis along world X
#   "horizontal_y"      lying down, long axis along world Y
#   "horizontal_y_tilt" lying along Y, then tilted by OBJECT_TILT_DEG about X
#   "tilt_x"            standing, tilted by OBJECT_TILT_DEG about X
# OBJECT_TILT_DEG = tilt angle for the *_tilt orientations.
# ============================================================
OBJECT_POS      = [-0.26806, 0.199, 1.0522]   # object center (world m)
OBJECT_ORIENT   = "standing"
# OBJECT_ORIENT   = "horizontal_y"
# OBJECT_ORIENT = "horizontal_y_tilt"

OBJECT_TILT_DEG = 20.0

# ============================================================
# SCENE REFERENCE POINTS (read from the scene earlier) — for the map view
# ============================================================
ORIGIN     = np.array([0.0, 0.0, 0.0])              # world origin (scene zero)
ROBOT_BASE = np.array([0.02093, -0.3375, 0.99275])  # robot base_link world pos

# ============================================================
# TACTILE PAD physical size (TSF-85, from Berith's paper: 22 x 37 mm)
# ============================================================
PAD_LONG  = 0.037   # 37 mm  -> the 7-taxel (long) direction
PAD_SHORT = 0.022   # 22 mm  -> the 4-taxel (short) direction

# ============================================================
# PAD GRID  — YOU EDIT THIS
# The grid is built around a CENTER contact point on the cylinder, then steps
# across the pad face. These are the SAME ideas as grid_config.py, but here we
# preview them visually first.
#
# CENTER of the grid = where the pad touches the cylinder, in WORLD meters.
# By default we put it at the cylinder's center height, on the +Y face
# (the side the gripper approaches).
# ============================================================
GRID_CENTER = np.array([-0.26806, 0.199, 1.0522])  # contact center on cylinder (m)

# Grid counts and step (meters). Sweep is in the PAD FACE plane:
#   "across" = horizontal, around the cylinder surface (world Y here)
#   "along"  = vertical, up/down the cylinder length    (world Z here)
N_ACROSS = 3        # points across (horizontal)
N_ALONG  = 3        # points along  (vertical, up/down the cylinder)
STEP     = 0.004    # 4 mm between grid points

# In-plane rotations to preview (degrees). [0] = no rotation.
ROTATIONS_DEG = [0]

# ============================================================
# Build the grid of pad poses (pure data)
# ============================================================
def build_pad_grid():
    pts = []
    acr = [(-(N_ACROSS-1)/2.0 + i) * STEP for i in range(N_ACROSS)]
    alo = [(-(N_ALONG -1)/2.0 + j) * STEP for j in range(N_ALONG)]
    for r, dz in enumerate(alo):          # row = vertical (along cylinder)
        for c, dy in enumerate(acr):      # col = horizontal (across)
            for rot in ROTATIONS_DEG:
                world = GRID_CENTER + np.array([0.0, dy, dz])
                pts.append({
                    "row": r, "col": c, "rot": rot,
                    "x": world[0], "y": world[1], "z": world[2],
                })
    return pts

# ============================================================
# GRASP / TOOL settings — needed so the saved grid drives the collector.
# The pad contact point on the cylinder maps to an END-EFFECTOR target that
# cuRobo plans to. The EE sits ABOVE the contact by the tool offset (the
# gripper+sensor length hanging below the wrist).
#
# From the proven working grasp:
#   contact center (cylinder)  Z = 1.0522
#   EE target (what we command) Z = 1.24244
#   tool offset = 1.24244 - 1.0522 = 0.19024 m  (EE is ~19 cm above contact)
# X and Y of the EE target match the contact's X/Y.
# ============================================================
TOOL_OFFSET_Z   = 0.19024   # EE sits this far above the contact point (m)
APPROACH_HEIGHT = 0.10      # how far above the EE target to start the descent
GRIPPER_CLOSE   = 0.55      # gripper close target (radians)
ROT_AXIS        = "z"       # in-plane rotation axis for grasp_one_grid_v2.py

# Where to save the grid + scene record
OUT_DIR        = "/home/kourosh/Paper3_Simulation/sim"
GRID_CSV_PATH  = os.path.join(OUT_DIR, "current_grid.csv")
SCENE_JSON_PATH = os.path.join(OUT_DIR, "current_scene.json")


def contact_to_ee(contact_xyz):
    """Map a contact point on the object to the END-EFFECTOR target the
    collector commands (add the tool offset in +Z)."""
    return np.array([contact_xyz[0], contact_xyz[1], contact_xyz[2] + TOOL_OFFSET_Z])


def save_grid_and_scene():
    """Write TWO files:
      1) current_grid.csv  — the grid the collector (run_grid.sh) reads.
         columns: label,X,Y,Z,approach,close,rot   (EE-target world coords)
      2) current_scene.json — the full scene record for rebuilding later
         (object pose, robot base, pad grid in CONTACT coords, settings).
    """
    pts = build_pad_grid()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1) collector CSV (EE-target coords) ----
    with open(GRID_CSV_PATH, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["label", "X", "Y", "Z", "approach", "close", "rot"])
        for p in pts:
            ee = contact_to_ee(np.array([p["x"], p["y"], p["z"]]))
            label = f"grid_r{p['row']:02d}_c{p['col']:02d}"
            if len(ROTATIONS_DEG) > 1:
                label += f"_rot{int(p['rot']):+03d}"
            w.writerow([label,
                        f"{ee[0]:.6f}", f"{ee[1]:.6f}", f"{ee[2]:.6f}",
                        f"{APPROACH_HEIGHT:.3f}", f"{GRIPPER_CLOSE:.3f}",
                        f"{p['rot']:.1f}"])

    # ---- 2) full scene record (for real-robot replay later) ----
    record = {
        "object": {
            "type": "cylinder",
            "center_world": CYL_CENTER.tolist(),
            "diameter_m": CYL_DIAMETER,
            "length_m": CYL_LENGTH,
            "long_axis": CYL_AXIS,
            # pose you set in this config (collector bolts the object here):
            "pose_world": list(OBJECT_POS),
            "orientation": OBJECT_ORIENT,
            "tilt_deg": OBJECT_TILT_DEG,
        },
        "robot_base_world": ROBOT_BASE.tolist(),
        "origin_world": ORIGIN.tolist(),
        "grasp_settings": {
            "tool_offset_z_m": TOOL_OFFSET_Z,
            "approach_height_m": APPROACH_HEIGHT,
            "gripper_close_rad": GRIPPER_CLOSE,
            "rot_axis": ROT_AXIS,
        },
        "grid": {
            "n_across": N_ACROSS, "n_along": N_ALONG, "step_m": STEP,
            "rotations_deg": ROTATIONS_DEG,
            "center_contact_world": GRID_CENTER.tolist(),
            "points_contact_world": [
                {"row": p["row"], "col": p["col"], "rot_deg": p["rot"],
                 "x": p["x"], "y": p["y"], "z": p["z"]} for p in pts
            ],
        },
    }
    with open(SCENE_JSON_PATH, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Saved grid  -> {GRID_CSV_PATH}  ({len(pts)} points)")
    print(f"Saved scene -> {SCENE_JSON_PATH}")
    return GRID_CSV_PATH, SCENE_JSON_PATH



def object_axis_endpoints():
    """Return the two endpoints (world xyz) of the cylinder's LONG axis and its
    radius, given OBJECT_POS / OBJECT_ORIENT / OBJECT_TILT_DEG. Used by the
    preview to draw the cylinder in its actual orientation."""
    c = np.array(OBJECT_POS, dtype=float)
    half = CYL_LENGTH / 2.0
    r = CYL_DIAMETER / 2.0
    t = np.deg2rad(OBJECT_TILT_DEG)

    # unit vector along the cylinder's long axis in WORLD, per orientation
    if OBJECT_ORIENT == "standing":
        axis = np.array([0.0, 0.0, 1.0])
    elif OBJECT_ORIENT == "horizontal_x":
        axis = np.array([1.0, 0.0, 0.0])
    elif OBJECT_ORIENT == "horizontal_y":
        axis = np.array([0.0, 1.0, 0.0])
    elif OBJECT_ORIENT == "horizontal_y_tilt":
        # lying along Y, tilted by t about X: axis rotates in the Y-Z plane
        axis = np.array([0.0, np.cos(t), np.sin(t)])
    elif OBJECT_ORIENT == "tilt_x":
        # standing, tilted by t about X: axis rotates from Z toward Y
        axis = np.array([0.0, np.sin(t), np.cos(t)])
    else:
        axis = np.array([0.0, 0.0, 1.0])

    p1 = c - half * axis
    p2 = c + half * axis
    return p1, p2, r, axis


def _draw_capsule(ax, a, b, r, i, j):
    """Draw a cylinder seen in a 2D projection as a thick line between its two
    end points a,b (using world component indices i,j for the two plot axes)."""
    ax.plot([a[i], b[i]], [a[j], b[j]],
            color="0.6", linewidth=max(2.0, r * 6000), solid_capstyle="round",
            zorder=1)
    # end caps as small circles
    for pt in (a, b):
        ax.add_patch(patches.Circle((pt[i], pt[j]), r,
                                    facecolor="0.8", edgecolor="0.4", zorder=2))


def preview():
    pts = build_pad_grid()

    fig = plt.figure(figsize=(15, 6))
    axM = fig.add_subplot(1, 3, 1)   # map view (top-down)
    axF = fig.add_subplot(1, 3, 2)   # front view (zoomed on contact)
    axS = fig.add_subplot(1, 3, 3)   # side view (zoomed on contact)

    # ---------- MAP VIEW (top-down, looking down Z): X vs Y ----------
    axM.set_title("Map view (top-down)\norigin, robot base, cylinder")
    # origin
    axM.plot(ORIGIN[0], ORIGIN[1], marker="+", color="black", markersize=14)
    axM.annotate("origin (0,0)", (ORIGIN[0], ORIGIN[1]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8)
    # robot base
    axM.plot(ROBOT_BASE[0], ROBOT_BASE[1], marker="s", color="tab:blue", markersize=10)
    axM.annotate("robot base", (ROBOT_BASE[0], ROBOT_BASE[1]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8, color="tab:blue")
    # cylinder drawn in its ACTUAL orientation (top-down: X vs Y, indices 0,1)
    p1, p2, r, axis = object_axis_endpoints()
    if OBJECT_ORIENT == "standing":
        # seen from above a standing cylinder is just a circle
        axM.add_patch(patches.Circle((OBJECT_POS[0], OBJECT_POS[1]), r,
                                     facecolor="0.8", edgecolor="0.4"))
    else:
        _draw_capsule(axM, p1, p2, r, 0, 1)
    axM.annotate("cylinder", (OBJECT_POS[0], OBJECT_POS[1]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8)
    axM.set_xlabel("world X (m)"); axM.set_ylabel("world Y (m)")
    axM.set_aspect("equal"); axM.grid(True, alpha=0.3); axM.autoscale_view()

    # ---------- FRONT VIEW (looking at the +Y face): Y across, Z up ----------
    axF.set_title("Front view (pad face)\nyellow = pad footprints")
    # cylinder in actual orientation, projected to (Y, Z) = indices 1,2
    if OBJECT_ORIENT == "standing":
        cyl_y0 = OBJECT_POS[1] - CYL_DIAMETER/2
        cyl_z0 = OBJECT_POS[2] - CYL_LENGTH/2
        axF.add_patch(patches.Rectangle((cyl_y0, cyl_z0), CYL_DIAMETER, CYL_LENGTH,
                                        facecolor="0.8", edgecolor="0.4", label="cylinder"))
    else:
        _draw_capsule(axF, p1, p2, r, 1, 2)
        axF.plot([], [], color="0.6", linewidth=4, label="cylinder")  # legend proxy
    for p in pts:
        _draw_pad(axF, p["y"], p["z"], PAD_SHORT, PAD_LONG, p["rot"])
    axF.set_xlabel("world Y (m) — across"); axF.set_ylabel("world Z (m) — up/down")
    axF.set_aspect("equal"); axF.autoscale_view()
    axF.legend(loc="upper right", fontsize=8)

    # ---------- SIDE VIEW (looking along Y): X depth, Z up ----------
    axS.set_title("Side view\n(pad approaches from +Y)")
    if OBJECT_ORIENT == "standing":
        axS.add_patch(patches.Rectangle((OBJECT_POS[0]-CYL_DIAMETER/2, cyl_z0),
                                        CYL_DIAMETER, CYL_LENGTH,
                                        facecolor="0.8", edgecolor="0.4"))
    else:
        _draw_capsule(axS, p1, p2, r, 0, 2)  # X depth, Z up = indices 0,2
    for p in pts:
        axS.plot([OBJECT_POS[0]-CYL_DIAMETER/2 - 0.002], [p["z"]],
                 marker="s", color="orange", markersize=6)
    axS.set_xlabel("world X (m) — depth"); axS.set_ylabel("world Z (m) — up/down")
    axS.set_aspect("equal"); axS.autoscale_view()

    fig.suptitle(
        f"Pad grid preview — {N_ACROSS}x{N_ALONG} points, {STEP*1000:.0f} mm step, "
        f"rotations={ROTATIONS_DEG}\n"
        f"cylinder: {CYL_LENGTH*1000:.0f} mm tall, {CYL_DIAMETER*1000:.0f} mm dia "
        f"| object: {OBJECT_ORIENT} | pad: {PAD_LONG*1000:.0f} x {PAD_SHORT*1000:.0f} mm",
        fontsize=10)
    fig.tight_layout()
    plt.show()

def _draw_pad(ax, cy, cz, w, h, rot_deg):
    """Draw one pad rectangle centered at (cy, cz), width w (Y), height h (Z),
    rotated by rot_deg in the plane."""
    t = patches.Rectangle((cy - w/2, cz - h/2), w, h,
                          facecolor="yellow", edgecolor="orange",
                          alpha=0.45, linewidth=1.0)
    # rotate about the rectangle center
    tr = patches.transforms.Affine2D().rotate_deg_around(cy, cz, rot_deg) + ax.transData
    t.set_transform(tr)
    ax.add_patch(t)

# ============================================================
# Print + preview when run in PyCharm
# ============================================================
if __name__ == "__main__":
    pts = build_pad_grid()
    print(f"Cylinder (fixed): center={CYL_CENTER}, {CYL_LENGTH*1000:.0f}mm tall, "
          f"{CYL_DIAMETER*1000:.0f}mm dia, axis {CYL_AXIS}")
    print(f"Pad grid: {N_ACROSS} across x {N_ALONG} along, step {STEP*1000:.0f}mm, "
          f"rotations={ROTATIONS_DEG}  -> {len(pts)} contact poses")
    print()
    print(f"{'row':>3} {'col':>3} {'rot':>5} {'X':>10} {'Y':>10} {'Z':>10}")
    print("-"*52)
    for p in pts:
        print(f"{p['row']:>3} {p['col']:>3} {p['rot']:>5.0f} "
              f"{p['x']:>10.5f} {p['y']:>10.5f} {p['z']:>10.5f}")
    print()
    print("Saving grid + scene record...")
    save_grid_and_scene()
    print()
    print("Opening preview window...")
    preview()
