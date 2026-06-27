"""
grid_config.py — THE ONE FILE YOU EDIT to control a grid scan.

This is pure Python (no Isaac Sim). You can open it in PyCharm and run it
directly to PRINT the grid and check it looks right before launching the sim.

WHAT THE GRID IS
----------------
The gripper touches the cylinder at a CENTER point (your proven grasp point),
then steps across a 2D grid on the finger-pad face:
  - X steps = sideways ACROSS the can
  - Y steps = up/down ALONG the can's length
At each grid point it does a full discrete touch: approach, descend, close,
record, open, ascend — then moves in free space to the next point.

HOW TO USE
----------
1. Edit the values in the CONFIG section below.
2. (Optional) Run this file in PyCharm to print the grid and sanity-check it.
3. Run the launcher:  ~/Paper3_Simulation/factory/run_grid.sh
"""

import numpy as np

# ============================================================
# CONFIG  — edit these
# ============================================================

# --- Grid CENTER = your proven working grasp point (world meters) ---
# This is the (X, Y, Z) end-effector target your single grasp already hits
# dead-center on the cylinder. The grid is built as offsets around this.
CENTER_X = -0.26806
CENTER_Y =  0.199
CENTER_Z =  1.24244

# --- Approach + grip (same values your working grasp used) ---
APPROACH_HEIGHT  = 0.10   # how far above the grasp point to start the descent
GRIPPER_CLOSE    = 0.55   # gripper close target (radians)

# --- Grid SIZE ---
# Number of points along each axis. 3 -> a 3x3 grid (9 points).
# Use odd numbers so the center point is included.
N_X = 3        # points sideways ACROSS the can
N_Y = 3        # points up/down ALONG the can

# --- Grid STEP (meters) ---
# Distance between neighboring grid points. 0.004 = 4 mm (Paper 1 used ~4 mm).
STEP_X = 0.004   # step size across the can
STEP_Y = 0.004   # step size along the can

# --- Which world axes the grid moves in ---
# From the scene: the cylinder stands up in Z, pads close in X.
# So the pad face spans Y (across) and Z (along the can length).
#   X-grid-axis -> world Y   (sideways across the can)
#   Y-grid-axis -> world Z   (up/down the can)
# You normally do NOT need to change these for the standing cylinder.
GRID_AXIS_X_WORLD = "Y"   # 'X', 'Y', or 'Z'
GRID_AXIS_Y_WORLD = "Z"

# --- Rotation of the fingers at each point (degrees) ---
# Start with [0] = no rotation (validate x/y motion first).
# Later you can add rotations, e.g. [0, 15, -15], and EACH grid point
# will be repeated once per rotation angle.
ROTATIONS_DEG = [0]

# ============================================================
# END CONFIG
# ============================================================


def build_grid():
    """Return a list of grid points. Each point is a dict with everything
    the sim script needs. Pure data — no Isaac Sim involved."""
    points = []

    # Center the grid: indices run from -(N-1)/2 .. +(N-1)/2
    xs = [(-(N_X - 1) / 2.0 + i) * STEP_X for i in range(N_X)]
    ys = [(-(N_Y - 1) / 2.0 + j) * STEP_Y for j in range(N_Y)]

    axis_to_index = {"X": 0, "Y": 1, "Z": 2}
    ax = axis_to_index[GRID_AXIS_X_WORLD]
    ay = axis_to_index[GRID_AXIS_Y_WORLD]

    for r, dy in enumerate(ys):          # row index = along-can (Y grid axis)
        for c, dx in enumerate(xs):      # col index = across-can (X grid axis)
            for rot in ROTATIONS_DEG:
                # Start from the center, then add the offset on the right world axes
                world = [CENTER_X, CENTER_Y, CENTER_Z]
                world[ax] += dx
                world[ay] += dy

                # Build a clean label/prefix for filenames
                if len(ROTATIONS_DEG) > 1:
                    label = f"grid_r{r:02d}_c{c:02d}_rot{int(rot):+03d}"
                else:
                    label = f"grid_r{r:02d}_c{c:02d}"

                points.append({
                    "label":     label,
                    "row":       r,
                    "col":       c,
                    "rot_deg":   float(rot),
                    "x":         round(world[0], 6),
                    "y":         round(world[1], 6),
                    "z":         round(world[2], 6),
                    "approach":  APPROACH_HEIGHT,
                    "close":     GRIPPER_CLOSE,
                })
    return points


def print_grid():
    """Print the grid as a readable table. Run this file in PyCharm to use it."""
    pts = build_grid()
    print(f"\nGrid: {N_X} x {N_Y}  (step {STEP_X*1000:.1f} x {STEP_Y*1000:.1f} mm), "
          f"rotations={ROTATIONS_DEG}")
    print(f"Center = ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
    print(f"Total points = {len(pts)}\n")
    print(f"{'label':<26} {'row':>3} {'col':>3} {'rot':>5} "
          f"{'X':>10} {'Y':>10} {'Z':>10}")
    print("-" * 80)
    for p in pts:
        print(f"{p['label']:<26} {p['row']:>3} {p['col']:>3} {p['rot_deg']:>5.0f} "
              f"{p['x']:>10.5f} {p['y']:>10.5f} {p['z']:>10.5f}")
    print()


if __name__ == "__main__":
    # Running this file directly (e.g. in PyCharm) just prints the grid.
    print_grid()
