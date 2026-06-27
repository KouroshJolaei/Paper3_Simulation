#!/bin/bash
# ============================================================
# run_grid.sh — Grid-scan data factory
#
# Reads the grid from sim/grid_config.py, then launches ONE fresh
# Isaac Sim per grid point (the proven, bulletproof pattern).
# Each point writes its own clean CSV set named grid_rXX_cXX_...
#
# HOW TO RUN:
#   chmod +x ~/Paper3_Simulation/factory/run_grid.sh   (first time only)
#   ~/Paper3_Simulation/factory/run_grid.sh
#
# TO CHANGE THE GRID:
#   Edit ~/Paper3_Simulation/sim/grid_config.py  (grid size, step, center,
#   rotations) — you do NOT edit this script.
# ============================================================

# --- CUDA 12.8 on the path (needed by cuRobo) ---
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# --- Paths ---
PYTHON_SH="$HOME/isaacsim/python.sh"
SIM_DIR="$HOME/Paper3_Simulation/sim"
GRID_SCRIPT="$SIM_DIR/grasp_one_grid.py"
CONFIG_DIR="$SIM_DIR"
EXAMPLES_DIR="$HOME/Paper3_Simulation/TSF-85/examples"

# --- One output folder for this whole grid run ---
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$HOME/Paper3_Simulation/Data/grid_${STAMP}"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "[grid-factory] Output folder: $RUN_DIR"
echo "============================================================"

# --- Ask grid_config.py for the list of points (pure Python, no Isaac Sim) ---
# Each line printed is:  label|X|Y|Z|approach|close|rot_deg
# We use the SYSTEM python3 here (not Isaac's) since grid_config.py is pure Python.
GRID_LINES=$(cd "$CONFIG_DIR" && python3 -c "
import grid_config as g
for p in g.build_grid():
    print(f\"{p['label']}|{p['x']}|{p['y']}|{p['z']}|{p['approach']}|{p['close']}|{p['rot_deg']}\")
")

if [ -z "$GRID_LINES" ]; then
    echo "[grid-factory] ERROR: grid_config.py produced no points. Check the config."
    exit 1
fi

# Count points
TOTAL=$(echo "$GRID_LINES" | wc -l)
echo "[grid-factory] Grid has $TOTAL points."
echo ""

# --- Must run from examples dir so relative scene paths resolve ---
cd "$EXAMPLES_DIR" || { echo "Cannot cd to $EXAMPLES_DIR"; exit 1; }

# --- Loop over grid points ---
INDEX=0
SUCCEEDED=0
FAILED=0
while IFS='|' read -r LABEL X Y Z APPROACH CLOSE ROT; do
    [ -z "$LABEL" ] && continue
    INDEX=$((INDEX + 1))

    echo "============================================================"
    echo "[grid-factory] Point $INDEX / $TOTAL : $LABEL"
    echo "[grid-factory]   pose=($X, $Y, $Z)  rot=$ROT deg  close=$CLOSE"
    echo "============================================================"

    GRASP_LABEL="$LABEL" \
    GRASP_X="$X" \
    GRASP_Y="$Y" \
    GRASP_Z="$Z" \
    GRASP_APPROACH="$APPROACH" \
    GRASP_CLOSE_RAD="$CLOSE" \
    GRASP_ROT_DEG="$ROT" \
    GRASP_OUTPUT_DIR="$RUN_DIR" \
    GRASP_BASENAME="$LABEL" \
    "$PYTHON_SH" "$GRID_SCRIPT"

    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[grid-factory] Point $INDEX SUCCEEDED."
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "[grid-factory] Point $INDEX FAILED (exit $RC)."
        FAILED=$((FAILED + 1))
    fi
    echo ""
done <<< "$GRID_LINES"

# --- Summary ---
echo "============================================================"
echo "[grid-factory] ALL DONE."
echo "[grid-factory]   Succeeded: $SUCCEEDED"
echo "[grid-factory]   Failed:    $FAILED"
echo "[grid-factory]   Output:    $RUN_DIR"
echo "============================================================"
ls -la "$RUN_DIR"
