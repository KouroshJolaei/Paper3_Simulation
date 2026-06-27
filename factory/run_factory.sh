#!/bin/bash
# ============================================================
# run_factory.sh — Tactile data factory (Option A)
#
# Launches Isaac Sim ONCE PER GRASP, each with a different pose.
# Each grasp gets a fresh session so the TSF extension writes
# clean CSV files with the correct name.
#
# HOW TO RUN:
#   chmod +x ~/Paper3_Simulation/run_factory.sh   (first time only)
#   ~/Paper3_Simulation/run_factory.sh
#
# OUTPUT:
#   ~/Paper3_Simulation/Data/run_<timestamp>/
#     grasp_001_cylinder_center_s1_tactile_maps.csv
#     grasp_001_cylinder_center_s1_deformations.csv
#     ... (6 files per grasp) ...
#     grasp_002_cylinder_left_5cm_s1_tactile_maps.csv
#     ...
#
# HOW TO ADD MORE POSES:
#   Add more lines to the GRASPS array below. Format per line:
#     "label|X|Y|Z|approach_height|gripper_close_rad"
# ============================================================

# --- Make sure CUDA 12.8 is on the path (needed by cuRobo) ---
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# --- Paths ---
PYTHON_SH="$HOME/isaacsim/python.sh"
GRASP_SCRIPT="$HOME/Paper3_Simulation/factory/grasp_one.py"
EXAMPLES_DIR="$HOME/Paper3_Simulation/TSF-85/examples"

# --- One output folder for this whole run ---
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$HOME/Paper3_Simulation/Data/run_${STAMP}"
mkdir -p "$RUN_DIR"
echo "============================================================"
echo "[factory] Output folder: $RUN_DIR"
echo "============================================================"

# --- Grasp configurations ---
# Format: "label|X|Y|Z|approach_height|gripper_close_rad"
GRASPS=(
  "cylinder_center|-0.26806|0.199|1.24244|0.10|0.55"
  "cylinder_left_5cm|-0.31806|0.199|1.24244|0.10|0.55"
  "cylinder_right_5cm|-0.21806|0.199|1.24244|0.10|0.55"
  "cylinder_forward_5cm|-0.26806|0.149|1.24244|0.10|0.55"
  "cylinder_higher_grasp|-0.26806|0.199|1.26244|0.12|0.50"
)

# --- Must run from examples dir so relative scene paths resolve ---
cd "$EXAMPLES_DIR" || { echo "Cannot cd to $EXAMPLES_DIR"; exit 1; }

# --- Loop ---
INDEX=0
SUCCEEDED=0
FAILED=0
for entry in "${GRASPS[@]}"; do
    INDEX=$((INDEX + 1))

    # Parse the pipe-separated fields
    IFS='|' read -r LABEL X Y Z APPROACH CLOSE <<< "$entry"

    # Zero-padded index for clean filenames
    IDX_PADDED=$(printf "%03d" "$INDEX")
    BASENAME="grasp_${IDX_PADDED}_${LABEL}"

    echo ""
    echo "============================================================"
    echo "[factory] Grasp $INDEX / ${#GRASPS[@]}: $LABEL"
    echo "[factory]   pose=($X, $Y, $Z)  approach=$APPROACH  close=$CLOSE"
    echo "[factory]   basename=$BASENAME"
    echo "============================================================"

    # Pass config to grasp_one.py via environment variables
    GRASP_LABEL="$LABEL" \
    GRASP_X="$X" \
    GRASP_Y="$Y" \
    GRASP_Z="$Z" \
    GRASP_APPROACH="$APPROACH" \
    GRASP_CLOSE_RAD="$CLOSE" \
    GRASP_OUTPUT_DIR="$RUN_DIR" \
    GRASP_BASENAME="$BASENAME" \
    "$PYTHON_SH" "$GRASP_SCRIPT"

    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[factory] Grasp $INDEX SUCCEEDED."
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "[factory] Grasp $INDEX FAILED (exit code $RC)."
        FAILED=$((FAILED + 1))
    fi
done

# --- Summary ---
echo ""
echo "============================================================"
echo "[factory] ALL DONE."
echo "[factory]   Succeeded: $SUCCEEDED"
echo "[factory]   Failed:    $FAILED"
echo "[factory]   Output:    $RUN_DIR"
echo "============================================================"
echo "[factory] Files written:"
ls -la "$RUN_DIR"
