#!/bin/bash
# ============================================================
# run_grid_from_csv.sh — collect the grid that scene_config.py saved.
#
# Reads:  ~/Paper3_Simulation/sim/current_grid.csv
#         (written by running scene_config.py in PyCharm)
# Runs:   one fresh Isaac Sim per grid point, using grasp_one_grid_v2.py
# Writes: ~/Paper3_Simulation/Data/grid_<timestamp>/grid_rXX_cXX_...
#
# This is the SAME grid you previewed — no numbers typed twice.
#
# RUN:
#   chmod +x ~/Paper3_Simulation/factory/run_grid_from_csv.sh   (first time)
#   ~/Paper3_Simulation/factory/run_grid_from_csv.sh
# ============================================================

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

PYTHON_SH="$HOME/isaacsim/python.sh"
GRID_SCRIPT="$HOME/Paper3_Simulation/sim/grasp_one_grid_v2.py"
GRID_CSV="$HOME/Paper3_Simulation/sim/current_grid.csv"
EXAMPLES_DIR="$HOME/Paper3_Simulation/TSF-85/examples"

if [ ! -f "$GRID_CSV" ]; then
    echo "ERROR: $GRID_CSV not found."
    echo "Run scene_config.py in PyCharm first to create it."
    exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$HOME/Paper3_Simulation/Data/grid_${STAMP}"
mkdir -p "$RUN_DIR"
echo "============================================================"
echo "[grid] Reading grid from: $GRID_CSV"
echo "[grid] Output folder:     $RUN_DIR"
echo "============================================================"

# Save a copy of the exact grid + scene record alongside the data,
# so this run is fully reproducible later (real-robot replay).
cp "$GRID_CSV" "$RUN_DIR/" 2>/dev/null
cp "$HOME/Paper3_Simulation/sim/current_scene.json" "$RUN_DIR/" 2>/dev/null

# Read the object pose from current_scene.json so every grasp bolts the object
# at the pose you set in scene_config.py. Falls back to "keep" (standing) if the
# record is missing or python isn't available here.
SCENE_JSON="$HOME/Paper3_Simulation/sim/current_scene.json"
OBJ_ORIENT="keep"; OBJ_POS_X=""; OBJ_POS_Y=""; OBJ_POS_Z=""; OBJ_TILT_DEG="20"
if [ -f "$SCENE_JSON" ]; then
    read OBJ_ORIENT OBJ_POS_X OBJ_POS_Y OBJ_POS_Z OBJ_TILT_DEG < <(python3 - "$SCENE_JSON" <<'PYEOF'
import json, sys
try:
    o = json.load(open(sys.argv[1]))["object"]
    pos = o.get("pose_world", [ "", "", "" ])
    print(o.get("orientation","keep"),
          pos[0] if len(pos)>0 else "",
          pos[1] if len(pos)>1 else "",
          pos[2] if len(pos)>2 else "",
          o.get("tilt_deg", 20))
except Exception:
    print("keep", "", "", "", 20)
PYEOF
)
fi
echo "[grid] object: orient=$OBJ_ORIENT pos=($OBJ_POS_X,$OBJ_POS_Y,$OBJ_POS_Z) tilt=$OBJ_TILT_DEG"

cd "$EXAMPLES_DIR" || { echo "Cannot cd to $EXAMPLES_DIR"; exit 1; }

TOTAL=$(( $(wc -l < "$GRID_CSV") - 1 ))   # minus header
echo "[grid] $TOTAL points to collect."
echo ""

INDEX=0
SUCCEEDED=0
FAILED=0

# Read CSV, skipping the header line
tail -n +2 "$GRID_CSV" | while IFS=',' read -r LABEL X Y Z APPROACH CLOSE ROT; do
    [ -z "$LABEL" ] && continue
    INDEX=$((INDEX + 1))

    echo "============================================================"
    echo "[grid] Point $INDEX / $TOTAL : $LABEL"
    echo "[grid]   EE=($X, $Y, $Z)  rot=$ROT deg"
    echo "============================================================"

    GRASP_LABEL="$LABEL" \
    GRASP_X="$X" GRASP_Y="$Y" GRASP_Z="$Z" \
    GRASP_APPROACH="$APPROACH" GRASP_CLOSE_RAD="$CLOSE" \
    GRASP_ROT_DEG="$ROT" GRASP_ROT_AXIS="z" \
    GRASP_OUTPUT_DIR="$RUN_DIR" \
    GRASP_BASENAME="$LABEL" \
    OBJ_ORIENT="$OBJ_ORIENT" OBJ_TILT_DEG="$OBJ_TILT_DEG" \
    OBJ_POS_X="$OBJ_POS_X" OBJ_POS_Y="$OBJ_POS_Y" OBJ_POS_Z="$OBJ_POS_Z" \
    "$PYTHON_SH" "$GRID_SCRIPT" > "$RUN_DIR/${LABEL}_run.log" 2>&1
    if grep -q "\[grid\] SUCCESS" "$RUN_DIR/${LABEL}_run.log"; then
        echo "  -> SUCCESS"
    else
        echo "  -> (check $RUN_DIR/${LABEL}_run.log)"
    fi

    echo ""
done

echo "============================================================"
echo "[grid] DONE. Output: $RUN_DIR"
echo "============================================================"
ls -la "$RUN_DIR" | grep tactile_maps
