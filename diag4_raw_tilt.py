# Does the RAW per-grasp data contain the tilt?
# For each grasp: find the hottest COLUMN (world-Y direction) inside the 7x4 map.
# If the cylinder is tilted, that hot column should shift as grid HEIGHT (Z) changes.
import json, sys, os, glob, re
import numpy as np, pandas as pd

run = sys.argv[1].strip().rstrip("/")
print(f"run: {run}")
if not os.path.isdir(run):
    print("  ERROR: not a directory"); sys.exit(1)

# --- find config, searching widely, and SAY which one ---
cfg = None; cfg_path = None
cands = [os.path.join(run, "gui_config_used.json"),
         os.path.join(run, "gui_config.json"),
         os.path.join(run, "..", "..", "gui_config.json"),
         os.path.expanduser("~/Paper3_Simulation/Data/gui_config.json")]
for p in cands:
    if os.path.exists(p):
        try:
            cfg = json.load(open(p)); cfg_path = p; break
        except Exception as e:
            print(f"  config parse error {p}: {e}")
if cfg is None:
    print("  ERROR: no config found. looked in:")
    for p in cands: print("   ", p)
    sys.exit(1)
print(f"config: {cfg_path}")

pts = cfg.get("points", [])
cmd = {f"pt{int(p['index']):02d}": (p["pad_offset_y_mm"], p["pad_offset_z_mm"]) for p in pts}
print(f"config has {len(cmd)} grid points")

files = sorted(glob.glob(os.path.join(run, "*_s1_tactile_maps.csv")))
print(f"found {len(files)} s1 tactile files")
if not files:
    print("  ERROR: no *_s1_tactile_maps.csv here. directory contains:")
    for f in sorted(os.listdir(run))[:20]: print("   ", f)
    sys.exit(1)

rows = []; skipped = []
for f in files:
    mm = re.search(r"pt(\d+)", os.path.basename(f))
    if not mm:
        skipped.append((os.path.basename(f), "no ptNN")); continue
    k = f"pt{int(mm.group(1)):02d}"
    if k not in cmd:
        skipped.append((k, "not in config")); continue
    df = pd.read_csv(f); pred = [c for c in df.columns if c.startswith("pred_")]
    if not pred:
        skipped.append((k, "no pred_ cols")); continue
    v = df[pred].to_numpy(); s = v.sum(1)
    if s.max() <= 0:
        skipped.append((k, "no pressure")); continue
    hold = v[s >= 0.5*s.max()].mean(0).reshape(7, 4)
    colp = hold.sum(0)
    hot_col = float((np.arange(4)*colp).sum()/colp.sum())
    rows.append((cmd[k][1], cmd[k][0], hot_col, float(hold.max())))

print(f"matched {len(rows)} grasps; skipped {len(skipped)}")
if skipped[:5]:
    print("  first skips:", skipped[:5])
if len(rows) < 3:
    print("  ERROR: too few matched grasps to analyze."); sys.exit(1)

rows.sort()
dz = np.array([r[0] for r in rows]); hotcol = np.array([r[2] for r in rows])
peak = np.array([r[3] for r in rows]); dy = np.array([r[1] for r in rows])
A = np.polyfit(dz, hotcol, 1)
print()
print(f"HOT-COLUMN vs grid-HEIGHT slope = {A[0]:+.4f} col/mm")
print(f"  tilt in data if this is clearly NONZERO (e.g. |slope|>0.02);")
print(f"  ~0 means each pad sees contact in the SAME column regardless of height")
print(f"peak pressure: min={peak.min():.0f} max={peak.max():.0f} "
      f"ratio={peak.max()/max(peak.min(),1):.1f}x")
# unique heights: is this a single column or a full grid?
print(f"distinct grid heights (Z offsets): {len(set(np.round(dz,1)))}, "
      f"distinct widths (Y offsets): {len(set(np.round(dy,1)))}")
