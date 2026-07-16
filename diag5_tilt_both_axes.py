# Track BOTH the hot ROW and hot COLUMN inside each 7x4 pad vs grid position.
# Tilt should make the in-pad hot spot move as we scan. diag4 only looked at
# columns (Y) and missed it; the raw heatmaps show motion mostly in ROWS (Z).
import json, sys, os, glob, re
import numpy as np, pandas as pd
run = sys.argv[1].strip().rstrip("/")
cfg=None; cfgp=None
for c in ("gui_config_used.json","gui_config.json","../../gui_config.json",
          os.path.expanduser("~/Paper3_Simulation/Data/gui_config.json")):
    p=os.path.join(run,c) if not c.startswith("/") else c
    if os.path.exists(p): cfg=json.load(open(p)); cfgp=p; break
cmd={f"pt{int(p['index']):02d}":(p["pad_offset_y_mm"],p["pad_offset_z_mm"]) for p in cfg["points"]}
print(f"config: {cfgp}  ({len(cmd)} pts)")
rows=[]
for f in sorted(glob.glob(os.path.join(run,"*_s1_tactile_maps.csv"))):
    mm=re.search(r"pt(\d+)",os.path.basename(f))
    if not mm: continue
    k=f"pt{int(mm.group(1)):02d}"
    if k not in cmd: continue
    df=pd.read_csv(f); pred=[c for c in df.columns if c.startswith("pred_")]
    v=df[pred].to_numpy(); s=v.sum(1)
    if s.max()<=0: continue
    hold=v[s>=0.5*s.max()].mean(0).reshape(7,4)
    rp=hold.sum(1); cp=hold.sum(0)
    hot_row=float((np.arange(7)*rp).sum()/rp.sum())     # 0=top ..6=bottom
    hot_col=float((np.arange(4)*cp).sum()/cp.sum())     # 0..3 along Y
    rows.append((cmd[k][0],cmd[k][1],hot_row,hot_col))  # dy, dz, hotrow, hotcol
dy=np.array([r[0] for r in rows]); dz=np.array([r[1] for r in rows])
hr=np.array([r[2] for r in rows]); hc=np.array([r[3] for r in rows])
print(f"grasps: {len(rows)}")
# how does in-pad hot ROW move as grid Z changes, and hot COL as grid Y changes?
print(f"hot-ROW vs grid-Z slope = {np.polyfit(dz,hr,1)[0]:+.4f} row/mm  "
      f"(expect ~ -1/pitchZ = -0.19 if pad just tracks height rigidly)")
print(f"hot-COL vs grid-Y slope = {np.polyfit(dy,hc,1)[0]:+.4f} col/mm  "
      f"(expect ~ -1/pitchY = -0.18 if pad just tracks sideways rigidly)")
# THE TILT TEST: at fixed grid-Y, does hot spot's WORLD-Y position move with Z?
# world contact Y ~ dy + (hot_col-1.5)*pitchY ; if tilted, this vs dz has slope=tan(tilt)
pitchY=22/4; pitchZ=37/7
contactY = dy + (hc-1.5)*pitchY
sl=np.polyfit(dz,contactY,1)[0]
print(f"\nTILT SIGNATURE: world-contact-Y vs grid-Z slope = {sl:+.4f} mm/mm "
      f"-> angle {np.degrees(np.arctan(abs(sl))):.1f} deg  (expect ~20)")
