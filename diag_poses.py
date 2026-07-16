import json, sys, statistics, glob, os
run = sys.argv[1]
ph = json.load(open(os.path.join(run,"pose_history.json")))
# find the config the same way stitching does
cfg=None
for c in ("gui_config_used.json","gui_config.json","../../gui_config.json"):
    p=os.path.join(run,c)
    if os.path.exists(p):
        cfg=json.load(open(p)); print("config:",p); break
ctr=cfg["object"]["center_world_mm"]
cmd={f"pt{int(p['index']):02d}":(ctr[1]+p["pad_offset_y_mm"],ctr[2]+p["pad_offset_z_mm"]) for p in cfg["points"]}
rows=[]
for p in ph["points"]:
    k=p["tag"]; a=p.get("pad_actual_pos_m")
    if k in cmd and a:
        ay,az=a[1]*1000,a[2]*1000
        rows.append((k, cmd[k][1], az, az-cmd[k][1]))  # commanded Z, actual Z, dZ
rows.sort(key=lambda r:r[1])
print(f"{'pt':6}{'cmd_Z':>9}{'act_Z':>9}{'dZ':>8}")
for k,cz,az,dz in rows[:6]+rows[-6:]:
    print(f"{k:6}{cz:9.1f}{az:9.1f}{dz:8.2f}")
dz=[r[3] for r in rows]
print(f"\ndZ: min={min(dz):.2f} max={max(dz):.2f} spread={max(dz)-min(dz):.2f} std={statistics.pstdev(dz):.2f}")
# is dZ correlated with commanded Z? (a slope = systematic, not noise)
import numpy as np
cz=np.array([r[1] for r in rows]); dzv=np.array(dz)
slope=np.polyfit(cz,dzv,1)[0]
print(f"dZ-vs-cmdZ slope = {slope:.4f} mm/mm  ({'SYSTEMATIC droop' if abs(slope)>0.02 else 'looks like noise'})")
