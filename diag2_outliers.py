import json, sys, os, numpy as np
run = sys.argv[1]
ph = json.load(open(os.path.join(run,"pose_history.json")))
cfg=None
for c in ("gui_config_used.json","gui_config.json"):
    p=os.path.join(run,c)
    if os.path.exists(p): cfg=json.load(open(p)); break
ctr=cfg["object"]["center_world_mm"]
cmd={f"pt{int(p['index']):02d}":(ctr[1]+p["pad_offset_y_mm"],ctr[2]+p["pad_offset_z_mm"]) for p in cfg["points"]}
dY,dZ=[],[]
bad=[]
for p in ph["points"]:
    k=p["tag"]; a=p.get("pad_actual_pos_m")
    if k in cmd and a:
        gy=a[1]*1000-cmd[k][0]; gz=a[2]*1000-cmd[k][1]
        dY.append(gy); dZ.append(gz)
        if abs(gz-np.median([68.9])) > 5: bad.append((k, round(gy,1), round(gz,1)))
dY,dZ=np.array(dY),np.array(dZ)
print(f"dY: median={np.median(dY):.2f} spread={dY.max()-dY.min():.2f}")
print(f"dZ: median={np.median(dZ):.2f} spread={dZ.max()-dZ.min():.2f}")
print(f"dZ within 2mm of median: {np.mean(np.abs(dZ-np.median(dZ))<2)*100:.0f}% of points")
print(f"outlier points (|dZ-median|>5mm): {len(bad)}")
for b in bad[:12]: print("   ",b)
