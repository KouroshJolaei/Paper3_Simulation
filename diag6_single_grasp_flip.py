# Single tilted grasp = clean orientation check (no overlap to hide errors).
# For each sensor, the hot contact's WORLD position must land ON the tilted
# cylinder's crest line. We test all 4 flips and report which puts the hot
# taxel closest to the rod crest at the pad's height.
import json, sys, os, glob, re
import numpy as np, pandas as pd

run = sys.argv[1].strip().rstrip("/")
cfg=None
for c in ("gui_config_used.json","gui_config.json","../../gui_config.json",
          os.path.expanduser("~/Paper3_Simulation/Data/gui_config.json")):
    p=os.path.join(run,c) if not c.startswith("/") else c
    if os.path.exists(p): cfg=json.load(open(p)); break
obj=cfg["object"]; ctr=obj["center_world_mm"]
tilt=np.deg2rad(float(obj.get("tilt_deg",0.0)))
cmd={f"pt{int(p['index']):02d}":(p["pad_offset_y_mm"],p["pad_offset_z_mm"]) for p in cfg["points"]}
PY,PZ=22/4,37/7; NR,NC=7,4

def hot_taxel(fp):
    df=pd.read_csv(fp); pred=[c for c in df.columns if c.startswith("pred_")]
    v=df[pred].to_numpy(); s=v.sum(1)
    if s.max()<=0: return None
    m=v[s>=0.5*s.max()].mean(0).reshape(NR,NC)
    r,c=np.unravel_index(np.argmax(m),m.shape)
    return r,c,m

print(f"tilt = {np.degrees(tilt):.0f} deg about X; object centre Y={ctr[1]}, Z={ctr[2]}")
for sensor in ("s1","s2"):
    fs=[f for f in glob.glob(os.path.join(run,f"*_{sensor}_tactile_maps.csv"))
        if re.search(r"pt\d+",os.path.basename(f))]
    if not fs: print(f"{sensor}: no file"); continue
    fp=fs[0]
    _n=int(re.search(r"pt(\d+)", os.path.basename(fp)).group(1))
    k="pt%02d" % _n
    ht=hot_taxel(fp)
    if ht is None: print(f"{sensor}: no pressure"); continue
    r,c,m=ht
    dy,dz=cmd[k]                       # pad centre offset from object centre (mm)
    padY=ctr[1]+dy; padZ=ctr[2]+dz
    print(f"\n{sensor}: hot taxel (row={r}, col={c}) at pad centre "
          f"Y={padY:.1f} Z={padZ:.1f}")
    print(f"   {'flip_lr':>7} {'flip_ud':>7} {'worldY':>8} {'worldZ':>8} "
          f"{'crestY@Z':>9} {'|err|':>7}")
    best=None
    for flr in (False,True):
        for fud in (False,True):
            ys=(np.arange(NC)-(NC-1)/2)*PY
            zs=((NR-1)/2-np.arange(NR))*PZ
            if flr: ys=ys[::-1]
            if fud: zs=zs[::-1]
            wy=padY+ys[c]; wz=padZ+zs[r]
            crestY=ctr[1]-np.tan(tilt)*(wz-ctr[2])   # rod crest Y at this height
            err=abs(wy-crestY)
            tag=""
            if best is None or err<best[0]: best=(err,flr,fud); 
            print(f"   {str(flr):>7} {str(fud):>7} {wy:8.1f} {wz:8.1f} "
                  f"{crestY:9.1f} {err:7.1f}")
    print(f"   -> BEST for {sensor}: flip_lr={best[1]}, flip_ud={best[2]} "
          f"(hot taxel lands on the rod crest)")
