"""
scene_probe.py  --  READ-ONLY scene geometry probe  (Step 1a, v3)

Reads the collision-relevant geometry (table, ground, object, mount) from the USD
so we can build a cuRobo collision world from REAL numbers.

WHY v3: v2 crashed silently inside BBoxCache.ComputeWorldBound (unreliable headless
with no renderer). v3 computes each prim's world AABB from its authored `extent`
attribute + world transform (renderer-free, robust) and writes ANY error to
scene_probe_summary.txt so nothing is ever swallowed again.

SAFE: boots a minimal headless app only so 'pxr' imports. No World, no robot,
no physics step, no gripper close -> the close-time NaN cannot occur.

Run:
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/scene_probe.py
  cat ~/Paper3_Simulation/Data/scene_probe_summary.txt
"""

import os, json, traceback

USD_PATH = "/home/kourosh/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd"
OUT_JSON = os.path.expanduser("~/Paper3_Simulation/Data/scene_probe.json")
OUT_TXT  = os.path.expanduser("~/Paper3_Simulation/Data/scene_probe_summary.txt")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

# Write an immediate marker so we KNOW the output path works, even if we crash.
with open(OUT_TXT, "w") as _f:
    _f.write("PROBE STARTED (if you see only this line, it crashed before finishing)\n")

from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf
try:
    from pxr import UsdPhysics
    _HAVE_PHYS = True
except Exception:
    _HAVE_PHYS = False

NAME_HINTS = ("table", "desk", "ground", "floor", "plane", "support",
              "mount", "pedestal", "stand", "object", "cylinder", "wall", "box")
GEOM_TYPES = ("Mesh", "Cube", "Cylinder", "Sphere", "Capsule", "Cone", "Plane")


def world_aabb(xcache, prim):
    """World AABB (meters) from authored `extent` + world transform. Renderer-free."""
    ext_attr = prim.GetAttribute("extent")
    ext = ext_attr.Get() if (ext_attr and ext_attr.HasAuthoredValue()) else None
    if not ext:
        return None, None
    mn, mx = ext[0], ext[1]
    M = xcache.GetLocalToWorldTransform(prim)
    corners = [Gf.Vec3d(x, y, z)
               for x in (mn[0], mx[0])
               for y in (mn[1], mx[1])
               for z in (mn[2], mx[2])]
    wc = [M.Transform(c) for c in corners]
    xs = [p[0] for p in wc]; ys = [p[1] for p in wc]; zs = [p[2] for p in wc]
    return ([float(min(xs)), float(min(ys)), float(min(zs))],
            [float(max(xs)), float(max(ys)), float(max(zs))])


def row_for(prim, mn, mx):
    size_mm = [round((mx[i] - mn[i]) * 1000.0, 1) for i in range(3)]
    ctr_mm  = [round((mx[i] + mn[i]) * 500.0, 1) for i in range(3)]
    return {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
            "center_mm": ctr_mm, "size_mm": size_mm,
            "aabb_min_m": [round(v, 4) for v in mn],
            "aabb_max_m": [round(v, 4) for v in mx]}


def run():
    if not os.path.exists(USD_PATH):
        raise FileNotFoundError(f"USD not found: {USD_PATH}")
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {USD_PATH}")
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    all_geoms, candidates, robot_paths, no_extent = [], [], [], []
    for prim in stage.Traverse():
        path  = str(prim.GetPath()); lname = prim.GetName().lower()
        lpath = path.lower();        tname = str(prim.GetTypeName())
        is_geom = tname in GEOM_TYPES
        is_coll = _HAVE_PHYS and prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (is_geom or is_coll):
            continue
        try:
            mn, mx = world_aabb(xcache, prim)
        except Exception as e:
            no_extent.append(f"{path}  ({tname})  ERR {e}")
            continue
        if mn is None:
            no_extent.append(f"{path}  ({tname})  no authored extent")
            continue
        r = row_for(prim, mn, mx); all_geoms.append(r)
        if any(k in lpath for k in ("robotiq", "wrist", "shoulder", "elbow",
                                    "forearm", "upper_arm", "finger", "tsf_85",
                                    "adapter", "tool0", "flange", "base_link")):
            robot_paths.append(path)
        dx, dy, dz = r["size_mm"]
        big_flat = (max(dx, dy) > 300.0 and dz < 120.0)
        name_hit = any(h in lname or h in lpath for h in NAME_HINTS)
        if name_hit or big_flat:
            r2 = dict(r)
            r2["why"] = ("name" if name_hit else "") + ("+flat" if big_flat else "")
            candidates.append(r2)

    out = {"usd_path": USD_PATH, "have_usdphysics": _HAVE_PHYS,
           "n_geom_prims": len(all_geoms), "candidates": candidates,
           "robot_like_paths": sorted(set(robot_paths)),
           "prims_without_extent": no_extent, "all_geoms": all_geoms}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    L = []
    L.append("=" * 64)
    L.append(f"scene_probe OK: {len(all_geoms)} geom prims measured, "
             f"{len(no_extent)} skipped (no extent)")
    L.append(f"json: {OUT_JSON}")
    L.append("-" * 64)
    L.append("CANDIDATE OBSTACLES  size_mm=[dx,dy,dz]  center_mm=[x,y,z]:")
    if not candidates:
        L.append("  (none matched -- see all_geoms / prims_without_extent in JSON)")
    for c in candidates:
        L.append(f"  {c['path']}")
        L.append(f"     type={c['type']}  size_mm={c['size_mm']}  "
                 f"center_mm={c['center_mm']}  ({c.get('why','')})")
    L.append("=" * 64)
    txt = "\n".join(L)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print(txt, flush=True)


try:
    run()
except Exception:
    tb = traceback.format_exc()
    with open(OUT_TXT, "w") as f:
        f.write("PROBE FAILED with traceback:\n\n" + tb + "\n")
    print("PROBE FAILED:\n" + tb, flush=True)
finally:
    _app.close()
