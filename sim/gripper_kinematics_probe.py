"""
gripper_kinematics_probe.py  --  READ-ONLY finger-linkage dump

Walks the modified Robotiq 2F-85 gripper USD and records, for every joint and
link in the finger chain: joint type, axis, drive target/limits, parent/child
bodies, and each prim's local translate/orient. This is the geometry we need to
compute the pad drop d_swing(theta) by forward kinematics instead of by eye.

SAFE: boots a minimal headless app only so pxr imports. No World, no physics
step, no articulation, no gripper close -> nothing can explode. Opens the USD
as a plain stage and reads authored attributes.

Run:
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/gripper_kinematics_probe.py
  cat ~/Paper3_Simulation/Data/gripper_kinematics_probe_summary.txt
"""

import os, json, traceback

USD_PATH = "/home/kourosh/Paper3_Simulation/TSF-85/assets/Robotiq_2F_85_modified.usd"
OUT_JSON = os.path.expanduser("~/Paper3_Simulation/Data/gripper_kinematics_probe.json")
OUT_TXT  = os.path.expanduser("~/Paper3_Simulation/Data/gripper_kinematics_probe_summary.txt")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_TXT, "w") as _f:
    _f.write("PROBE STARTED (if only this line remains, it crashed early)\n")

from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Sdf


def _get(prim, name):
    a = prim.GetAttribute(name)
    if a and a.IsValid() and a.Get() is not None:
        v = a.Get()
        if isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return str(v)
    return None


def _rel_targets(prim, name):
    r = prim.GetRelationship(name)
    if r and r.IsValid():
        return [str(t) for t in r.GetTargets()]
    return []


def local_xform(prim):
    t = _get(prim, "xformOp:translate")
    o = None
    a = prim.GetAttribute("xformOp:orient")
    if a and a.IsValid() and a.Get() is not None:
        q = a.Get()
        try:
            o = [float(q.GetReal()),
                 float(q.GetImaginary()[0]),
                 float(q.GetImaginary()[1]),
                 float(q.GetImaginary()[2])]
        except Exception:
            o = str(q)
    return {"translate": t, "orient_wxyz": o}


def run():
    if not os.path.exists(USD_PATH):
        raise FileNotFoundError(f"USD not found: {USD_PATH}")
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError("Usd.Stage.Open returned None")

    joints, links = [], []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        tname = str(prim.GetTypeName())

        # --- joints ---------------------------------------------------------
        if prim.IsA(UsdPhysics.Joint):
            j = {
                "path": path,
                "type": tname,
                "body0": _rel_targets(prim, "physics:body0"),
                "body1": _rel_targets(prim, "physics:body1"),
                "localPos0":  _get(prim, "physics:localPos0"),
                "localPos1":  _get(prim, "physics:localPos1"),
                "localRot0":  _get(prim, "physics:localRot0"),
                "localRot1":  _get(prim, "physics:localRot1"),
                "axis":       _get(prim, "physics:axis"),
                "lowerLimit": _get(prim, "physics:lowerLimit"),
                "upperLimit": _get(prim, "physics:upperLimit"),
            }
            # revolute/prismatic axis may live under drive/limit APIs; grab common ones
            for extra in ("drive:angular:physics:targetPosition",
                          "drive:linear:physics:targetPosition",
                          "physxJoint:jointFriction"):
                v = _get(prim, extra)
                if v is not None:
                    j[extra] = v
            joints.append(j)

        # --- candidate finger/knuckle links --------------------------------
        ln = prim.GetName().lower()
        if any(k in ln for k in ("finger", "knuckle", "coupl", "pad",
                                 "base_link", "adapter", "tsf")):
            if tname in ("Xform", "Mesh"):
                links.append({
                    "path": path, "type": tname, "name": prim.GetName(),
                    "local": local_xform(prim),
                })

    out = {"usd_path": USD_PATH,
           "n_joints": len(joints), "n_links": len(links),
           "joints": joints, "links": links}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    L = ["=" * 72,
         f"gripper_kinematics_probe OK: {len(joints)} joints, {len(links)} links",
         f"json: {OUT_JSON}", "-" * 72, "JOINTS:"]
    for j in joints:
        L.append(f"  {j['path']}")
        L.append(f"     type={j['type']} axis={j.get('axis')} "
                 f"limits=({j.get('lowerLimit')},{j.get('upperLimit')})")
        L.append(f"     body0={j['body0']}  localPos0={j.get('localPos0')}")
        L.append(f"     body1={j['body1']}  localPos1={j.get('localPos1')}")
    L.append("-" * 72)
    L.append("LINKS (finger/knuckle/pad/adapter):")
    for k in links:
        L.append(f"  {k['name']:24s} {k['type']:6s} "
                 f"translate={k['local']['translate']}  {k['path']}")
    L.append("=" * 72)
    txt = "\n".join(L)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print(txt, flush=True)


try:
    run()
except Exception:
    tb = traceback.format_exc()
    with open(OUT_TXT, "w") as f:
        f.write("PROBE FAILED:\n\n" + tb + "\n")
    print("PROBE FAILED:\n" + tb, flush=True)
finally:
    _app.close()
