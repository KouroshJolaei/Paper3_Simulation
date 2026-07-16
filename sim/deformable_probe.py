"""
deformable_probe.py  --  READ-ONLY physics-attribute probe  (NaN investigation)

Dumps the physics setup (applied schemas + mass / density / inertia / deformable /
collision attributes) of the TSF-85 sensor prims and the finger links, so we can
see whether the recurring "invalid inertia tensor / negative mass" warning is a
real bad value we can fix.

SAFE: boots a minimal headless app ONLY so pxr imports. No World, no robot spawn,
no physics step, no gripper close -> the close-time NaN cannot occur. Opens the
USD as a plain stage and reads authored attributes.

Run:
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/deformable_probe.py
  cat ~/Paper3_Simulation/Data/deformable_probe_summary.txt
"""

import os, json, traceback

USD_PATH = "/home/kourosh/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd"
OUT_JSON = os.path.expanduser("~/Paper3_Simulation/Data/deformable_probe.json")
OUT_TXT  = os.path.expanduser("~/Paper3_Simulation/Data/deformable_probe_summary.txt")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_TXT, "w") as _f:
    _f.write("PROBE STARTED (if only this line remains, it crashed early)\n")

from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

from pxr import Usd

# Which prims we care about: anything named TSF_85, and the finger links.
def is_target(path, name):
    p, n = path.lower(), name.lower()
    if "tsf_85" in p:
        return True
    if n.endswith("inner_finger") or n.endswith("outer_finger"):
        return True
    if n.endswith("inner_knuckle") or n.endswith("outer_knuckle"):
        return True
    return False

# Attribute name keywords worth reporting.
ATTR_KEYS = ("mass", "density", "inertia", "deformable", "physx", "collision",
             "rigid", "physics", "restpoint", "rest_point", "point", "youngs",
             "poisson", "damping", "stiffness", "solver", "kinematic")


def short_val(v):
    """Stringify a value, truncating big arrays."""
    try:
        n = len(v)
        if n > 8:
            head = ", ".join(str(x) for x in list(v)[:3])
            return f"<array len={n}: [{head}, ...]>"
        return str(list(v))
    except TypeError:
        return str(v)


def dump_prim(prim):
    path = str(prim.GetPath())
    info = {
        "path": path,
        "type": str(prim.GetTypeName()),
        "applied_schemas": [str(s) for s in prim.GetAppliedSchemas()],
        "n_children": len(list(prim.GetChildren())),
        "attrs": {},
    }
    for a in prim.GetAttributes():
        an = a.GetName().lower()
        if not any(k in an for k in ATTR_KEYS):
            continue
        if not a.HasAuthoredValue():
            continue
        try:
            info["attrs"][a.GetName()] = short_val(a.Get())
        except Exception as e:
            info["attrs"][a.GetName()] = f"ERR {e}"
    return info


def run():
    if not os.path.exists(USD_PATH):
        raise FileNotFoundError(f"USD not found: {USD_PATH}")
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError("Usd.Stage.Open returned None")

    targets = []
    for prim in stage.Traverse():
        if is_target(str(prim.GetPath()), prim.GetName()):
            targets.append(dump_prim(prim))

    out = {"usd_path": USD_PATH, "n_targets": len(targets), "targets": targets}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    L = ["=" * 70,
         f"deformable_probe OK: {len(targets)} target prims (TSF_85 + fingers)",
         f"json: {OUT_JSON}", "-" * 70]
    for t in targets:
        L.append(f"{t['path']}")
        L.append(f"   type={t['type']}  children={t['n_children']}")
        if t["applied_schemas"]:
            L.append(f"   schemas: {', '.join(t['applied_schemas'])}")
        if t["attrs"]:
            for k, v in t["attrs"].items():
                L.append(f"      {k} = {v}")
        else:
            L.append("      (no mass/inertia/deformable attrs authored here)")
        L.append("")
    L.append("=" * 70)
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
