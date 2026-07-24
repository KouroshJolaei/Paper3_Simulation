"""
diag11_sensor_geometry.py — measure the sensor pad geometry STATICALLY,
straight out of the USD file. No SimulationApp, no physics, no contact.

WHY: PAD_CENTER_ABOVE_CASE_M (Case prim origin -> taxel-array centre) has
only ever been INFERRED from contact patterns. Those inferences are
confounded once the pad is partially backed (the inner finger tips under
an off-centre load, so pressure stops mapping to contact area). The
geometry itself is authored in the scene and can just be read.

WHAT IT DOES: opens scene_cylinder.usd, finds every TSF_85 'Case' prim,
and for the Case and each of its descendant meshes reports:
  - the prim path and type
  - the mesh point-cloud bounding box IN THE CASE'S LOCAL FRAME
  - the offset from the Case ORIGIN to the mesh CENTRE, per axis

The axis whose extent is ~37-41 mm is the pad's long (7-row) axis; the
offset along that axis from Case origin to the mesh centre is the number
we want. Compare against the current 0.0329 m.

Run with Isaac's python (it has pxr), but it does NOT launch Isaac:
  ~/isaacsim/python.sh ~/Paper3_Simulation/diag11_sensor_geometry.py
"""
import os
import numpy as np
from pxr import Usd, UsdGeom, Gf

USD_PATH = os.path.expanduser(
    "~/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd")
CURRENT_CONST_MM = 32.9        # PAD_CENTER_ABOVE_CASE_M currently in use


def mat_to_np(m):
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=float)


def xform_points(M, pts):
    """Apply a 4x4 row-vector USD matrix to (N,3) points."""
    P = np.hstack([pts, np.ones((len(pts), 1))])
    return (P @ M)[:, :3]


def main():
    if not os.path.exists(USD_PATH):
        print(f"scene not found: {USD_PATH}")
        return
    stage = Usd.Stage.Open(USD_PATH)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    cases = [p for p in stage.Traverse()
             if p.GetName() == "Case" and "TSF_85" in str(p.GetPath())]
    if not cases:
        print("no TSF_85 'Case' prim found — printing all TSF prims instead:")
        for p in stage.Traverse():
            if "TSF_85" in str(p.GetPath()):
                print(f"  {p.GetTypeName():18} {p.GetPath()}")
        return

    for case in cases:
        print("=" * 70)
        print(f"CASE prim: {case.GetPath()}")
        M_case = mat_to_np(cache.GetLocalToWorldTransform(case))
        M_case_inv = np.linalg.inv(M_case)

        found_any = False
        for prim in Usd.PrimRange(case):
            pb = UsdGeom.PointBased(prim)
            if not pb:
                continue
            pts_attr = pb.GetPointsAttr()
            if not pts_attr or pts_attr.Get() is None:
                continue
            pts = np.array([[p[0], p[1], p[2]] for p in pts_attr.Get()],
                           dtype=float)
            if len(pts) == 0:
                continue
            found_any = True
            M_prim = mat_to_np(cache.GetLocalToWorldTransform(prim))
            world = xform_points(M_prim, pts)
            local = xform_points(M_case_inv,
                                 np.hstack([world]))   # -> Case frame
            lo, hi = local.min(axis=0), local.max(axis=0)
            ctr = 0.5 * (lo + hi)
            ext = hi - lo
            print(f"\n  mesh: {prim.GetPath()}")
            print(f"        type={prim.GetTypeName()}  n_points={len(pts)}")
            print(f"        extent  (mm): "
                  f"X={1000*ext[0]:7.2f}  Y={1000*ext[1]:7.2f}  Z={1000*ext[2]:7.2f}")
            print(f"        centre offset from CASE ORIGIN (mm): "
                  f"X={1000*ctr[0]:+7.2f}  Y={1000*ctr[1]:+7.2f}  Z={1000*ctr[2]:+7.2f}")
            long_axis = int(np.argmax(ext))
            print(f"        longest axis = {'XYZ'[long_axis]} "
                  f"({1000*ext[long_axis]:.2f} mm), "
                  f"offset along it = {1000*ctr[long_axis]:+.2f} mm  "
                  f"(current constant {CURRENT_CONST_MM:+.1f})")
        if not found_any:
            print("  (no point-based geometry under this Case — listing children)")
            for prim in Usd.PrimRange(case):
                print(f"    {prim.GetTypeName():18} {prim.GetPath()}")


if __name__ == "__main__":
    main()
