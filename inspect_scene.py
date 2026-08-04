from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd
s = Usd.Stage.Open("/home/kourosh/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd")
for p in s.Traverse():
    path = str(p.GetPath())
    if "Object_0" in path:
        print(p.GetTypeName(), path)

app.close()