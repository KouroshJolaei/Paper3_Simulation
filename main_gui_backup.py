"""
main_gui.py — Paper 3 data-collection cockpit (STAGE A).

Pure PyCharm / Tkinter. NO Isaac needed. Lets you:
  - enter the OBJECT pose (mm) + orientation (cylinder for now)
  - enter the PAD initial pose (mm); rotation greyed out for now
  - enter a 2D GRID: n steps in X, n steps in Y, one step size (mm)
    (X,Y are along the cylinder surface, parallel to the pad face)
  - SEE a live TOP-DOWN + FRONT preview:
      TOP-DOWN: cylinder circle in middle, TWO pads facing each other along X
                (one -X side, one +X side), symmetric, pressing on the rim.
      FRONT:    pad face(s) on the cylinder surface, with the full 2D grid.

All distances shown in mm. Stage B will add the "write config + run Isaac"
bridge; Stage C the heatmap + pose-history read-back buttons.

Run in PyCharm:  python3 main_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os, json, subprocess, threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- real hardware sizes (mm) ----
PAD_W = 22.0    # pad short side (4 taxels)
PAD_H = 37.0    # pad long side  (7 taxels)
CYL_D = 26.0    # cylinder diameter
CYL_L = 140.0   # cylinder length
GRIP_OPEN = 12.0  # half-gap of each pad from the cylinder rim before closing (mm, visual)
ROBOT_BASE_MM = np.array([20.93, -337.5, 992.75])  # robot base_link world (mm)

# ---- paths for the config-save + Isaac-launch bridge (Stage B) ----
PROJECT   = os.path.expanduser("~/Paper3_Simulation")
CONFIG_JSON = os.path.join(PROJECT, "Data", "gui_config.json")
CALIB_CONFIG_JSON = os.path.join(PROJECT, "Data", "gui_calib_config.json")
ISAAC_PY  = os.path.expanduser("~/isaacsim/python.sh")
COLLECT_PY = os.path.join(PROJECT, "sim", "collect_from_config.py")
EXAMPLES_DIR = os.path.expanduser("~/Paper3_Simulation/TSF-85/examples")


def grid_2d(nx, ny, step_mm):
    """Return list of (dx, dy) offsets in mm for the grasp grid.

    SIGNED / ANCHORED convention:
      the grid is ANCHORED at the entered pad offset — offset (0,0) is
      always the FIRST point (pt00). |n| = number of points along that
      axis; the SIGN picks the direction the grid extends:
        nx = +3 -> 0, +step, +2*step      nx = -3 -> 0, -step, -2*step
      (the old centred grid = an anchored grid whose base sits at one edge)
    """
    def axis(n):
        n = int(n) if int(n) != 0 else 1
        sgn = 1.0 if n > 0 else -1.0
        return sgn * np.arange(abs(n)) * step_mm
    xs = axis(nx)
    ys = axis(ny)
    pts = []
    for gy in ys:
        for gx in xs:
            pts.append((gx, gy))
    return pts


class CockpitGUI:
    def __init__(self, root):
        self.root = root
        root.title("Paper 3 — Collection Cockpit (Stage A)")

        # defaults in mm (from our proven scene)
        self.vars = {
            "obj_x": tk.StringVar(value="-268.06"),
            "obj_y": tk.StringVar(value="199.0"),
            "obj_z": tk.StringVar(value="1052.2"),
            "obj_tilt_deg": tk.StringVar(value="0.0"),      # 0 = standing
            "obj_tilt_axis": tk.StringVar(value="X"),       # tilt about this axis
            # pad pose = offset from OBJECT CENTER (mm). X is fixed (centered grasp).
            "pad_dy": tk.StringVar(value="0.0"),
            "pad_dz": tk.StringVar(value="0.0"),
            "grid_nx":  tk.StringVar(value="2"),
            "grid_ny":  tk.StringVar(value="3"),
            "grid_step": tk.StringVar(value="8.0"),   # mm
            "headless": tk.BooleanVar(value=False),   # False = show Isaac window
            "calib_headless": tk.BooleanVar(value=False),  # Calibrate tab headless toggle
            "calib_dz": tk.StringVar(value="0.0"),  # Calibrate pad Z offset (Y stays centered)
        }

        # ---- Notebook: tab 1 = collection cockpit, tab 2 = stitching ----
        self.nb = ttk.Notebook(root)
        self.nb.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.tab_collect = ttk.Frame(self.nb)
        self.tab_stitch = ttk.Frame(self.nb)
        self.tab_calib = ttk.Frame(self.nb)
        self.nb.add(self.tab_collect, text="Collection")
        self.nb.add(self.tab_calib, text="Calibrate")
        self.nb.add(self.tab_stitch, text="Stitching (Block 2)")

        self._build_inputs()
        self._build_preview()
        self._build_stitch_tab()
        self._build_calib_tab()
        self.refresh()

    def _build_inputs(self):
        frm = ttk.Frame(self.tab_collect, padding=10)
        frm.grid(row=0, column=0, sticky="ns")
        r = 0

        ttk.Label(frm, text="OBJECT pose (world, mm)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        for key, lab in [("obj_x", "x"), ("obj_y", "y"), ("obj_z", "z")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="tilt (deg)").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["obj_tilt_deg"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="tilt axis").grid(row=r, column=0, sticky="e")
        ob = ttk.Combobox(frm, textvariable=self.vars["obj_tilt_axis"], width=12,
                          values=["X", "Y", "Z"])
        ob.grid(row=r, column=1, sticky="w"); ob.bind("<<ComboboxSelected>>", lambda ev: self.refresh()); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

        ttk.Label(frm, text="PAD offset from object center (mm)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, text="x").grid(row=r, column=0, sticky="e")
        xe = ttk.Entry(frm, width=12, state="disabled")
        xe.grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(frm, text="(x fixed: centered grasp)", foreground="#888").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        for key, lab in [("pad_dy", "y"), ("pad_dz", "z")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="rotation (deg)").grid(row=r, column=0, sticky="e")
        ttk.Entry(frm, width=12, state="disabled").grid(row=r, column=1, sticky="w"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1

        ttk.Label(frm, text="GRID (anchored at pad offset; sign = direction)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, text="n steps X").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["grid_nx"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="n steps Y").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["grid_ny"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="step (mm)").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["grid_step"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1

        ttk.Button(frm, text="Update Preview", command=self.refresh).grid(
            row=r, column=0, columnspan=2, pady=(12, 4), sticky="ew"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Checkbutton(frm, text="Run headless (no Isaac window)",
                        variable=self.vars["headless"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Save Config", command=self.save_config).grid(
            row=r, column=0, columnspan=2, pady=4, sticky="ew"); r += 1
        ttk.Button(frm, text="Save + Show Run Command", command=self.save_and_show_cmd).grid(
            row=r, column=0, columnspan=2, pady=4, sticky="ew"); r += 1

        # ---- Save / Load a named EXPERIMENT (full recipe: pose, tilt, grid) ----
        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(frm, text="EXPERIMENT recipe:", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Save Experiment As…", command=self.save_experiment).grid(
            row=r, column=0, columnspan=2, pady=2, sticky="ew"); r += 1
        ttk.Button(frm, text="Load Experiment…", command=self.load_experiment).grid(
            row=r, column=0, columnspan=2, pady=2, sticky="ew"); r += 1

        self.status = ttk.Label(frm, text="", foreground="#06a", wraplength=190)
        self.status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 0)); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Label(frm, text="AFTER the run:", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        # plot source: newest run (default) OR a saved folder you pick
        ttk.Button(frm, text="Plot from folder…", command=self.choose_plot_folder).grid(
            row=r, column=0, pady=2, sticky="ew")
        ttk.Button(frm, text="Use newest", command=self.use_newest_run).grid(
            row=r, column=1, pady=2, sticky="ew"); r += 1
        self.plot_src_lbl = ttk.Label(frm, text="plot source: newest run (auto)",
                                      foreground="#666", wraplength=190)
        self.plot_src_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Show Heatmaps (s1+s2)", command=self.show_heatmaps).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Show Pose History", command=self.show_pose_history).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Make Verification Plots", command=self.make_verifications).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Show Temporal Snapshots (4-step)", command=self.show_temporal).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1
        ttk.Button(frm, text="Check Pad Truth (safety)", command=self.show_pad_truth).grid(
            row=r, column=0, columnspan=2, pady=3, sticky="ew"); r += 1

        self.info = ttk.Label(frm, text="", foreground="#0a6", wraplength=190)
        self.info.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

    def _build_preview(self):
        self.fig = Figure(figsize=(11.0, 4.4), dpi=100)
        self.ax_top   = self.fig.add_subplot(1, 3, 1)
        self.ax_front = self.fig.add_subplot(1, 3, 2)
        self.ax_3d    = self.fig.add_subplot(1, 3, 3, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_collect)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.tab_collect.columnconfigure(1, weight=1)
        self.tab_collect.rowconfigure(0, weight=1)

    def _read(self):
        def fnum(key, default=0.0):
            try:
                return float(str(self.vars[key].get()).strip())
            except Exception:
                return default
        def inum(key, default=1):
            # signed integer: |n| = number of points, sign = grid direction
            try:
                v = int(float(str(self.vars[key].get()).strip()))
                return v if v != 0 else 1
            except Exception:
                return default
        try:
            obj = np.array([fnum("obj_x"), fnum("obj_y"), fnum("obj_z")])
            pad = obj + np.array([0.0, fnum("pad_dy"), fnum("pad_dz")])
            return {
                "obj": obj,
                "tilt_deg": fnum("obj_tilt_deg"),
                "tilt_axis": self.vars["obj_tilt_axis"].get(),
                "pad": pad,
                "pad_dy": fnum("pad_dy"),
                "pad_dz": fnum("pad_dz"),
                "nx": inum("grid_nx"),
                "ny": inum("grid_ny"),
                "step": fnum("grid_step", 1.0),
            }
        except Exception:
            return None

    def refresh(self):
        cfg = self._read()
        if cfg is None:
            self.info.config(text="check numeric inputs", foreground="red"); return

        obj, pad = cfg["obj"], cfg["pad"]
        offs = grid_2d(cfg["nx"], cfg["ny"], cfg["step"])  # (dx,dy) mm

        # ---------- TOP-DOWN (X-Y): two pads squeezing the cylinder along X ----------
        ax = self.ax_top; ax.clear()
        ax.set_title("TOP-DOWN (X-Y)\ntwo pads squeeze along X")
        ax.set_xlabel("world X (mm)"); ax.set_ylabel("world Y (mm)")
        # cylinder circle at object centre
        th = np.linspace(0, 2*np.pi, 80)
        ax.plot(obj[0] + (CYL_D/2)*np.cos(th), obj[1] + (CYL_D/2)*np.sin(th),
                color="steelblue", linewidth=2, label="cylinder")
        ax.scatter(obj[0], obj[1], color="steelblue", s=15)
        # two pads on -X and +X sides, tangent to the rim (+ small opening gap).
        # In top-down the pad's SHORT side (width, across Y) is visible as a line.
        rim = CYL_D/2 + GRIP_OPEN
        for gx, gy in offs:
            cy = pad[1] + gy    # the grid's Y offset shifts pads along Y
            # -X pad
            ax.plot([obj[0]-rim, obj[0]-rim], [cy-PAD_W/2, cy+PAD_W/2],
                    color="crimson", linewidth=3)
            # +X pad
            ax.plot([obj[0]+rim, obj[0]+rim], [cy-PAD_W/2, cy+PAD_W/2],
                    color="darkorange", linewidth=3)
        ax.plot([], [], color="crimson", linewidth=3, label="pad -X (s1)")
        ax.plot([], [], color="darkorange", linewidth=3, label="pad +X (s2)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

        # ---------- FRONT (Y-Z): pad face on the cylinder surface, 2D grid ----------
        ax = self.ax_front; ax.clear()
        ax.set_title("FRONT (Y-Z)\npad face on cylinder, grid")
        ax.set_xlabel("world Y (mm)"); ax.set_ylabel("world Z (mm)")
        # Cylinder drawn as a rectangle (length x diameter) tilted by tilt_deg.
        # In the FRONT (Y-Z) view we show tilt about X as an in-plane rotation
        # (standing=0 -> vertical bar; 90 -> horizontal). Tilt about Y/Z tips it
        # out of / within this plane; we approximate by rotating in-plane for X,
        # and note the axis in the title.
        tilt = np.radians(cfg["tilt_deg"])
        # rectangle corners centred at object (in Y-Z), long axis = Z when standing
        L, D = CYL_L, CYL_D
        corners = np.array([[-D/2, -L/2], [ D/2, -L/2], [ D/2, L/2], [-D/2, L/2], [-D/2, -L/2]])
        c, s = np.cos(tilt), np.sin(tilt)
        Rt = np.array([[c, -s], [s, c]])
        rot = corners @ Rt.T
        ax.plot(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue",
                linewidth=2, label=f"cylinder (tilt {cfg['tilt_deg']:.0f}° about {cfg['tilt_axis']})")
        ax.fill(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue", alpha=0.35)
        # grid of pad footprints (pad short side across Y, long side up Z)
        for i, (gx, gy) in enumerate(offs):
            py = pad[1] + gy - PAD_W/2
            pz = pad[2] + gx - PAD_H/2   # use X-grid as the up/down (Z) sweep on the face
            ax.add_patch(mpatches.Rectangle((py, pz), PAD_W, PAD_H, fill=False,
                                            edgecolor="crimson", linewidth=1.5,
                                            label="pad" if i == 0 else None))
            ax.scatter(pad[1]+gy, pad[2]+gx, color="crimson", s=10)
        # visit path: the exact order the collector executes (pt00 -> last)
        if len(offs) > 1:
            px = [pad[1] + gy for gx, gy in offs]
            pz = [pad[2] + gx for gx, gy in offs]
            ax.plot(px, pz, color="dimgray", linestyle="--", linewidth=1.2,
                    alpha=0.9, zorder=4, label="visit path")
            ax.scatter(px[0], pz[0], color="green", s=45, zorder=5,
                       label="pt00 (start = base)")
            ax.scatter(px[-1], pz[-1], color="red", marker="X", s=45,
                       zorder=5, label="last pt")
            if len(offs) <= 24:
                for i, (x_, z_) in enumerate(zip(px, pz)):
                    ax.annotate(str(i), (x_, z_), textcoords="offset points",
                                xytext=(3, 3), fontsize=6, color="dimgray")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6, loc="upper right"); ax.grid(alpha=0.3)

        # ---------- 3D scene: cylinder + two pads + base + grid (real size) ----------
        ax = self.ax_3d; ax.clear()
        ax.set_title("3D scene (real size)")
        # cylinder as a 3D surface, tilted by tilt_deg about the chosen axis
        zc = np.linspace(-CYL_L/2, CYL_L/2, 20)
        th = np.linspace(0, 2*np.pi, 24)
        TH, ZC = np.meshgrid(th, zc)
        XC = (CYL_D/2)*np.cos(TH)
        YC = (CYL_D/2)*np.sin(TH)
        pts = np.stack([XC.ravel(), YC.ravel(), ZC.ravel()], axis=1)
        tilt = np.radians(cfg["tilt_deg"]); axis = cfg["tilt_axis"]
        c, s = np.cos(tilt), np.sin(tilt)
        if axis == "X":
            R3 = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == "Y":
            R3 = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            R3 = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        pts = pts @ R3.T + obj
        Xc = pts[:, 0].reshape(XC.shape); Yc = pts[:, 1].reshape(YC.shape); Zc = pts[:, 2].reshape(ZC.shape)
        ax.plot_surface(Xc, Yc, Zc, color="steelblue", alpha=0.4, linewidth=0)

        # two pads on -X and +X of the object, at each grid point (real rectangles)
        rim = CYL_D/2 + GRIP_OPEN
        def pad_rect(center, normal_x_sign):
            # pad face spans Y (width) and Z (height); positioned at +/-X from center
            cx = obj[0] + normal_x_sign*rim
            cy, cz = center[1], center[2]
            ys = np.array([cy-PAD_W/2, cy+PAD_W/2, cy+PAD_W/2, cy-PAD_W/2, cy-PAD_W/2])
            zs = np.array([cz-PAD_H/2, cz-PAD_H/2, cz+PAD_H/2, cz+PAD_H/2, cz-PAD_H/2])
            xs = np.full_like(ys, cx)
            return xs, ys, zs
        for gx, gy in offs:
            cen = np.array([obj[0], pad[1]+gy, pad[2]+gx])
            xs, ys, zs = pad_rect(cen, -1); ax.plot(xs, ys, zs, color="crimson", linewidth=1.5)
            xs, ys, zs = pad_rect(cen, +1); ax.plot(xs, ys, zs, color="darkorange", linewidth=1.5)

        # robot base marker
        ax.scatter(*ROBOT_BASE_MM, color="black", s=40, marker="s", label="robot base")
        ax.plot([], [], color="crimson", label="pad -X (s1)")
        ax.plot([], [], color="darkorange", label="pad +X (s2)")
        ax.plot([], [], color="steelblue", label="cylinder")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.legend(fontsize=6, loc="upper left")
        # equalize aspect roughly around the object
        rng = 120
        ax.set_xlim(obj[0]-rng, obj[0]+rng)
        ax.set_ylim(obj[1]-rng, obj[1]+rng)
        ax.set_zlim(obj[2]-rng, obj[2]+rng)

        n = abs(cfg["nx"] * cfg["ny"])
        self.info.config(
            text=f"pad offset from center: y={cfg['pad_dy']:.1f}, z={cfg['pad_dz']:.1f} mm\n"
                 f"grid: {cfg['nx']}x{cfg['ny']} = {n} grasp poses (anchored: pt00 = base,\n"
                 f"sign = direction), step {cfg['step']:.1f} mm "
                 f"[upd #{getattr(self, '_refresh_count', 0)+1}]",
            foreground="#0a6")

        # actually redraw the canvas (this was missing -> preview never updated)
        self.fig.tight_layout()
        self.canvas.draw()
        try:
            self.canvas.flush_events()
        except Exception:
            pass
        self._refresh_count = getattr(self, "_refresh_count", 0) + 1


    # ---------- Stage B: build config, save, launch Isaac ----------
    def build_config(self):
        """Assemble the full config dict: object pose+tilt, pad offset, and the
        list of grasp points (each = pad Y-Z offset from object center, mm).
        All poses in mm; the sim converts to metres."""
        cfg = self._read()
        if cfg is None:
            return None
        offs = grid_2d(cfg["nx"], cfg["ny"], cfg["step"])   # (dx,dy) mm on the face
        # each grasp point = pad offset from object centre (y,z), plus the base pad offset
        points = []
        for k, (gx, gy) in enumerate(offs):
            points.append({
                "index": k,
                "pad_offset_y_mm": cfg["pad_dy"] + gy,   # Y across the face
                "pad_offset_z_mm": cfg["pad_dz"] + gx,   # Z up/down the face
            })
        return {
            "object": {
                "center_world_mm": cfg["obj"].tolist(),
                "tilt_deg": cfg["tilt_deg"],
                "tilt_axis": cfg["tilt_axis"],
                "shape": "cylinder",
                "diameter_mm": CYL_D,
                "length_mm": CYL_L,
            },
            "pad": {
                "base_offset_y_mm": cfg["pad_dy"],
                "base_offset_z_mm": cfg["pad_dz"],
                "x_fixed_centered": True,
            },
            "grid": {
                "nx": cfg["nx"], "ny": cfg["ny"], "step_mm": cfg["step"],
                "n_points": len(points),
            },
            "points": points,
        }

    def save_config(self):
        cfg = self.build_config()
        if cfg is None:
            messagebox.showerror("Config", "Check numeric inputs."); return None
        os.makedirs(os.path.dirname(CONFIG_JSON), exist_ok=True)
        with open(CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        self.status.config(
            text=f"saved config: {cfg['grid']['n_points']} points\n{CONFIG_JSON}",
            foreground="#0a6")
        return cfg

    def save_experiment(self):
        """Save the full experiment recipe to a named JSON so it can be
        reloaded later and reproduced exactly. Stores both the GUI field
        values (for exact reload) and the built config (for reference)."""
        from tkinter import filedialog
        cfg = self.build_config()
        if cfg is None:
            messagebox.showerror("Experiment", "Check numeric inputs."); return
        fields = {k: self.vars[k].get() for k in self.vars
                  if k != "headless"}
        recipe = {"gui_fields": fields, "config": cfg,
                  "note": "Paper3 experiment recipe — reload in the cockpit"}
        exp_dir = os.path.join(PROJECT, "Data", "experiments")
        os.makedirs(exp_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=exp_dir, defaultextension=".json",
            filetypes=[("Experiment JSON", "*.json")],
            title="Save experiment recipe as")
        if not path:
            return
        with open(path, "w") as f:
            json.dump(recipe, f, indent=2)
        self.status.config(text="experiment saved:\n" + os.path.basename(path),
                           foreground="#0a6")

    def load_experiment(self):
        """Load a saved experiment recipe back into the GUI fields."""
        from tkinter import filedialog
        exp_dir = os.path.join(PROJECT, "Data", "experiments")
        path = filedialog.askopenfilename(
            initialdir=exp_dir if os.path.isdir(exp_dir) else PROJECT,
            filetypes=[("Experiment JSON", "*.json"), ("All", "*.*")],
            title="Load experiment recipe")
        if not path:
            return
        try:
            with open(path) as f:
                recipe = json.load(f)
            fields = recipe.get("gui_fields", {})
            for k, v in fields.items():
                if k in self.vars:
                    self.vars[k].set(v)
            self.refresh()
            self.status.config(text="experiment loaded:\n" + os.path.basename(path),
                               foreground="#0a6")
        except Exception as e:
            messagebox.showerror("Experiment", "Could not load:\n%s" % e)

    def save_and_show_cmd(self):
        cfg = self.save_config()
        if cfg is None:
            return
        # the exact, proven terminal command (this is what worked for you)
        headless = "1" if self.vars["headless"].get() else "0"
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n'
            f'GRASP_BASENAME="gui" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            f"{ISAAC_PY} {COLLECT_PY} \\\n"
            f"  --config {CONFIG_JSON}"
        )
        # pop a window with the command, selectable + a Copy button
        win = tk.Toplevel(self.root)
        win.title("Run command — copy into a terminal")
        tk.Label(win, text="Config saved. Copy this into a terminal and run it:\n"
                           "(watch the Isaac window; come back and press Show Heatmaps when done)",
                 justify="left").pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=80, height=7, wrap="none")
        txt.insert("1.0", cmd); txt.configure(state="normal")
        txt.pack(padx=10, pady=4)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
            self.status.config(text="command copied to clipboard.", foreground="#0a6")
        tk.Button(win, text="Copy to clipboard", command=_copy).pack(pady=(4, 10))
        self.status.config(text=f"config saved: {cfg['grid']['n_points']} points.\n"
                                f"copy the command to run.", foreground="#0a6")

    # ================= Calibrate tab =================
    def _build_calib_tab(self):
        """Calibrate the pad Z-offset for the CURRENT object diameter. Closes the
        gripper on the object (centered) once, measures TOOL_OFFSET_Z, stores it.
        Same copy-run-command + headless-toggle UX as the Collection tab."""
        frm = ttk.Frame(self.tab_calib, padding=10)
        frm.grid(row=0, column=0, sticky="ns")
        self.tab_calib.columnconfigure(1, weight=1)
        self.tab_calib.rowconfigure(0, weight=1)

        r = 0
        ttk.Label(frm, text="CALIBRATE pad Z-offset",
                  font=("", 11, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, wraplength=250, foreground="#444", justify="left",
                  text=("Closes the gripper on the CURRENT object (centered) and "
                        "measures the exact TOOL_OFFSET_Z for its diameter, then "
                        "stores it. Collection refuses an object until it is "
                        "calibrated.")).grid(row=r, column=0, columnspan=2, sticky="w",
                                             pady=(2, 8)); r += 1

        self.calib_obj_lbl = ttk.Label(frm, text="", foreground="#06a", wraplength=250)
        self.calib_obj_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="PAD placement (mm)",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(frm, text="Z offset (up/down)").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["calib_dz"], width=10)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda _e: self.refresh_calib()); r += 1
        ttk.Label(frm, text="Y locked to center (needed for a valid diameter grip)",
                  foreground="#888", wraplength=250).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Checkbutton(frm, text="Run headless (no Isaac window)",
                        variable=self.vars["calib_headless"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Update Preview", command=self.refresh_calib).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(6, 2)); r += 1
        ttk.Button(frm, text="Save + Show Calibrate Command",
                   command=self.save_and_show_calib_cmd).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
        self.calib_status = ttk.Label(frm, text="", foreground="#06a", wraplength=250)
        self.calib_status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0)); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="AFTER the calibrate run:",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Load calibration result",
                   command=self.show_calibration).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
        self.calib_result = tk.Text(frm, width=34, height=10, wrap="word")
        self.calib_result.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0)); r += 1

        # preview (FRONT Y-Z only) — object + centered pad
        self.fig_calib = Figure(figsize=(4.6, 5.2), dpi=100)
        self.ax_calib = self.fig_calib.add_subplot(1, 1, 1)
        self.canvas_calib = FigureCanvasTkAgg(self.fig_calib, master=self.tab_calib)
        self.canvas_calib.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.refresh_calib()

    def refresh_calib(self):
        cfg = self._read()
        ax = self.ax_calib; ax.clear()
        if cfg is None:
            ax.set_title("check numeric inputs"); self.canvas_calib.draw(); return
        obj = cfg["obj"]
        try:
            cdz = float(str(self.vars["calib_dz"].get()).strip())
        except Exception:
            cdz = 0.0
        pad_z = obj[2] + cdz
        # is the pad still on the cylinder body (within +/- length/2)?
        on_body = abs(cdz) + PAD_H/2 <= CYL_L/2 + 1e-6
        flag = "" if on_body else "   [WARNING: pad off the object end]"
        self.calib_obj_lbl.config(
            text=(f"object center (mm): {obj.round(1).tolist()}\n"
                  f"diameter: {CYL_D} mm    (Y centered)\n"
                  f"pad Z: {pad_z:.1f} mm  (offset {cdz:+.1f}){flag}"))
        ax.set_title("CALIBRATE — pad on object (FRONT Y-Z)")
        ax.set_xlabel("world Y (mm)"); ax.set_ylabel("world Z (mm)")
        tilt = np.radians(cfg["tilt_deg"])
        L, D = CYL_L, CYL_D
        corners = np.array([[-D/2, -L/2], [D/2, -L/2], [D/2, L/2], [-D/2, L/2], [-D/2, -L/2]])
        c, s = np.cos(tilt), np.sin(tilt); Rt = np.array([[c, -s], [s, c]]); rot = corners @ Rt.T
        ax.plot(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue",
                linewidth=2, label="cylinder")
        ax.fill(obj[1] + rot[:, 0], obj[2] + rot[:, 1], color="steelblue", alpha=0.35)
        pad_color = "crimson" if on_body else "red"
        ax.add_patch(mpatches.Rectangle((obj[1]-PAD_W/2, pad_z-PAD_H/2), PAD_W, PAD_H,
                     fill=False, edgecolor=pad_color, linewidth=1.8,
                     linestyle="-" if on_body else "--", label="pad (Y centered)"))
        ax.scatter(obj[1], pad_z, color=pad_color, s=12)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        self.canvas_calib.draw()

    def build_calib_config(self):
        """Config for a calibrate grasp: ONE point, Y centered, Z = calib_dz.
        Y is intentionally locked to center so the pads meet the full diameter."""
        cfg = self._read()
        if cfg is None:
            return None
        try:
            cdz = float(str(self.vars["calib_dz"].get()).strip())
        except Exception:
            cdz = 0.0
        return {
            "object": {
                "center_world_mm": cfg["obj"].tolist(),
                "tilt_deg": cfg["tilt_deg"],
                "tilt_axis": cfg["tilt_axis"],
                "shape": "cylinder",
                "diameter_mm": CYL_D,
                "length_mm": CYL_L,
            },
            "pad": {"base_offset_y_mm": 0.0, "base_offset_z_mm": cdz,
                    "x_fixed_centered": True},
            "grid": {"nx": 1, "ny": 1, "step_mm": 8.0, "n_points": 1},
            "points": [{"index": 0, "pad_offset_y_mm": 0.0, "pad_offset_z_mm": cdz}],
            "calibrate": True,
        }

    def save_and_show_calib_cmd(self):
        cfg = self.build_calib_config()
        if cfg is None:
            messagebox.showerror("Calibrate", "Check numeric inputs."); return
        os.makedirs(os.path.dirname(CALIB_CONFIG_JSON), exist_ok=True)
        with open(CALIB_CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        headless = "1" if self.vars["calib_headless"].get() else "0"
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n'
            f'GRASP_BASENAME="calib" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            f'GRASP_CALIBRATE="1" \\\n'
            f"{ISAAC_PY} {COLLECT_PY} \\\n"
            f"  --config {CALIB_CONFIG_JSON}"
        )
        win = tk.Toplevel(self.root)
        win.title("Calibrate command — copy into a terminal")
        tk.Label(win, justify="left",
                 text=("Config saved. Copy this into a terminal and run it.\n"
                       "It closes on the object ONCE (Y centered, at your Z) and stores "
                       "the offset for this diameter.\nThen press 'Load calibration result'."
                       )).pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=82, height=8, wrap="none")
        txt.insert("1.0", cmd); txt.pack(padx=10, pady=4)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
            self.calib_status.config(text="calibrate command copied.", foreground="#0a6")
        tk.Button(win, text="Copy to clipboard", command=_copy).pack(pady=(4, 10))
        self.calib_status.config(text="calib config saved. copy the command to run.",
                                 foreground="#0a6")

    def show_calibration(self):
        cal_path = os.path.join(PROJECT, "Data", "pad_offset_calibration.json")
        self.calib_result.delete("1.0", "end")
        if not os.path.exists(cal_path):
            self.calib_result.insert("end", f"No calibration file yet:\n{cal_path}\n\n"
                                            f"Run the calibrate command first.")
            return
        try:
            with open(cal_path) as f:
                cal = json.load(f)
        except Exception as e:
            self.calib_result.insert("end", f"Could not read:\n{e}"); return
        if not cal:
            self.calib_result.insert("end", "Calibration file is empty."); return
        lines = ["Calibrated offsets by diameter:", ""]
        for k, v in sorted(cal.items(), key=lambda kv: float(kv[0])):
            lines.append(f"  \u00d8{v.get('diameter_mm', k)} mm  ->  "
                         f"TOOL_OFFSET_Z = {v.get('TOOL_OFFSET_Z')}")
        key = f"{CYL_D:.1f}"
        if key in cal:
            lines.append(f"\nCurrent object (\u00d8{CYL_D}): "
                         f"{cal[key]['TOOL_OFFSET_Z']}  [calibrated]")
        else:
            lines.append(f"\nCurrent object (\u00d8{CYL_D}): NOT calibrated yet")
        self.calib_result.insert("end", "\n".join(lines))
        self.calib_status.config(text="calibration loaded.", foreground="#0a6")

    # ---------- Stage C: read-back heatmaps + pose history ----------
    def _run_dir(self):
        # If the user picked a folder to re-plot from, use it. Otherwise the
        # collector writes to gui_run/run_<timestamp>/ — pick the newest.
        forced = getattr(self, "_forced_run_dir", None)
        if forced:
            return forced
        import glob
        base = os.path.join(PROJECT, "Data", "gui_run")
        runs = sorted(glob.glob(os.path.join(base, "run_*")))
        if runs:
            return runs[-1]          # newest run
        return base                   # fallback (old flat layout)

    def choose_plot_folder(self):
        """Pick a saved run folder to (re)generate plots from. All four plot
        buttons then read from — and save back into — this folder."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=os.path.join(PROJECT, "Data", "gui_run"),
            title="Pick a run folder to plot from")
        if d:
            self._forced_run_dir = d
            self.plot_src_lbl.config(text="plot source: " + os.path.basename(d),
                                     foreground="#06a")

    def use_newest_run(self):
        """Clear the folder override — plot from the newest run again."""
        self._forced_run_dir = None
        self.plot_src_lbl.config(text="plot source: newest run (auto)",
                                 foreground="#666")

    def show_heatmaps(self):
        """Hold-average heatmap per grasp, s1 | s2 side by side, one window
        per grid point. PNGs are saved to <run>/Heatmaps/ by viz/heatmaps.py."""
        import glob, traceback, importlib.util
        run = self._run_dir()
        if not glob.glob(os.path.join(run, "*_s1_tactile_maps.csv")):
            messagebox.showinfo("Heatmaps",
                f"No tactile files found in:\n{run}\n\nRun the simulation first.")
            return
        try:
            hpath = None
            for cand in (os.path.join(PROJECT, "viz", "heatmaps.py"),
                         os.path.join(PROJECT, "heatmaps.py"),
                         os.path.join(PROJECT, "sim", "heatmaps.py")):
                if os.path.exists(cand):
                    hpath = cand; break
            if hpath is None:
                messagebox.showerror("Heatmaps",
                    "heatmaps.py not found (expected in viz/).")
                return
            spec = importlib.util.spec_from_file_location("heatmaps", hpath)
            hm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hm)

            made = hm.plot_run(run)          # saves Heatmaps/heatmap_<tag>.png
            if not made:
                messagebox.showinfo("Heatmaps", "No heatmaps produced.")
                return

            # one window per grid point (the proven verification-style display)
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            for png in made:
                fig = plt.figure(figsize=(8.5, 4.5))
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(mpimg.imread(png)); ax.axis("off")
                ax.set_title(os.path.basename(png))
                fig.tight_layout()
            plt.show()
            self.status.config(
                text=f"heatmaps: {len(made)} grasps (s1|s2)\nsaved to Heatmaps/",
                foreground="#0a6")
        except Exception:
            messagebox.showerror("Heatmaps",
                "Heatmaps failed:\n\n" + traceback.format_exc())

    def show_pose_history(self):
        import json as _json
        run = self._run_dir()
        ph = os.path.join(run, "pose_history.json")
        if not os.path.exists(ph):
            messagebox.showinfo("Pose History",
                f"No pose_history.json in:\n{run}\n\nRun the simulation first.")
            return
        with open(ph) as f:
            data = _json.load(f)
        lines = ["Real pad/EE pose reached at each grasp (world, m):\n"]
        for p in data.get("points", []):
            ee = p["ee_world_m"]
            lines.append(f"  {p['tag']}:  x={ee[0]:+.4f}  y={ee[1]:+.4f}  z={ee[2]:+.4f}")
        win = tk.Toplevel(self.root)
        win.title("Pose History")
        txt = tk.Text(win, width=60, height=max(6, len(lines)+2), wrap="none")
        txt.insert("1.0", "\n".join(lines))
        txt.pack(padx=10, pady=10)
        self.status.config(text=f"showing {len(data.get('points', []))} poses.",
                           foreground="#0a6")

    def make_verifications(self):
        """Generate one Paper-2-style desired-vs-actual plot per grasp into
        <run>/Individual_Verifications/, then open the folder's first image."""
        run = self._run_dir()
        ph = os.path.join(run, "pose_history.json")
        if not os.path.exists(ph):
            messagebox.showinfo("Verification",
                f"No pose_history.json in:\n{run}\n\nRun the simulation first.")
            return
        try:
            import importlib.util
            vpath = os.path.join(PROJECT, "viz", "individual_verifications.py")
            if not os.path.exists(vpath):
                # fall back to sim/ or project root
                for alt in (os.path.join(PROJECT, "individual_verifications.py"),
                            os.path.join(PROJECT, "sim", "individual_verifications.py")):
                    if os.path.exists(alt):
                        vpath = alt; break
            spec = importlib.util.spec_from_file_location("indiv_verif", vpath)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            made = mod.plot_all(run)
        except Exception as e:
            messagebox.showerror("Verification", f"Error making plots:\n{e}")
            return
        if not made:
            messagebox.showinfo("Verification",
                "No plots made (pose history may lack pad poses — re-run the collector).")
            return
        # show each verification plot in its OWN separate window
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        for png in made:
            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mpimg.imread(png)); ax.axis("off")
            ax.set_title(os.path.basename(png))
            fig.tight_layout()
        plt.show()
        self.status.config(
            text=f"made {len(made)} verification plots (separate windows) in\n"
                 f"Individual_Verifications/",
            foreground="#0a6")


    def show_temporal(self):
        """Extract the paper's 4 temporal snapshots (5/50/95% + 3s) from this
        run's tactile CSVs and display them (rows=grasps, cols=squeeze stages)."""
        run = self._run_dir()
        import glob
        if not glob.glob(os.path.join(run, "*_s1_tactile_maps.csv")):
            messagebox.showinfo("Temporal Snapshots",
                f"No tactile files in:\n{run}\n\nRun the simulation first.")
            return
        try:
            import importlib.util
            tpath = os.path.join(PROJECT, "viz", "temporal_snapshots.py")
            for alt in (tpath, os.path.join(PROJECT, "temporal_snapshots.py"),
                        os.path.join(PROJECT, "sim", "temporal_snapshots.py")):
                if os.path.exists(alt):
                    tpath = alt; break
            spec = importlib.util.spec_from_file_location("temporal_snapshots", tpath)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            made = mod.plot_run(run)
        except Exception as e:
            messagebox.showerror("Temporal Snapshots", f"Error:\n{e}")
            return
        if not made:
            messagebox.showinfo("Temporal Snapshots", "No snapshots produced.")
            return
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        for png in made:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mpimg.imread(png)); ax.axis("off")
            ax.set_title(os.path.basename(png))
            fig.tight_layout()
        plt.show()
        self.status.config(
            text=f"temporal snapshots saved + shown ({len(made)} sensor plots).",
            foreground="#0a6")


    def show_pad_truth(self):
        """Safety report: GUI-designed pad pose vs pose reached in Isaac.
        Red banner if any grasp exceeds the threshold (once measured exists);
        amber 'FK-only' banner until the physics measurement is wired in."""
        run = self._run_dir()
        if not os.path.exists(os.path.join(run, "pose_history.json")):
            messagebox.showinfo("Pad Truth", f"No pose_history.json in:\n{run}"); return
        try:
            import importlib.util
            ppath = os.path.join(PROJECT, "viz", "pad_truth.py")
            spec = importlib.util.spec_from_file_location("pad_truth", ppath)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            png, any_bad, rows = mod.check_run(run)
        except Exception:
            import traceback; messagebox.showerror("Pad Truth", traceback.format_exc()); return
        if png is None:
            messagebox.showinfo("Pad Truth", "Nothing to check."); return
        import matplotlib.pyplot as plt, matplotlib.image as mpimg
        fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mpimg.imread(png)); ax.axis("off"); fig.tight_layout(); plt.show()
        self.status.config(
            text=("PAD MISMATCH — see report (red)" if any_bad else "pad truth checked"),
            foreground=("#c00" if any_bad else "#0a6"))

    # ---------- Stitching tab (Block 2) ----------
    def _build_stitch_tab(self):
        frm = ttk.Frame(self.tab_stitch, padding=14)
        frm.grid(row=0, column=0, sticky="nw")
        r = 0
        ttk.Label(frm, text="BLOCK 2 — stitch per-grasp maps into ONE extended contact map",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(frm, justify="left", foreground="#555", text=(
            "Every grasp's HOLD-AVERAGE map is projected at its recorded pad offset\n"
            "(pose_history.json) onto one mm canvas; overlapping cells are averaged.\n"
            "Outputs land in <run>/Stitched/:  stitched_s1/s2 .png + .npy (+ mask),\n"
            "and training_pair.npz = INPUT (center grasp) -> TARGET (extended map).")
                  ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 10)); r += 1

        ttk.Label(frm, text="run folder").grid(row=r, column=0, sticky="e")
        self.stitch_run_lbl = ttk.Label(frm, text="(newest run — auto)", foreground="#06a")
        self.stitch_run_lbl.grid(row=r, column=1, sticky="w", padx=4)
        ttk.Button(frm, text="Browse…", command=self._stitch_browse).grid(
            row=r, column=2, sticky="ew"); r += 1
        ttk.Button(frm, text="Use newest run", command=self._stitch_use_newest).grid(
            row=r, column=2, sticky="ew", pady=2); r += 1

        ttk.Label(frm, text="canvas resolution (mm/cell)").grid(row=r, column=0, sticky="e")
        self.vars["stitch_res"] = tk.StringVar(value="1.0")
        ttk.Entry(frm, textvariable=self.vars["stitch_res"], width=8).grid(
            row=r, column=1, sticky="w", padx=4); r += 1

        ttk.Button(frm, text="Build Stitched Maps (s1 + s2)",
                   command=self.do_stitch).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(10, 3)); r += 1
        ttk.Button(frm, text="Export Training Pair (center -> extended)",
                   command=self.do_export_pair).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1

        self.stitch_status = ttk.Label(frm, text="", foreground="#0a6",
                                       wraplength=430, justify="left")
        self.stitch_status.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1

    def _stitch_target_dir(self):
        return self._stitch_dir if getattr(self, "_stitch_dir", None) else self._run_dir()

    def _stitch_browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=os.path.join(PROJECT, "Data", "gui_run"),
            title="Pick a run folder")
        if d:
            self._stitch_dir = d
            self.stitch_run_lbl.config(text=os.path.basename(d))

    def _stitch_use_newest(self):
        self._stitch_dir = None
        self.stitch_run_lbl.config(text="(newest run — auto)")

    def _load_stitching_module(self):
        import importlib.util
        for cand in (os.path.join(PROJECT, "viz", "stitching.py"),
                     os.path.join(PROJECT, "stitching.py"),
                     os.path.join(PROJECT, "sim", "stitching.py")):
            if os.path.exists(cand):
                spec = importlib.util.spec_from_file_location("stitching", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def do_stitch(self):
        import traceback
        try:
            mod = self._load_stitching_module()
            if mod is None:
                messagebox.showerror("Stitching",
                    "stitching.py not found (expected in viz/).")
                return
            run = self._stitch_target_dir()
            try:
                res = float(self.vars["stitch_res"].get())
            except Exception:
                res = 1.0
            made = mod.stitch_run(run, res)
            if not made:
                messagebox.showinfo("Stitching",
                    f"No stitched maps produced in:\n{run}\n\n"
                    "Needs tactile CSVs + pose_history.json (run the sim first).")
                return
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            for png in made:
                fig = plt.figure(figsize=(10.5, 4.8))
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(mpimg.imread(png)); ax.axis("off")
                ax.set_title(os.path.basename(png))
                fig.tight_layout()
            plt.show()
            self.stitch_status.config(
                text=f"stitched {len(made)} sensor map(s) -> "
                     f"{os.path.basename(run)}/Stitched/", foreground="#0a6")
        except Exception:
            messagebox.showerror("Stitching",
                "Stitching failed:\n\n" + traceback.format_exc())

    def do_export_pair(self):
        import traceback
        try:
            mod = self._load_stitching_module()
            if mod is None:
                messagebox.showerror("Stitching",
                    "stitching.py not found (expected in viz/).")
                return
            run = self._stitch_target_dir()
            try:
                res = float(self.vars["stitch_res"].get())
            except Exception:
                res = 1.0
            npz = mod.export_pair(run, res)
            self.stitch_status.config(
                text=f"training pair exported:\n{npz}", foreground="#0a6")
        except Exception:
            messagebox.showerror("Stitching",
                "Export failed:\n\n" + traceback.format_exc())


if __name__ == "__main__":
    root = tk.Tk()
    app = CockpitGUI(root)
    root.mainloop()
