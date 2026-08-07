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

# --- FRONT (Y-Z) preview tick spacing, mm. Labelled ticks every MAJOR,
# --- faint unlabelled gridlines every MINOR. Change these to taste.
PREVIEW_TICK_MAJOR_MM = 10.0
PREVIEW_TICK_MINOR_MM = 2.0

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os, json, subprocess, threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- real hardware sizes (mm) ----
PAD_W = 22.0    # pad short side (4 taxels)
PAD_H = 37.0    # pad long side  (7 taxels)
# Object size. These stay module-level because ~18 places read them, but they
# are now REFRESHED FROM THE GUI on every cfg() call (see _sync_object_size).
# The scene's Object_02/Cylinder is a UNIT mesh scaled by the collector, so a
# new diameter needs no new USD and no STL — just these numbers.
CYL_D = 26.0    # cylinder diameter (mm) — live, set by the Object size field
CYL_L = 140.0   # cylinder length   (mm) — live, set by the Object size field
GRIP_OPEN = 12.0  # half-gap of each pad from the cylinder rim before closing (mm, visual)
ROBOT_BASE_MM = np.array([20.93, -337.5, 992.75])  # robot base_link world (mm)

# ---- paths for the config-save + Isaac-launch bridge (Stage B) ----
PROJECT   = os.path.expanduser("~/Paper3_Simulation")
CONFIG_JSON = os.path.join(PROJECT, "Data", "gui_config.json")
# 3-panel preview snapshot, written next to the config on every save. The
# collector copies it into the run folder so each run carries a picture of the
# grid design it was launched with (added 2026-08-04).
PREVIEW_PNG = os.path.join(PROJECT, "Data", "gui_preview.png")
# colour-scale policy shared with stitching.py / heatmaps.py / temporal_snapshots.py
PLOT_SCALE_JSON = os.path.join(PROJECT, "Data", "plot_scale.json")
# Reachability dry-runs used to land in Data/gui_run beside real data runs,
# so every test produced two folders. They now go to their own directory.
REACH_OUT_DIR = os.path.join(PROJECT, "Data", "reach_check")
CALIB_CONFIG_JSON = os.path.join(PROJECT, "Data", "gui_calib_config.json")
ISAAC_PY  = os.path.expanduser("~/isaacsim/python.sh")
COLLECT_PY = os.path.join(PROJECT, "sim", "collect_from_config.py")
EXAMPLES_DIR = os.path.expanduser("~/Paper3_Simulation/TSF-85/examples")
 
def grid_2d(nx, ny, step_mm, centered=False):
    """Return list of (dx, dy) offsets in mm for the grasp grid.
    dx runs along Z (up/down the face), dy along Y (across it).

    ANCHORED (centered=False, the original behaviour):
      offset (0,0) is the FIRST point (pt00) and the grid grows away from
      it. |n| = number of points on that axis; the SIGN picks direction:
        nx = +3 -> 0, +step, +2*step      nx = -3 -> 0, -step, -2*step

    CENTERED (centered=True):
      the entered pad offset sits in the MIDDLE and the grid mirrors both
      ways, so the initial grasp can be extrapolated in every direction.
      |n| = steps PER SIDE, so an axis holds 2|n|+1 points:
        nx = 2 -> -2, -1, 0, +1, +2  (5 points)
      Total points = (2|nx|+1) * (2|ny|+1) — 2,3 gives 35, not 12.
      Sign is ignored here: a mirrored grid has no direction.
      pt00 is still (0,0), the CENTRE: the collector starts there, and it
      is the frame the stitched training pair is anchored on.
    """
    def axis(n):
        n = int(n) if int(n) != 0 else 1
        sgn = 1.0 if n > 0 else -1.0
        return sgn * np.arange(abs(n)) * step_mm

    def axis_centered(n):
        k = abs(int(n))
        return (np.arange(-k, k + 1) * float(step_mm)) if k else np.zeros(1)

    xs = axis_centered(nx) if centered else axis(nx)
    ys = axis_centered(ny) if centered else axis(ny)
    # SERPENTINE (boustrophedon): reverse every other Y-column so the sweep
    # snakes instead of resetting to the far end of the next column.
    # Straight raster made the column wrap a hypot(Z_span, step) jump — 22.7 mm
    # for a 5-row column — and the grasp right after that jump landed ~2.2 mm
    # short of target in BOTH repeatability runs (A: pt10, B: pt06, the only
    # two 22.7 mm moves in the run). Short moves land to <0.01 mm.
    pts = []
    for j, gy in enumerate(ys):
        col = xs if (j % 2 == 0) else xs[::-1]
        for gx in col:
            pts.append((float(gx), float(gy)))
    if centered:
        # Move the centre to the front WITHOUT reordering the rest: raster
        # order keeps consecutive grasps adjacent, which the pad-to-pad
        # motion with joint-seed continuity depends on.
        for i, (gx, gy) in enumerate(pts):
            if abs(gx) < 1e-9 and abs(gy) < 1e-9:
                pts.insert(0, pts.pop(i))
                break
    return pts

class CockpitGUI:
    def __init__(self, root):
        self.root = root
        root.title("Paper 3 — Collection Cockpit (Stage A)")
        self._show_measured = False   # green 'measured pad' only after Load result

        # defaults in mm (from our proven scene)
        self.vars = {
            "obj_x": tk.StringVar(value="-268.06"),
            "obj_y": tk.StringVar(value="199.0"),
            "obj_z": tk.StringVar(value="1052.2"),
            "obj_diam": tk.StringVar(value="26.0"),         # object diameter mm
            "obj_len": tk.StringVar(value="140.0"),         # object length   mm
            "obj_tilt_deg": tk.StringVar(value="0.0"),      # 0 = standing
            "obj_tilt_axis": tk.StringVar(value="X"),       # tilt about this axis
            # pad pose = offset from OBJECT CENTER (mm). X is fixed (centered grasp).
            "pad_dy": tk.StringVar(value="0.0"),
            "pad_dz": tk.StringVar(value="0.0"),
            # In-plane ROLL of the pad about its own face normal (deg).
            # Emitted as GRASP_ROT_DEG; the collector applies it about the
            # TOOL-LOCAL Y axis (verified 2026-08-03: Y rolls the pad in its
            # face plane; X tips the gripper off the rod).
            "pad_rot": tk.StringVar(value="0.0"),
            "grid_nx":  tk.StringVar(value="2"),
            "grid_ny":  tk.StringVar(value="3"),
            "grid_step": tk.StringVar(value="8.0"),   # mm
            # grid mirrors around the entered pad offset instead of starting there
            "grid_centered": tk.BooleanVar(value=False),
            "headless": tk.BooleanVar(value=False),   # False = show Isaac window
            "calib_headless": tk.BooleanVar(value=False),  # Calibrate tab headless toggle
            "calib_dz": tk.StringVar(value="0.0"),  # Calibrate pad Z offset (Y stays centered)
            # finger-joint angle to squeeze to during calibration.
            # blank = let the collector decide (stored value, else 26 mm value)
            "calib_close_rad": tk.StringVar(value=""),
            "stitch_want_gsr": tk.BooleanVar(value=False),
            # ---- colour-scale policy (2026-08-04) ----
            # Written to Data/plot_scale.json and read by stitching.py,
            # heatmaps.py and temporal_snapshots.py, so every figure in the
            # project is scaled the same way.
            "scale_shared": tk.BooleanVar(value=True),
            "scale_fixed":  tk.StringVar(value=""),   # blank = not fixed  # Stitch-tab: include GSR in validation
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
        # The left panel grew past the window height and had no way to reach
        # the buttons at the bottom. Put it inside a Canvas with a vertical
        # scrollbar: the panel keeps its natural width, only scrolls in Y.
        _outer = ttk.Frame(self.tab_collect)
        _outer.grid(row=0, column=0, sticky="nsew")
        _outer.rowconfigure(0, weight=1)
        _outer.columnconfigure(0, weight=1)

        _cv = tk.Canvas(_outer, highlightthickness=0, borderwidth=0)
        _sb = ttk.Scrollbar(_outer, orient="vertical", command=_cv.yview)
        _cv.configure(yscrollcommand=_sb.set)
        _cv.grid(row=0, column=0, sticky="nsew")
        _sb.grid(row=0, column=1, sticky="ns")

        frm = ttk.Frame(_cv, padding=10)
        _cv.create_window((0, 0), window=frm, anchor="nw")

        def _fit(_e=None):
            _cv.configure(scrollregion=_cv.bbox("all"))
            # never clip horizontally: canvas matches the panel's needed width
            _cv.configure(width=frm.winfo_reqwidth())
        frm.bind("<Configure>", _fit)

        def _wheel(e):
            if getattr(e, "num", None) == 4 or getattr(e, "delta", 0) > 0:
                _cv.yview_scroll(-1, "units")
            elif getattr(e, "num", None) == 5 or getattr(e, "delta", 0) < 0:
                _cv.yview_scroll(1, "units")

        def _grab(_e=None):        # only while the pointer is over this panel,
            for _b in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                _cv.bind_all(_b, _wheel)

        def _release(_e=None):     # so the plot area keeps its own wheel
            for _b in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                _cv.unbind_all(_b)

        for _w in (_cv, frm):
            _w.bind("<Enter>", _grab)
            _w.bind("<Leave>", _release)

        self._inputs_canvas = _cv          # kept for later resizing
        r = 0

        # ---------------- SESSION FOLDER (added 2026-08-04) ----------------
        # One folder per test. Before this, the reachability dry-run and the
        # real run each minted their own timestamp, so a single test produced
        # two folders minutes apart. Now the GUI decides the name ONCE and
        # passes it to both via GRASP_RUN_DIR, and that folder is also the
        # default for loading configs and for re-plotting heatmaps/stitches.
        ttk.Label(frm, text="SESSION FOLDER (one per test)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2,
                                              sticky="w"); r += 1
        self.session_lbl = ttk.Label(frm, text="(none)", foreground="#06a",
                                     wraplength=430, justify="left")
        self.session_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        _sf = ttk.Frame(frm); _sf.grid(row=r, column=0, columnspan=2,
                                       sticky="ew", pady=(2, 6)); r += 1
        ttk.Button(_sf, text="New session (stamp + angles)",
                   command=self.new_session).pack(side="left")
        ttk.Button(_sf, text="Use existing...",
                   command=self.pick_session).pack(side="left", padx=4)
        ttk.Button(_sf, text="Open folder",
                   command=self.open_session_folder).pack(side="left")
        ttk.Label(frm, text="(the folder name is fixed when you press New "
                            "session; press it again after changing angles)",
                  foreground="#888", wraplength=430,
                  justify="left").grid(row=r, column=0, columnspan=2,
                                       sticky="w"); r += 1

        ttk.Label(frm, text="OBJECT pose (world, mm)",
                  font=("", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        for key, lab in [("obj_x", "x"), ("obj_y", "y"), ("obj_z", "z")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="tilt (deg)").grid(row=r, column=0, sticky="e")
        for key, lab in [("obj_diam", "diameter (mm)"), ("obj_len", "length (mm)")]:
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="e")
            e = ttk.Entry(frm, textvariable=self.vars[key], width=12)
            e.grid(row=r, column=1, sticky="w")
            e.bind("<Return>", lambda ev: self.refresh()); r += 1
        self.obj_cal_lbl = ttk.Label(frm, text="", foreground="#555",
                                     wraplength=250)
        self.obj_cal_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

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
        e = ttk.Entry(frm, textvariable=self.vars["pad_rot"], width=12)
        e.grid(row=r, column=1, sticky="w"); e.bind("<Return>", lambda ev: self.refresh()); r += 1
        ttk.Label(frm, text="(pad roll in its own face plane; tool-local Y)",
                  foreground="#888").grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

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
        ttk.Checkbutton(frm, text="centered grid (mirror both sides; n = steps PER SIDE)",
                        variable=self.vars["grid_centered"],
                        command=self.refresh).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self.grid_count = ttk.Label(frm, text="", foreground="#555")
        self.grid_count.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Button(frm, text="Update Preview", command=self.refresh).grid(
            row=r, column=0, columnspan=2, pady=(12, 4), sticky="ew"); r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=6); r += 1
        ttk.Checkbutton(frm, text="Run headless (no Isaac window)",
                        variable=self.vars["headless"]).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Save Config", command=self.save_config).grid(
            row=r, column=0, columnspan=2, pady=4, sticky="ew"); r += 1
        ttk.Button(frm, text="Check Reachability (no motion)",
                   command=self.save_and_show_reach_cmd).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
        ttk.Button(frm, text="Load reachability result",
                   command=self.load_reachability).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
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
        # ---------------- COLOUR SCALE (added 2026-08-04) ----------------
        ttk.Label(frm, text="COLOUR SCALE (all heatmaps / stitches / temporal)",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=2,
                                             sticky="w", pady=(8, 0)); r += 1
        ttk.Checkbutton(frm, text="one scale across the whole run",
                        variable=self.vars["scale_shared"],
                        command=self.save_plot_scale).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        _cs = ttk.Frame(frm); _cs.grid(row=r, column=0, columnspan=2,
                                       sticky="w"); r += 1
        ttk.Label(_cs, text="fixed max (blank = off):").pack(side="left")
        _ce = ttk.Entry(_cs, textvariable=self.vars["scale_fixed"], width=8)
        _ce.pack(side="left", padx=4)
        _ce.bind("<Return>", lambda ev: self.save_plot_scale())
        ttk.Button(_cs, text="Apply", command=self.save_plot_scale).pack(side="left")
        ttk.Label(frm, text="(fixed max makes DIFFERENT tests comparable; "
                            "2400 matches Paper 2's tactile counts)",
                  foreground="#888", wraplength=430,
                  justify="left").grid(row=r, column=0, columnspan=2,
                                       sticky="w"); r += 1

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

    def _cal_entry(self):
        """Calibration entry for the current diameter, or None."""
        try:
            with open(os.path.join(PROJECT, "Data", "pad_offset_calibration.json")) as f:
                cal = json.load(f)
            return cal.get(f"{CYL_D:.1f}")
        except Exception:
            return None

    def _newest_probe(self):
        """Newest pad_truth_probe.json (a real measured grasp), or None."""
        import glob
        try:
            cands = glob.glob(os.path.join(PROJECT, "Data", "gui_run", "*",
                                           "pad_truth_probe.json"))
            if not cands:
                return None
            with open(max(cands, key=os.path.getmtime)) as f:
                return json.load(f)
        except Exception:
            return None

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
            self._sync_object_size(fnum)
            obj = np.array([fnum("obj_x"), fnum("obj_y"), fnum("obj_z")])
            pad = obj + np.array([0.0, fnum("pad_dy"), fnum("pad_dz")])
            return {
                "obj": obj,
                "tilt_deg": fnum("obj_tilt_deg"),
                "tilt_axis": self.vars["obj_tilt_axis"].get(),
                "pad": pad,
                "pad_dy": fnum("pad_dy"),
                "pad_dz": fnum("pad_dz"),
                "pad_rot": fnum("pad_rot"),
                "nx": inum("grid_nx"),
                "ny": inum("grid_ny"),
                "step": fnum("grid_step", 1.0),
                "centered": bool(self.vars["grid_centered"].get()),
            }
        except Exception:
            return None

    def refresh(self):
        cfg = self._read()
        if cfg is None:
            self.info.config(text="check numeric inputs", foreground="red"); return

        obj, pad = cfg["obj"], cfg["pad"]
        offs = grid_2d(cfg["nx"], cfg["ny"], cfg["step"],
                       centered=cfg["centered"])            # (dx,dy) mm
        if hasattr(self, "grid_count"):
            _j = [float(np.hypot(offs[i+1][0]-offs[i][0],
                                 offs[i+1][1]-offs[i][1]))
                  for i in range(len(offs)-1)]
            _mx = max(_j) if _j else 0.0
            self.grid_count.config(
                text=f"{len(offs)} grasp points"
                     + ("  (centered: n = per side)" if cfg["centered"]
                        else "  (anchored at pad offset)")
                     + f"   ~{len(offs) * 2.5:.0f} min"
                     + f"   max jump {_mx:.1f} mm"
                     + ("   ⚠ pad rolled: grid still steps in world Y/Z"
                        if (abs(cfg["pad_rot"]) > 1e-6 and len(offs) > 1) else ""),
                foreground=("#b00" if (_mx > 15.0
                                       or (abs(cfg["pad_rot"]) > 1e-6
                                           and len(offs) > 1)) else "#555"))

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
        # A pad rolled by theta about its own face normal still lies in the same
        # plane, but its shadow along Y grows: W*|cos| + H*|sin|.
        _pr = np.radians(cfg["pad_rot"])
        half_y = (PAD_W*abs(np.cos(_pr)) + PAD_H*abs(np.sin(_pr))) / 2.0
        for gx, gy in offs:
            cy = pad[1] + gy    # the grid's Y offset shifts pads along Y
            # -X pad
            ax.plot([obj[0]-rim, obj[0]-rim], [cy-half_y, cy+half_y],
                    color="crimson", linewidth=3)
            # +X pad
            ax.plot([obj[0]+rim, obj[0]+rim], [cy-half_y, cy+half_y],
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
        # colour by reachability if a report has been loaded:
        #   green = reachable, red = unreachable, crimson = not checked yet
        _rm = getattr(self, "reach_map", None) or {}
        _lbl_done = {"ok": False, "bad": False, "raw": False}
        for i, (gx, gy) in enumerate(offs):
            py = pad[1] + gy - PAD_W/2
            pz = pad[2] + gx - PAD_H/2   # use X-grid as the up/down (Z) sweep on the face
            if i in _rm:
                _ok = bool(_rm[i])
                col = "#0a9d3a" if _ok else "#d81b1b"
                key = "ok" if _ok else "bad"
                lab = ("pad (reachable)" if _ok else "pad (UNREACHABLE)") \
                      if not _lbl_done[key] else None
                _lbl_done[key] = True
                ls = "-" if _ok else "--"
            else:
                col, ls = "crimson", "-"
                lab = "pad" if not _lbl_done["raw"] else None
                _lbl_done["raw"] = True
            if abs(cfg["pad_rot"]) < 1e-6:
                ax.add_patch(mpatches.Rectangle((py, pz), PAD_W, PAD_H,
                                                fill=False, edgecolor=col,
                                                linewidth=1.5, linestyle=ls,
                                                label=lab))
            else:
                # Rotate the footprint about the PAD CENTRE, not a corner.
                # Drawn as a Polygon so this works on any matplotlib version
                # (Rectangle's rotation_point= needs >= 3.6).
                _cy, _cz = pad[1] + gy, pad[2] + gx
                _k = np.array([[-PAD_W/2, -PAD_H/2], [ PAD_W/2, -PAD_H/2],
                               [ PAD_W/2,  PAD_H/2], [-PAD_W/2,  PAD_H/2]])
                _c, _s = np.cos(_pr), np.sin(_pr)
                _kr = _k @ np.array([[_c, _s], [-_s, _c]])
                ax.add_patch(mpatches.Polygon(_kr + [_cy, _cz], closed=True,
                                              fill=False, edgecolor=col,
                                              linewidth=1.5, linestyle=ls,
                                              label=lab))
            ax.scatter(pad[1]+gy, pad[2]+gx, color=col, s=10)
        # visit path: the exact order the collector executes (pt00 -> last)
        if len(offs) > 1:
            px = [pad[1] + gy for gx, gy in offs]
            pz = [pad[2] + gx for gx, gy in offs]
            ax.plot(px, pz, color="dimgray", linestyle="--", linewidth=1.2,
                    alpha=0.9, zorder=4, label="visit path")
            # start/last markers are HOLLOW rings so they never hide the
            # green/red reachability colour of the dot underneath them.
            ax.scatter(px[0], pz[0], facecolors="none", edgecolors="green",
                       s=90, linewidths=1.8, zorder=5, label="pt00 (start = base)")
            ax.scatter(px[-1], pz[-1], facecolors="none", edgecolors="red",
                       marker="s", s=90, linewidths=1.8, zorder=5, label="last pt")
            if len(offs) <= 24:
                for i, (x_, z_) in enumerate(zip(px, pz)):
                    ax.annotate(str(i), (x_, z_), textcoords="offset points",
                                xytext=(3, 3), fontsize=6, color="dimgray")
        ax.set_aspect("equal", adjustable="datalim")

        # ---------- flange / palm / pad-centre overlay (calibration made visible) --
        # Shows, at the anchor pad (pt00), where the EE flange and gripper palm sit
        # above the pad, using the CALIBRATED offset for this diameter. Makes the
        # ~157mm EE->pad and the palm-vs-rod clearance visible before you run.
        try:
            cal = self._cal_entry()
            if cal is not None:
                off_mm   = float(cal["TOOL_OFFSET_Z"]) * 1000.0      # pad centre -> EE
                palm_mm  = 10.79                                     # EE -> palm (measured)
                anchor_y = pad[1] + offs[0][1]
                anchor_z = pad[2] + offs[0][0]                       # pad-centre target
                ee_z     = anchor_z + off_mm

                PALM_HOUSING_DROP_MM = 75.9
                palm_z = ee_z - 10.79 - PALM_HOUSING_DROP_MM



                # palm_z   = ee_z - palm_mm
                rod_top  = obj[2] + CYL_L / 2
                ax.scatter([anchor_y], [ee_z], marker="v", s=70, color="#333",
                           zorder=6, label="EE (flange)")
                ax.scatter([anchor_y], [palm_z], marker="_", s=200, color="#8a6d3b",
                           zorder=6, label="palm")
                ax.plot([anchor_y, anchor_y], [palm_z, anchor_z], color="#999",
                        lw=1, ls=":", zorder=3)
                clr = palm_z - rod_top
                ax.annotate(f"EE↕pad {off_mm:.0f}mm\npalm↕rodtop {clr:+.0f}mm",
                            (anchor_y, ee_z), textcoords="offset points",
                            xytext=(6, -2), fontsize=6, color="#333")
        except Exception:
            pass

        ax.legend(fontsize=6, loc="upper right")
        # Fine ticks: the grid steps are ~5 mm, so 20/50 mm ticks were far too
        # coarse to read a pad position off the plot.
        ax.xaxis.set_major_locator(MultipleLocator(PREVIEW_TICK_MAJOR_MM))
        ax.yaxis.set_major_locator(MultipleLocator(PREVIEW_TICK_MAJOR_MM))
        ax.xaxis.set_minor_locator(MultipleLocator(PREVIEW_TICK_MINOR_MM))
        ax.yaxis.set_minor_locator(MultipleLocator(PREVIEW_TICK_MINOR_MM))
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="both", which="minor", length=2)
        ax.grid(which="major", alpha=0.35, linewidth=0.7)
        ax.grid(which="minor", alpha=0.15, linewidth=0.4)

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
        offs = grid_2d(cfg["nx"], cfg["ny"], cfg["step"],
                       centered=cfg["centered"])   # (dx,dy) mm on the face
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
                # Documentation only — the collector reads the roll from
                # GRASP_ROT_DEG / GRASP_ROT_AXIS, not from this file.
                "rotation_deg": cfg["pad_rot"],
                "rotation_axis": "y",
            },
            "grid": {
                "nx": cfg["nx"], "ny": cfg["ny"], "step_mm": cfg["step"],
                "centered": cfg["centered"],
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
        # Snapshot the three preview plots next to the config. The collector
        # copies BOTH into the run folder, so no manual screenshots are needed.
        try:
            self.fig.savefig(PREVIEW_PNG, dpi=110, bbox_inches="tight")
            _png_note = "\npreview image saved"
        except Exception as e:
            _png_note = f"\n(preview image NOT saved: {e})"

        # ALSO drop both straight into the session folder, so pressing Save
        # Config is enough — you no longer have to run the grid before the
        # folder has a record of what you set up (fixed 2026-08-04).
        _sess_note = ""
        _sess = self._session_or_none()
        if _sess:
            try:
                os.makedirs(_sess, exist_ok=True)
                with open(os.path.join(_sess, "gui_config_used.json"), "w") as f:
                    json.dump(cfg, f, indent=2)
                try:
                    self.fig.savefig(os.path.join(_sess, "gui_preview.png"),
                                     dpi=110, bbox_inches="tight")
                except Exception:
                    pass
                _sess_note = "\nalso saved into session folder"
            except Exception as e:
                _sess_note = f"\n(session copy failed: {e})"

        self.status.config(
            text=f"saved config: {cfg['grid']['n_points']} points\n{CONFIG_JSON}"
                 + _png_note + _sess_note,
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
        """Load a saved recipe OR a plain config back into the GUI fields.

        THREE file shapes are accepted, because all three describe a run:
          1. an experiment recipe from "Save Experiment As..."   (gui_fields)
          2. Data/gui_config.json                                (built config)
          3. a run folder's gui_config_used.json / pose_history.json
        Before 2026-08-04 only (1) worked: picking a config set NOTHING and
        still reported success in green, which is why the button looked broken.
        Now (2) and (3) are mapped back onto the fields, and a file that
        matches nothing says so instead of pretending.
        """
        from tkinter import filedialog
        exp_dir = os.path.join(PROJECT, "Data", "experiments")
        path = filedialog.askopenfilename(
            initialdir=(self._session_or_none() or
                        (exp_dir if os.path.isdir(exp_dir)
                         else os.path.join(PROJECT, "Data"))),
            filetypes=[("Experiment or config JSON", "*.json"), ("All", "*.*")],
            title="Load experiment recipe or config")
        if not path:
            return
        try:
            with open(path) as f:
                doc = json.load(f)

            # pose_history.json nests the config one level down
            if "gui_fields" not in doc and isinstance(doc.get("config"), dict):
                doc = doc["config"]

            fields, src = {}, ""
            if isinstance(doc.get("gui_fields"), dict):
                fields, src = dict(doc["gui_fields"]), "experiment recipe"
            elif "object" in doc and "grid" in doc:
                o = doc.get("object", {}) or {}
                p = doc.get("pad", {}) or {}
                g = doc.get("grid", {}) or {}
                c = list(o.get("center_world_mm", [None, None, None]))
                fields = {
                    "obj_x": c[0], "obj_y": c[1], "obj_z": c[2],
                    "obj_diam": o.get("diameter_mm"),
                    "obj_len": o.get("length_mm"),
                    "obj_tilt_deg": o.get("tilt_deg"),
                    "obj_tilt_axis": o.get("tilt_axis"),
                    "pad_dy": p.get("base_offset_y_mm"),
                    "pad_dz": p.get("base_offset_z_mm"),
                    "pad_rot": p.get("rotation_deg", 0.0),
                    "grid_nx": g.get("nx"), "grid_ny": g.get("ny"),
                    "grid_step": g.get("step_mm"),
                }
                fields = {k: v for k, v in fields.items() if v is not None}
                src = "config"

            n = 0
            for k, v in fields.items():
                if k in self.vars:
                    self.vars[k].set(f"{v:g}" if isinstance(v, float) else v)
                    n += 1
            if src == "config" and "grid_centered" in self.vars:
                self.vars["grid_centered"].set(
                    bool((doc.get("grid") or {}).get("centered", False)))
                n += 1

            if n == 0:
                messagebox.showwarning(
                    "Load",
                    "Nothing loaded — that file has no GUI fields and no "
                    "object/grid blocks I can read.\n\nPick an experiment recipe "
                    "from Data/experiments, Data/gui_config.json, or a run's "
                    "gui_config_used.json.")
                self.status.config(text="nothing loaded from that file",
                                   foreground="#b00")
                return

            self.refresh()
            self.status.config(
                text=f"{src} loaded ({n} fields):\n{os.path.basename(path)}",
                foreground="#0a6")
        except Exception as e:
            messagebox.showerror("Experiment", "Could not load:\n%s" % e)

    def save_and_show_cmd(self):
        cfg = self.save_config()
        if cfg is None:
            return
        # the exact, proven terminal command (this is what worked for you)
        headless = "1" if self.vars["headless"].get() else "0"
        _rot = cfg.get("pad", {}).get("rotation_deg", 0.0)
        _sess = self._session_or_none()
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            + (f'GRASP_RUN_DIR="{_sess}" \\\n' if _sess else
               f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n')
            + f'GRASP_BASENAME="gui" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            + (f'GRASP_ROT_DEG="{_rot:g}" \\\n'
               f'GRASP_ROT_AXIS="y" \\\n' if abs(_rot) > 1e-6 else "")
            + f"{ISAAC_PY} {COLLECT_PY} \\\n"
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

    # ================= Reachability =================
    def save_and_show_reach_cmd(self):
        """Dry-run every grid point in Isaac (IK + Paper-2 limit gates). No motion."""
        cfg = self.save_config()
        if cfg is None:
            return
        headless = "1" if self.vars["headless"].get() else "0"
        _rot = cfg.get("pad", {}).get("rotation_deg", 0.0)
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            + (f'GRASP_RUN_DIR="{self._session_or_none()}" \\\n'
               if self._session_or_none() else
               f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/reach_check" \\\n')
            + f'GRASP_BASENAME="reach" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            f'GRASP_REACH_ONLY="1" \\\n'
            + (f'GRASP_ROT_DEG="{_rot:g}" \\\n'
               f'GRASP_ROT_AXIS="y" \\\n' if abs(_rot) > 1e-6 else "")
            + f"{ISAAC_PY} {COLLECT_PY} \\\n"
            f"  --config {CONFIG_JSON}"
        )
        win = tk.Toplevel(self.root)
        win.title("Reachability check — copy into a terminal")
        tk.Label(win, justify="left",
                 text=("Config saved. This checks EVERY grid point and writes\n"
                       "reachability_report.json. The robot does NOT move.\n"
                       "Then press 'Load reachability result' to colour the grid.")
                 ).pack(anchor="w", padx=10, pady=(10, 4))
        txt = tk.Text(win, width=82, height=8, wrap="none")
        txt.insert("1.0", cmd); txt.pack(padx=10, pady=4)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(cmd)
            self.status.config(text="reachability command copied.", foreground="#0a6")
        tk.Button(win, text="Copy to clipboard", command=_copy).pack(pady=(4, 10))
        self.status.config(text="config saved. copy the reachability command.",
                           foreground="#0a6")

    def _find_reach_report(self):
        """Newest reachability_report.json: next to the config, else newest run dir."""
        cands = []
        side = os.path.join(os.path.dirname(CONFIG_JSON), "reachability_report.json")
        if os.path.exists(side):
            cands.append(side)
        run_root = os.path.join(PROJECT, "Data", "gui_run")
        if os.path.isdir(run_root):
            for d in os.listdir(run_root):
                p = os.path.join(run_root, d, "reachability_report.json")
                if os.path.exists(p):
                    cands.append(p)
        if not cands:
            return None
        return max(cands, key=os.path.getmtime)

    def load_reachability(self):
        """Load the newest report and colour the grid green/red."""
        path = self._find_reach_report()
        if path is None:
            messagebox.showinfo("Reachability",
                                "No reachability_report.json found yet.\n\n"
                                "Press 'Check Reachability (no motion)' first and run "
                                "the command it shows.")
            return
        try:
            with open(path) as f:
                rep = json.load(f)
        except Exception as e:
            messagebox.showerror("Reachability", f"Could not read:\n{path}\n\n{e}")
            return
        self.reach_map = {int(p["index"]): bool(p.get("reachable", False))
                          for p in rep.get("points", [])}
        n_ok = sum(1 for v in self.reach_map.values() if v)
        n_all = len(self.reach_map)
        bad = [f"pt{i:02d}" for i, v in sorted(self.reach_map.items()) if not v]
        self.refresh()
        msg = f"{n_ok}/{n_all} points reachable."
        if bad:
            reasons = {p["index"]: p.get("reason", "") for p in rep.get("points", [])}
            detail = "\n".join(f"  pt{i:02d}: {reasons.get(i,'')}"
                               for i, v in sorted(self.reach_map.items()) if not v)
            msg += f"\n\nUNREACHABLE (will be skipped):\n{detail}"
        self.status.config(
            text=f"reachability: {n_ok}/{n_all} OK" + (f", {len(bad)} skipped" if bad else ""),
            foreground="#0a6" if not bad else "#c60")
        messagebox.showinfo("Reachability", f"{msg}\n\nreport:\n{path}")

    # ================= Calibrate tab =================
    def _sync_object_size(self, fnum):
        """Copy the GUI's diameter/length into the module-level CYL_D/CYL_L.

        Everything downstream (preview, palm-clearance line, 3D scene,
        reachability, the written config) already reads those two names, so
        this one call makes the whole GUI follow the entry fields. Also warns
        when the chosen diameter has no calibration yet -- the collector will
        refuse to run in that case, and it is better to see it here."""
        global CYL_D, CYL_L
        d = fnum("obj_diam", 26.0)
        L = fnum("obj_len", 140.0)
        if d > 0:
            CYL_D = d
        if L > 0:
            CYL_L = L
        if not hasattr(self, "obj_cal_lbl"):
            return
        try:
            with open(os.path.join(PROJECT, "Data",
                                   "pad_offset_calibration.json")) as f:
                cal = json.load(f)
        except Exception:
            cal = {}
        ent = cal.get(f"{CYL_D:.1f}")
        if ent:
            self.obj_cal_lbl.config(
                text=f"\u00d8{CYL_D:.1f} calibrated  "
                     f"(TOOL_OFFSET_Z {ent.get('TOOL_OFFSET_Z')}, "
                     f"close_rad {ent.get('close_rad')})",
                foreground="#070")
        else:
            est = max(0.05, (85.0 - CYL_D) / 106.0)
            self.obj_cal_lbl.config(
                text=f"\u00d8{CYL_D:.1f} NOT calibrated - collection will "
                     f"refuse. Calibrate tab first; try "
                     f"GRASP_CLOSE_RAD={est:.3f}",
                foreground="#b00")

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

        ttk.Label(frm, text="close_rad").grid(row=r, column=0, sticky="e")
        e = ttk.Entry(frm, textvariable=self.vars["calib_close_rad"], width=10)
        e.grid(row=r, column=1, sticky="w")
        e.bind("<Return>", lambda _e: self.refresh_calib()); r += 1
        self.calib_rad_lbl = ttk.Label(frm, text="", foreground="#555",
                                       wraplength=250)
        self.calib_rad_lbl.grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Button(frm, text="Use estimate for this diameter",
                   command=self._fill_close_rad_estimate).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(0, 6)); r += 1

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
            row=r, column=0, sticky="ew", pady=2)
        ttk.Button(frm, text="Reset view",
                   command=self.reset_calibration_view).grid(
            row=r, column=1, sticky="ew", pady=2); r += 1
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
        self._update_close_rad_hint()
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

        # ---- overlay the MEASURED pad centre from the last real grasp (#2) ----
        # Only shown AFTER 'Load calibration result' is pressed (self._show_measured),
        # so opening the GUI shows a clean target-only plot, not a stale run.
        try:
            if getattr(self, "_show_measured", False):
                pr = self._newest_probe()
                cal = self._cal_entry()
                if pr and cal and ("TSF_right_CASE_closed_world_mm" in pr):
                    r = np.array(pr["TSF_right_CASE_closed_world_mm"])
                    l = np.array(pr["TSF_left_CASE_closed_world_mm"])
                    case_mid = 0.5 * (r + l)                 # Case origin (mm, world)
                    shift = float(cal.get("pad_center_above_case_m", 0.0)) * 1000.0
                    meas_y = case_mid[1]
                    meas_z = case_mid[2] - shift             # Case origin -> pad centre
                    ax.add_patch(mpatches.Rectangle((meas_y-PAD_W/2, meas_z-PAD_H/2),
                                 PAD_W, PAD_H, fill=False, edgecolor="#0a9d3a",
                                 linewidth=1.6, linestyle="--", label="pad measured"))
                    ax.scatter(meas_y, meas_z, color="#0a9d3a", s=12)
                    ax.annotate(f"Δz {meas_z-pad_z:+.1f} mm", (meas_y, meas_z),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=7, color="#0a7d2a")
        except Exception:
            pass

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

    def close_rad_estimate(self, d_mm=None):
        """Finger-joint angle that closes the 2F-85 to `d_mm`.

        The jaw span is close to linear in the joint angle: fully open is
        ~85 mm at ~0 rad, and the VERIFIED 26 mm grasp sits at 0.55 rad, so
        span ~= 85 - 106*rad. This is a STARTING POINT, not a calibration —
        the calibrate run checks the tactile peak against the 26 mm
        reference and refuses to store anything outside a sane band."""
        d = CYL_D if d_mm is None else float(d_mm)
        return max(0.05, (85.0 - d) / 106.0)

    def _fill_close_rad_estimate(self):
        self.vars["calib_close_rad"].set(f"{self.close_rad_estimate():.3f}")
        self.refresh_calib()

    def _update_close_rad_hint(self):
        """Show where close_rad will come from, and what it means in mm."""
        if not hasattr(self, "calib_rad_lbl"):
            return
        txt = self.vars["calib_close_rad"].get().strip()
        ent = self._cal_entry()
        est = self.close_rad_estimate()
        if txt:
            try:
                rad = float(txt)
            except ValueError:
                self.calib_rad_lbl.config(
                    text="close_rad must be a number (or blank)",
                    foreground="#b00")
                return
            span = 85.0 - 106.0 * rad
            self.calib_rad_lbl.config(
                text=f"squeezes to ~{span:.1f} mm span vs \u00d8{CYL_D:.1f} "
                     f"object  ({CYL_D - span:+.1f} mm of squeeze). "
                     f"Estimate for this diameter: {est:.3f}",
                foreground=("#070" if 0.0 < (CYL_D - span) < 8.0 else "#b00"))
        elif ent and ent.get("close_rad") is not None:
            self.calib_rad_lbl.config(
                text=f"blank -> collector uses the stored "
                     f"{float(ent['close_rad']):.3f} rad for this diameter",
                foreground="#555")
        else:
            self.calib_rad_lbl.config(
                text=f"blank -> collector falls back to the 26 mm value "
                     f"0.550 rad, which will MISS a \u00d8{CYL_D:.1f} object. "
                     f"Use {est:.3f}",
                foreground="#b00")

    def save_and_show_calib_cmd(self):
        cfg = self.build_calib_config()
        if cfg is None:
            messagebox.showerror("Calibrate", "Check numeric inputs."); return
        os.makedirs(os.path.dirname(CALIB_CONFIG_JSON), exist_ok=True)
        with open(CALIB_CONFIG_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        headless = "1" if self.vars["calib_headless"].get() else "0"
        _crad = self.vars["calib_close_rad"].get().strip()
        if _crad:
            try:
                float(_crad)
            except ValueError:
                messagebox.showerror("Calibrate",
                                     "close_rad must be a number, or blank.")
                return
        cmd = (
            f"cd {EXAMPLES_DIR} && \\\n"
            f'GRASP_OUTPUT_DIR="$HOME/Paper3_Simulation/Data/gui_run" \\\n'
            f'GRASP_BASENAME="calib" \\\n'
            f'GRASP_HEADLESS="{headless}" \\\n'
            f'GRASP_CALIBRATE="1" \\\n'
            + (f'GRASP_CLOSE_RAD="{_crad}" \\\n' if _crad else "")
            + f"{ISAAC_PY} {COLLECT_PY} \\\n"
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
            v = cal[key]
            lines += ["", f"Current object (\u00d8{CYL_D}):  [calibrated]",
                      f"  TOOL_OFFSET_Z   = {v.get('TOOL_OFFSET_Z')} m",
                      f"  method          = {v.get('method','?')}"]
            if "TOOL_OFFSET_Z_case_origin" in v:
                lines += [f"  case-origin      = {v['TOOL_OFFSET_Z_case_origin']}",
                          f"  + pad-centre     = {v.get('pad_center_above_case_m')}"]
            # left/right pad symmetry (from the calibration grasp)
            try:
                pr = np.array(v["pad_right_closed_world_m"]) * 1000
                pl = np.array(v["pad_left_closed_world_m"]) * 1000
                lines += ["", "Pad symmetry (L vs R):",
                          f"  Z apart : {abs(pr[2]-pl[2]):.2f} mm  (should be ~0)",
                          f"  X gap   : {abs(pr[0]-pl[0]):.2f} mm  (pad-origin to origin;"
                          f" not the contact gap)"]
            except Exception:
                pass
            # landing error from the newest real grasp
            try:
                probe = self._newest_probe()
                if probe and "TSF_right_CASE_closed_world_mm" in probe:
                    r = np.array(probe["TSF_right_CASE_closed_world_mm"])
                    l = np.array(probe["TSF_left_CASE_closed_world_mm"])
                    shift = float(v.get("pad_center_above_case_m", 0.0)) * 1000
                    meas_z = 0.5*(r[2]+l[2]) - shift
                    tgt_z = probe.get("gui_target_m", [0,0,0])[2]*1000
                    lines += ["", "Last grasp landing:",
                              f"  pad centre Z : {meas_z:.1f} mm",
                              f"  target Z     : {tgt_z:.1f} mm",
                              f"  error        : {meas_z-tgt_z:+.1f} mm"]
            except Exception:
                pass
        else:
            lines.append(f"\nCurrent object (\u00d8{CYL_D}): NOT calibrated yet")
        self.calib_result.insert("end", "\n".join(lines))
        self.calib_status.config(text="calibration loaded.", foreground="#0a6")
        self._show_measured = True    # now the green measured pad may draw
        self.refresh_calib()   # refresh the preview so the measured pad shows

    def reset_calibration_view(self):
        """Clear the measured overlay + text -> a fresh target-only plot."""
        self._show_measured = False
        try:
            self.calib_result.delete("1.0", "end")
        except Exception:
            pass
        self.calib_status.config(text="view reset - target only.", foreground="#06a")
        self.refresh_calib()

    # ---------- Stage C: read-back heatmaps + pose history ----------
    def _run_dir(self):
        # Priority: an explicit "Plot from folder..." override, then the active
        # SESSION folder, then the newest run on disk.
        forced = getattr(self, "_forced_run_dir", None)
        if forced:
            return forced
        sess = getattr(self, "_session_dir", None)
        if sess and os.path.isdir(sess):
            return sess
        import glob
        base = os.path.join(PROJECT, "Data", "gui_run")
        runs = sorted(glob.glob(os.path.join(base, "run_*")))
        if runs:
            return runs[-1]          # newest run
        return base                   # fallback (old flat layout)

    def save_plot_scale(self):
        """Write Data/plot_scale.json — read by stitching / heatmaps /
        temporal_snapshots so the GUI and standalone runs agree."""
        raw = self.vars["scale_fixed"].get().strip()
        fixed = None
        if raw:
            try:
                fixed = float(raw)
                if fixed <= 0:
                    fixed = None
            except ValueError:
                messagebox.showerror("Colour scale",
                                     "Fixed max must be a number (or blank).")
                return
        doc = {"shared": bool(self.vars["scale_shared"].get()),
               "fixed_vmax": fixed}
        try:
            os.makedirs(os.path.dirname(PLOT_SCALE_JSON), exist_ok=True)
            with open(PLOT_SCALE_JSON, "w") as f:
                json.dump(doc, f, indent=2)
        except Exception as e:
            messagebox.showerror("Colour scale", "Could not save:\n%s" % e)
            return
        mode = (f"fixed {fixed:g}" if fixed is not None
                else ("shared across run" if doc["shared"] else "auto per figure"))
        self.status.config(text="colour scale: " + mode + "\n(replot to apply)",
                           foreground="#0a6")

    # ================= Session folder =================
    @staticmethod
    def _ang_tag(v):
        """Angle -> filename-safe token. -10 -> 'm10', 35.5 -> '35p5'."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        s = f"{abs(v):g}".replace(".", "p")
        return ("m" + s) if v < 0 else s

    def _session_name(self):
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        obj = self._ang_tag(self.vars["obj_tilt_deg"].get())
        pad = self._ang_tag(self.vars["pad_rot"].get())
        return f"run_{stamp}_obj{obj}_pad{pad}"

    def new_session(self):
        """Mint a new session folder from the clock and the current angles.
        Everything after this — reachability, the run, heatmaps, stitching —
        lands in this one folder."""
        path = os.path.join(PROJECT, "Data", "gui_run", self._session_name())
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Session", "Could not create folder:\n%s" % e)
            return
        self._session_dir = path
        self._forced_run_dir = None      # session takes over as the plot source
        self._stitch_dir = None
        self.session_lbl.config(text=os.path.basename(path), foreground="#06a")
        if hasattr(self, "plot_src_lbl"):
            self.plot_src_lbl.config(text="plot source: session folder",
                                     foreground="#06a")
        if hasattr(self, "stitch_run_lbl"):
            self.stitch_run_lbl.config(text=os.path.basename(path))
        self.status.config(text="new session:\n" + path, foreground="#0a6")

    def pick_session(self):
        """Point the session at an existing run folder (to re-plot or re-stitch)."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=os.path.join(PROJECT, "Data", "gui_run"),
            title="Pick an existing run folder to use as the session")
        if not d:
            return
        self._session_dir = d
        self._forced_run_dir = None
        self._stitch_dir = None
        self.session_lbl.config(text=os.path.basename(d), foreground="#06a")
        if hasattr(self, "plot_src_lbl"):
            self.plot_src_lbl.config(text="plot source: session folder",
                                     foreground="#06a")
        if hasattr(self, "stitch_run_lbl"):
            self.stitch_run_lbl.config(text=os.path.basename(d))
        self.status.config(text="session set to:\n" + d, foreground="#0a6")

    def open_session_folder(self):
        """Open the active plot/session folder in the desktop file manager.
        Falls back to the folder the plot buttons are pointing at, so this
        works even before a session has been minted."""
        import subprocess, sys
        path = self._session_or_none() or self._run_dir()
        if not path or not os.path.isdir(path):
            messagebox.showinfo(
                "Open folder",
                "No folder to open yet.\n\nPress 'New session' first, or run a "
                "grid so there is something on disk.")
            return
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)                        # noqa: S606
            else:
                subprocess.Popen(["xdg-open", path])
            self.status.config(text="opened:\n" + path, foreground="#0a6")
        except Exception as e:
            messagebox.showerror("Open folder",
                                 f"Could not open:\n{path}\n\n{e}")

    def _session_or_none(self):
        return getattr(self, "_session_dir", None)

    def choose_plot_folder(self):
        """Pick a saved run folder to (re)generate plots from. All four plot
        buttons then read from — and save back into — this folder."""
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=(self._session_or_none()
                        or os.path.join(PROJECT, "Data", "gui_run")),
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

        # ---- Block 2 validation: stitch round-trip fidelity (SSIM / TC / GSR) ----
        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(10, 6)); r += 1
        ttk.Label(frm, text="VALIDATE stitch fidelity (round-trip)",
                  font=("", 9, "bold")).grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(frm, justify="left", foreground="#555", wraplength=430, text=(
            "Re-samples the stitched canvas at each grasp's own taxel positions and\n"
            "compares recovered-vs-original with SSIM, Tactile-Centroid error, and\n"
            "GSR. This checks the STITCHER as a container (high SSIM / low TC is\n"
            "expected, esp. the center grasp) — it is NOT model completion.\n"
            "GSR needs TensorFlow; it prints 'disabled' and skips if unavailable.")
                  ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 4)); r += 1
        ttk.Checkbutton(frm, text="include GSR (needs TensorFlow + model)",
                        variable=self.vars["stitch_want_gsr"]).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Button(frm, text="Validate Stitch (SSIM / TC / GSR)",
                   command=self.do_validate).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(4, 3)); r += 1

        self.stitch_status = ttk.Label(frm, text="", foreground="#0a6",
                                       wraplength=430, justify="left")
        self.stitch_status.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1

    def _stitch_target_dir(self):
        return self._stitch_dir if getattr(self, "_stitch_dir", None) else self._run_dir()

    def _stitch_browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            initialdir=(self._session_or_none()
                        or os.path.join(PROJECT, "Data", "gui_run")),
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

    def _load_validation_module(self):
        """Load viz/validation.py. It imports stitching.py itself, so make sure
        the viz/ dir is on sys.path first."""
        import importlib.util, sys
        for cand in (os.path.join(PROJECT, "viz", "validation.py"),
                     os.path.join(PROJECT, "validation.py"),
                     os.path.join(PROJECT, "sim", "validation.py")):
            if os.path.exists(cand):
                d = os.path.dirname(cand)
                if d not in sys.path:
                    sys.path.insert(0, d)
                spec = importlib.util.spec_from_file_location("validation", cand)
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

    def do_validate(self):
        """Round-trip validate the stitch (SSIM / TC / GSR) and show the report
        in a scrollable window. Runs in a thread because GSR/TensorFlow can take
        a few seconds to import the first time."""
        import traceback, threading
        run = self._stitch_target_dir()
        try:
            res = float(self.vars["stitch_res"].get())
        except Exception:
            res = 1.0
        want_gsr = bool(self.vars["stitch_want_gsr"].get())
        self.stitch_status.config(
            text="validating stitch (this can take a few seconds"
                 + (", loading TensorFlow for GSR…" if want_gsr else "") + ")",
            foreground="#06a")

        def _worker():
            try:
                mod = self._load_validation_module()
                if mod is None:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Validate", "validation.py not found (expected in viz/)."))
                    return
                results, report = mod.validate_and_save(run, res, want_gsr=want_gsr)
                self.root.after(0, lambda: self._show_validation_report(run, report))
            except Exception:
                tb = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror(
                    "Validate", "Validation failed:\n\n" + tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_validation_report(self, run, report):
        win = tk.Toplevel(self.root)
        win.title("Stitch round-trip validation — " + os.path.basename(run))
        txt = tk.Text(win, width=78, height=28, wrap="none",
                      font=("TkFixedFont", 10))
        txt.insert("1.0", report)
        txt.configure(state="normal")
        yscroll = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=yscroll.set)
        txt.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        yscroll.grid(row=0, column=1, sticky="ns", pady=10)
        win.columnconfigure(0, weight=1); win.rowconfigure(0, weight=1)
        def _copy():
            self.root.clipboard_clear(); self.root.clipboard_append(report)
        ttk.Button(win, text="Copy report", command=_copy).grid(
            row=1, column=0, columnspan=2, pady=(0, 10))
        self.stitch_status.config(
            text="validation done → saved to " + os.path.basename(run)
                 + "/Stitched/validation_report.txt", foreground="#0a6")

if __name__ == "__main__":
    root = tk.Tk()
    app = CockpitGUI(root)
    root.mainloop()
