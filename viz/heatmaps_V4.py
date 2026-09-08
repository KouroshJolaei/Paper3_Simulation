"""
heatmaps.py — Per-grasp HOLD-AVERAGE heatmaps, s1 | s2 side by side.

The representative map of a grasp is the HOLD-AVERAGE (project convention):
the mean of all frames whose taxel SUM is >= 0.5 x that grasp's peak sum.

For every grasp in a run folder this makes ONE figure:
    left = s1, right = s2, shared color scale, peak sum in each title,
and saves it to  <run_dir>/Heatmaps/heatmap_<tag>.png.

Pure post-processing (PyCharm side). Uses a standalone Figure + Agg canvas —
it NEVER touches the global matplotlib backend (GUI safety rule).

Usage:
  from heatmaps import plot_run
  pngs = plot_run(run_dir)
or standalone:
  python3 heatmaps.py <run_dir>
"""

import os, sys, glob
import numpy as np
import pandas as pd

HOLD_FRAC = 0.9      # FALLBACK ONLY — hold_average() below delegates to
                     # stitching.hold_average whenever stitching imports.
                     # Kept in step with stitching.HOLD_FRAC so the fallback
                     # cannot silently disagree with the real pipeline.

# ---- tolerate the tactile-writer race (see stitching._read_tactile_csv) ----
# Berith's per-frame writer occasionally collides two writes into one line,
# giving 31 fields instead of 30. One bad frame out of ~325 is irrelevant
# here, so skip it rather than abort the whole figure.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# hold_average is imported from stitching too: keeping a second copy here
# meant heatmaps and stitches could silently disagree (this file used
# 0.5*peak as the hold threshold, stitching uses min + 0.5*range, and only
# stitching knew about baseline subtraction).
try:
    from stitching import _read_tactile_csv as _read_csv_tolerant
    from stitching import hold_average as _hold_average_shared
    # colour policy lives in stitching.py (one source of truth, 2026-08-04)
    from stitching import resolve_vmax as _resolve_vmax
    from stitching import run_hold_peak as _run_hold_peak
except Exception:                                    # standalone fallback
    _hold_average_shared = None
    _resolve_vmax = None
    _run_hold_peak = None

    def _read_csv_tolerant(path):
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except TypeError:                            # pandas < 1.3
            return pd.read_csv(path, error_bad_lines=False,
                               warn_bad_lines=False)

MIRROR_S2 = True     # show s2 mirrored L-R (the two pads face each other)

# ---------------------------------------------------- expected patch ------
# A measured 7x4 on its own cannot say whether a round blob is round because
# the CONTACT is round or because the SENSOR made it round. So draw, beside
# it, the 7x4 a perfect sensor would report for this same pad pose on this
# same object — blob_axis.expected_patch_map, the identical function the blob
# report uses, so the two cannot disagree.
SHOW_EXPECTED_PANEL = True

# Band width. blob_axis.BAND_WIDTH_MM still defaults to 8.0, which is right
# for a thin rod and wrong for a fat one, and passing it silently is how a
# whole series ends up incomparable. So derive it from the object instead:
# a rod of diameter D indented by INDENT_MM contacts over a band of
#     2 * sqrt(INDENT * (D - INDENT))
# which gives 15.05 mm at D=26 -- matching the 15.0 mm fitted from a real
# upright run -- and 8.5 mm at D=10. The number used is printed on the panel.
INDENT_MM = 2.4
BAND_WIDTH_MM = None      # None = derive from the rod's diameter (above)

# The pad's OUTER TIP -- the free end of the finger, furthest from the palm.
# Marked on every panel because a contact dying at that edge is geometry (the
# sensor ran out of pad), not a sensor fault, and the two look identical in a
# bare 7x4. Row 0 of the array is the physical BOTTOM of the pad and displays
# use origin="lower", so the tip is the bottom edge of each heatmap.
SHOW_PAD_TIP = True
TIP_COLOR = "#7b2fbe"


def _band_width_mm(scene):
    """Contact-band width in mm for this object, or the blob_axis default."""
    if BAND_WIDTH_MM is not None:
        return float(BAND_WIDTH_MM), "set"
    try:
        d = float(scene["d"])
        if d > 2 * INDENT_MM:
            return (2.0 * float(np.sqrt(INDENT_MM * (d - INDENT_MM))),
                    f"from \u00d8{d:.0f}")
    except Exception:
        pass
    try:
        import blob_axis as BA
        return float(BA.BAND_WIDTH_MM), "blob_axis default"
    except Exception:
        return 8.0, "fallback"


def _tip_edge_local(oy, oz, basis):
    """The pad's tip edge, computed HERE.

    This used to call stitching._draw_pad_tip with arrow_mm/lw and unpack a
    return value, which meant an older stitching.py raised TypeError and the
    tip vanished from the pose panel while every other panel still had it —
    a version mismatch that looked exactly like a missing feature. Nothing in
    this calculation needs the stitcher, so it no longer asks: PAD_W, PAD_H
    and the measured basis are all it takes.

    Returns ((Y0, Y1), (Z0, Z1), (uy, uz)) — the two corners of the short edge
    furthest from the palm, and the outward direction.
    """
    try:
        import stitching as ST
        PW, PH = float(ST.PAD_W), float(ST.PAD_H)
        flat = ST.is_flat(basis)
        a, u = (ST.FLAT_BASIS if flat else
                (np.asarray(basis[0], float), np.asarray(basis[1], float)))
    except Exception:                       # last-ditch: upright 22 x 37
        PW, PH = 22.0, 37.0
        a, u = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    hw, hh = PW / 2.0, PH / 2.0
    Y = (oy - hw * a[0] - hh * u[0], oy + hw * a[0] - hh * u[0])
    Z = (oz - hw * a[1] - hh * u[1], oz + hw * a[1] - hh * u[1])
    return Y, Z, (-u[0], -u[1])


def _mark_tip(ax):
    """Thick bar along the bottom edge of a 7x4 heatmap = the finger's tip."""
    if not SHOW_PAD_TIP:
        return
    ax.plot([-0.5, N_COLS_DISP - 0.5], [-0.5, -0.5], "-", color=TIP_COLOR,
            lw=5.0, solid_capstyle="butt", clip_on=False, zorder=6)
    ax.text(N_COLS_DISP / 2.0 - 0.5, -0.95, "pad tip", ha="center", va="top",
            fontsize=6.5, color=TIP_COLOR)


N_COLS_DISP = 4

# ---------------------------------------------------------- pose panel ----
# A heatmap on its own does not say WHERE on the object it came from, which
# on a 35-point grid means cross-referencing pose_history by hand for every
# figure. The third column answers it directly: the same front (Y-Z) view as
# the GUI preview, with the rod, every pad pose in the run, and THIS grasp's
# pad highlighted — plus the robot's visit number, which is not the same as
# the number in the filename as soon as a point gets skipped.
SHOW_POSE_PANEL = True


def _load_run_layout(run_dir):
    """Everything the pose panel needs, read once per run.

    Returns None when the run has no pose_history or stitching is not
    importable, in which case the figure falls back to the original two
    panels and nothing else changes.

      order   : [tag, ...] in the order the ROBOT actually visited them
      offs    : {tag: (y_mm, z_mm)}    measured pad centre (EE+FK)
      bases   : {tag: basis}           measured pad roll
      planned : [(y_mm, z_mm), ...]    every point the GUI asked for
      skipped : [(index, (y, z)), ...] planned points never visited
      scene   : rod geometry, or None
    """
    import json
    try:
        import stitching as ST
    except Exception:
        return None
    ph = os.path.join(run_dir, "pose_history.json")
    if not os.path.exists(ph):
        return None
    try:
        with open(ph) as f:
            d = json.load(f)
    except Exception:
        return None

    order = [str(p.get("tag")) for p in d.get("points", []) if p.get("tag")]
    try:
        offs, _src = ST.load_offsets(run_dir)
    except Exception:
        offs = {}
    try:
        bases = ST.load_pad_bases(run_dir)
    except Exception:
        bases = {}
    try:
        scene = ST._load_scene(run_dir, verbose=False)
    except Exception:
        scene = None

    # PLANNED vs VISITED. The config lists every grid point the GUI asked
    # for; pose_history lists the ones the robot actually reached. Anything
    # in the first and not the second was skipped (unreachable, IK failure),
    # which belongs on the panel because it puts a hole in the grid.
    cfg = d.get("config", {})
    c = cfg.get("object", {}).get("center_world_mm", [0.0, 0.0, 0.0])
    planned, skipped = [], []
    for i, p in enumerate(cfg.get("points", [])):
        try:
            y = float(c[1]) + float(p.get("pad_offset_y_mm", 0.0))
            z = float(c[2]) + float(p.get("pad_offset_z_mm", 0.0))
        except Exception:
            continue
        planned.append((y, z))
        if f"pt{i:02d}" not in order:
            skipped.append((i, (y, z)))
    return {"order": order, "offs": offs, "bases": bases, "scene": scene,
            "planned": planned, "skipped": skipped}


def _draw_pose_panel(ax, tag, lay):
    """Front (Y-Z) view: the rod, every grid pose, and THIS grasp's pad.

    Same view and conventions as the GUI preview and stitching's column 3,
    including the ROLLED pad outline via stitching.pad_corners, so nothing
    here can disagree with what the stitcher painted."""
    import stitching as ST
    scene, offs, bases = lay["scene"], lay["offs"], lay["bases"]
    key = tag.split("_")[-1] if "_" in tag else tag        # gui_pt07 -> pt07

    ys, zs = [], []
    if scene:                                   # the rod, as stitching draws it
        th = np.deg2rad(scene["tilt"]) if scene["axis"] == "X" else 0.0
        cth, sth = np.cos(th), np.sin(th)
        yl = np.array([-1, 1, 1, -1, -1]) * scene["d"] / 2.0
        zl = np.array([-1, -1, 1, 1, -1]) * scene["L"] / 2.0
        ry = scene["cy"] + cth * yl - sth * zl
        rz = scene["cz"] + sth * yl + cth * zl
        ax.plot(ry, rz, color="#1f77b4", lw=1.4, zorder=2)
        ax.fill(ry, rz, color="#1f77b4", alpha=0.12, zorder=1)
        ys += list(ry); zs += list(rz)

    for k, (oy, oz) in offs.items():            # the rest of the run, faint
        if k == key:
            continue
        Y, Z = ST.pad_corners(oy, oz, bases.get(k))
        ax.plot(Y, Z, color="0.55", lw=0.5, alpha=0.45, zorder=3)
        ys += list(Y); zs += list(Z)

    for _i, (sy, sz) in lay["skipped"]:         # never reached
        ax.plot([sy], [sz], "x", color="crimson", ms=6, mew=1.6, zorder=6)
        ys.append(sy); zs.append(sz)

    if key in offs:                             # THIS grasp
        oy, oz = offs[key]
        Y, Z = ST.pad_corners(oy, oz, bases.get(key))
        ax.plot(Y, Z, color="k", lw=3.0, alpha=0.8, zorder=7)
        ax.plot(Y, Z, color="#d62728", lw=1.8, zorder=8)
        ax.plot([oy], [oz], "o", color="#d62728", ms=4, zorder=9)
        ys += list(Y); zs += list(Z)

    if ys and zs:
        my = (max(ys) - min(ys)) * 0.08 + 3.0
        mz = (max(zs) - min(zs)) * 0.05 + 3.0
        ax.set_xlim(min(ys) - my, max(ys) + my)
        ax.set_ylim(min(zs) - mz, max(zs) + mz)

    # TIP LAST, and sized to the view: this panel spans a whole 140 mm rod,
    # not the ~90 mm of the stitched map, so a fixed 5 mm arrow is drawn but
    # invisible. Everything here is computed locally — no stitching version
    # can take the marker away again.
    if SHOW_PAD_TIP and key in offs and zs:
        try:
            _span = max(max(zs) - min(zs), max(ys) - min(ys), 1.0)
            oy, oz = offs[key]
            (tY0, tY1), (tZ0, tZ1), (uy, uz) = _tip_edge_local(
                oy, oz, bases.get(key))
            ax.plot([tY0, tY1], [tZ0, tZ1], "-", color=TIP_COLOR, lw=5.5,
                    solid_capstyle="butt", zorder=11)
            mtY, mtZ = (tY0 + tY1) / 2.0, (tZ0 + tZ1) / 2.0
            _ar = 0.09 * _span
            ax.annotate("", xy=(mtY + _ar * uy, mtZ + _ar * uz),
                        xytext=(mtY, mtZ),
                        arrowprops=dict(arrowstyle="-|>", color=TIP_COLOR,
                                        lw=1.8),
                        zorder=11)
            ax.text(mtY + 0.15 * _span * uy, mtZ + 0.15 * _span * uz,
                    "pad tip", color=TIP_COLOR, fontsize=7,
                    ha="center", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=TIP_COLOR, lw=0.5, alpha=0.85))
        except Exception as e:
            print(f"[maps] pad tip not drawn on the pose panel ({e})")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("world Y (mm)", fontsize=8)
    ax.set_ylabel("world Z (mm)", fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)


def _pose_title(tag, lay):
    """('visit 8 of 34', note) — the ROBOT's order, not the filename number.

    The two agree only while nothing is skipped, and catching exactly that
    case is the point of showing it."""
    key = tag.split("_")[-1] if "_" in tag else tag
    order, skipped = lay["order"], lay["skipped"]
    n_plan = len(lay["planned"]) or len(order)
    if key not in order:
        return f"pad pose on rod — {key} (not in pose_history)", ""
    i = order.index(key)
    line = f"pad pose on rod — visit {i+1} of {len(order)}"
    if n_plan and n_plan != len(order):
        line += f" ({n_plan} planned, {len(skipped)} skipped)"
    line += f"   [{key}]"

    note = ""                       # was a NEIGHBOURING planned point skipped?
    try:
        idx = int(key.replace("pt", ""))
        miss = {j for j, _ in skipped}
        gaps = [w for w in ("previous", "next")
                if (idx - 1 if w == "previous" else idx + 1) in miss]
        if gaps:
            note = "grid gap: " + " and ".join(gaps) + " point skipped"
    except Exception:
        pass
    return line, note


def hold_average(csv_path):
    """Return (map_7x4, n_hold_frames, peak_sum) for one tactile CSV.

    Delegates to stitching.hold_average so the heatmap you look at is
    EXACTLY the map that gets stitched — same hold window, same baseline
    handling. The local version below is only a standalone fallback."""
    if _hold_average_shared is not None:
        return _hold_average_shared(csv_path)
    df = _read_csv_tolerant(csv_path)
    pred = [c for c in df.columns if c.startswith("pred_")]
    v = df[pred].to_numpy()
    s = v.sum(1)
    peak = float(s.max()) if len(s) else 0.0
    if peak <= 0:
        return np.zeros((7, 4)), 0, 0.0
    mask = s >= HOLD_FRAC * peak
    return v[mask].mean(0).reshape(7, 4), int(mask.sum()), peak


def plot_run(run_dir):
    """One heatmap figure per grasp (s1 left | s2 right) -> Heatmaps/*.png.
    Returns the list of saved PNG paths (empty if no tactile files)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s1_files = sorted(glob.glob(os.path.join(run_dir, "*_pt*_s1_tactile_maps.csv")))
    if not s1_files:
        return []
    out_dir = os.path.join(run_dir, "Heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # One pass over the run FIRST, so "shared across run" mode has a ceiling
    # before the first figure is drawn. Cheap: the CSVs are already cached.
    run_peak = _run_hold_peak(run_dir) if _run_hold_peak is not None else None
    lay = _load_run_layout(run_dir) if SHOW_POSE_PANEL else None

    made = []
    for s1 in s1_files:
        tag = os.path.basename(s1).split("_s1_")[0]        # e.g. gui_pt00
        s2 = s1.replace("_s1_", "_s2_")

        panels = []                                        # (name, map, n, peak, ok)
        for name, path in [("s1", s1), ("s2", s2)]:
            if os.path.exists(path):
                m, nfr, peak = hold_average(path)
                panels.append((name, m, nfr, peak, True))
            else:
                panels.append((name, np.zeros((7, 4)), 0, 0.0, False))

        panel_max = max(float(p[1].max()) for p in panels)

        # ---- what a PERFECT sensor would report at this same pad pose -----
        exp_map, exp_txt = None, ""
        if SHOW_EXPECTED_PANEL and lay and lay.get("scene"):
            try:
                import stitching as ST
                import blob_axis as BA
                _k = tag.split("_")[-1] if "_" in tag else tag
                if _k in lay["offs"]:
                    _bw, _src = _band_width_mm(lay["scene"])
                    exp_map = BA.expected_patch_map(
                        lay["offs"][_k], lay["bases"].get(_k), lay["scene"],
                        band_width_mm=_bw, cal=ST.CAL["s1"])
                    _mi = BA.blob_axis(exp_map)
                    _me = BA.blob_axis(panels[0][1])
                    exp_txt = (f"band {_bw:.1f} mm ({_src})\n"
                               f"expected axis "
                               + (f"{_mi['angle_from_vertical_deg']:+.1f}\u00b0 "
                                  f"elong {_mi['elongation']:.2f}"
                                  if _mi.get("ok") else "n/a")
                               + "   |   s1 measured "
                               + (f"{_me['angle_from_vertical_deg']:+.1f}\u00b0 "
                                  f"elong {_me['elongation']:.2f}"
                                  if _me.get("ok") else "n/a"))
            except Exception as e:
                print(f"[maps] expected panel unavailable for {tag} ({e})")
                exp_map = None

        if _resolve_vmax is not None:
            vmax, scale_lbl = _resolve_vmax(panel_max, run_peak)
        else:
            vmax, scale_lbl = (panel_max if panel_max > 0 else 1.0), "auto"
        if vmax <= 0:
            vmax = 1.0

        # Columns: s1 | s2 | [expected] | [pose]. The two measured maps keep
        # their original width so nothing already readable gets smaller.
        n_extra = (1 if exp_map is not None else 0) + (1 if lay else 0)
        ncol = 2 + n_extra
        fig = Figure(figsize=(7.6 + 3.8 * n_extra, 4.0))
        FigureCanvasAgg(fig)
        axes = [fig.add_subplot(1, ncol, 1), fig.add_subplot(1, ncol, 2)]
        im = None
        for ax, (name, m, nfr, peak, ok) in zip(axes, panels):
            # s2 faces the opposite way from s1, so show it MIRRORED L-R
            # (facing-pad view) — display only; the stored data is untouched.
            disp = m[:, ::-1] if (name == "s2" and MIRROR_S2) else m
            # im = ax.imshow(disp, cmap="jet", aspect="auto", vmin=0.0, vmax=vmax)
            im = ax.imshow(disp, cmap="jet", aspect="auto", vmin=0.0, vmax=vmax, origin="lower")
            mtag = " [mirrored]" if (name == "s2" and MIRROR_S2) else ""
            # peak sum = whole-map total; max = biggest single taxel. With a
            # fixed or shared colour ceiling the panel can look faint, so both
            # numbers stay in the title and nothing is hidden by the scale.
            ttl = (f"{name}{mtag} — hold-avg of {nfr} frames\n"
                   f"peak sum {peak:.0f}   max {float(m.max()):.0f}"
                   if ok else f"{name} — FILE MISSING")
            ax.set_title(ttl, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            # MIRROR_S2 flips columns, never rows, so the tip stays at the
            # bottom on both panels.
            _mark_tip(ax)
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.set_label(f"pressure (a.u.) — {scale_lbl}", fontsize=8)

        if exp_map is not None:
            axe = fig.add_subplot(1, ncol, 3)
            # OWN colour scale, deliberately: the expected patch is in
            # arbitrary units normalised to 1000, so sharing the measured
            # ceiling would say something false about magnitude. What is
            # comparable between the panels is SHAPE, not brightness.
            axe.imshow(exp_map, cmap="jet", aspect="auto", origin="lower",
                       vmin=0.0, vmax=float(exp_map.max()) or 1.0)
            axe.set_xticks([]); axe.set_yticks([])
            axe.set_title("EXPECTED — perfect sensor, same pose\n"
                          "(shape comparable, brightness NOT)", fontsize=9)
            _mark_tip(axe)
            if exp_txt:
                axe.text(0.5, -0.14, exp_txt, transform=axe.transAxes,
                         ha="center", va="top", fontsize=7, color="#333")

        if lay:
            ax3 = fig.add_subplot(1, ncol, ncol)
            try:
                _draw_pose_panel(ax3, tag, lay)
                head, note = _pose_title(tag, lay)
                # WHOSE pose? Neither s1's nor s2's alone: the position comes
                # from pose_history's FK-derived pad centre, and the two pads
                # differ only in X, which this Y-Z view does not show. So one
                # panel is correct for both.
                ax3.set_title(head + "\n(same for s1 and s2: they differ only "
                                     "in X)", fontsize=8)
                if note:
                    ax3.text(0.5, 0.985, note, transform=ax3.transAxes,
                             ha="center", va="top", fontsize=7.5,
                             color="crimson", zorder=10,
                             bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                       ec="crimson", lw=0.6, alpha=0.9))
            except Exception as e:                # never lose the heatmaps
                ax3.axis("off")
                ax3.set_title(f"(pose panel unavailable: {e})", fontsize=7)
        fig.suptitle(f"Grasp {tag} — hold-average heatmaps (s1 | s2)",
                     fontsize=11, y=1.04)

        png = os.path.join(out_dir, f"heatmap_{tag}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        made.append(png)
        print(f"saved {png}")

    # the 28 numbers behind every figure above, in one CSV
    try:
        from stitching import save_hold_averages
        save_hold_averages(run_dir)
    except Exception as e:
        print(f"[maps] could not save hold_average_maps.csv ({e})")
    return made


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Paper3_Simulation/Data/gui_run")
    if not plot_run(rd):
        print(f"no *_s1_tactile_maps.csv found in {rd}")
