"""
individual_verifications.py — Paper-2-style desired-vs-actual plot per grasp.

Reads pose_history.json (written by collect_from_config.py) and, for EACH grasp,
draws a figure showing the SENSOR PAD:
  - LEFT (3D): pad frame (axes) + pad rectangle at DESIRED (black) vs ACTUAL (red),
    at real 22x37 mm size, in world coordinates.
  - RIGHT-TOP: per-axis desired vs actual pad position (mm).
  - RIGHT-BOTTOM: banner with per-axis error (mm) and total, plus any pad
    orientation drift.

Saves one PNG per grasp into  <run>/Individual_Verifications/.

Run in normal Python (PyCharm):
  python3 individual_verifications.py <run_dir>
or import plot_all(run_dir).
"""

import os, sys, json
import numpy as np
import matplotlib
# NOTE: do NOT call matplotlib.use() here — this module is imported by the GUI,
# and forcing a backend would break the GUI's ability to show windows. We save
# figures via fig.savefig(), which works on any backend.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa

PAD_W = 0.022   # m
PAD_H = 0.037   # m


def _frame(ax, p, R, scale=0.02, label=None):
    cols = ["red", "green", "blue"]
    for i in range(3):
        d = R[:, i] * scale
        ax.plot([p[0], p[0]+d[0]], [p[1], p[1]+d[1]], [p[2], p[2]+d[2]],
                color=cols[i], linewidth=2)
    if label:
        ax.text(p[0], p[1], p[2]+scale*0.5, label, fontsize=9)


def _pad_rect(ax, p, R, color):
    u = R[:, 1]; v = R[:, 2]
    c = np.array([
        p + ( PAD_W/2)*u + ( PAD_H/2)*v,
        p + (-PAD_W/2)*u + ( PAD_H/2)*v,
        p + (-PAD_W/2)*u + (-PAD_H/2)*v,
        p + ( PAD_W/2)*u + (-PAD_H/2)*v,
        p + ( PAD_W/2)*u + ( PAD_H/2)*v,
    ])
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, linewidth=1.8)


def plot_one(entry, out_png):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    tag = entry["tag"]
    p_act = np.array(entry["pad_actual_pos_m"])
    R_act = np.array(entry.get("pad_actual_R", np.eye(3).tolist()))
    p_des = np.array(entry["pad_desired_pos_m"])
    R_des = R_act  # desired orientation = same (no rotation); actual R shows any drift

    des_mm = p_des * 1000.0
    act_mm = p_act * 1000.0
    err_mm = (p_act - p_des) * 1000.0
    total  = float(np.linalg.norm(err_mm))

    # standalone Figure (does NOT touch pyplot's interactive backend)
    fig = Figure(figsize=(11, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _frame(ax, p_des, R_des, label="DESIRED"); _pad_rect(ax, p_des, R_des, "black")
    _frame(ax, p_act, R_act, label="ACTUAL");  _pad_rect(ax, p_act, R_act, "red")
    ax.scatter(*p_des, color="black", s=50, marker="X")
    ax.scatter(*p_act, color="red",   s=50)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"{tag}: pad desired (black) vs actual (red)")
    allp = np.vstack([p_des, p_act])
    ctr = allp.mean(0); span = 0.06
    ax.set_xlim(ctr[0]-span, ctr[0]+span)
    ax.set_ylim(ctr[1]-span, ctr[1]+span)
    ax.set_zlim(ctr[2]-span, ctr[2]+span)

    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(3); w = 0.35
    ax2.bar(x - w/2, des_mm, w, label="desired", color="0.6")
    ax2.bar(x + w/2, act_mm, w, label="actual",  color="tab:blue")
    ax2.set_xticks(x); ax2.set_xticklabels(["X", "Y", "Z"])
    ax2.set_ylabel("pad position (mm)")
    ax2.set_title("Desired vs actual pad position")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 4); ax3.axis("off")
    txt = (
        f"GRASP {tag}\n\n"
        "--- PER-AXIS ERROR (actual - desired) ---\n"
        f"X = {err_mm[0]:+7.3f} mm\n"
        f"Y = {err_mm[1]:+7.3f} mm\n"
        f"Z = {err_mm[2]:+7.3f} mm\n"
        f"TOTAL = {total:7.3f} mm"
    )
    ax3.text(0.02, 0.98, txt, va="top", ha="left", family="monospace",
             fontsize=11, bbox=dict(boxstyle="round", facecolor="white", edgecolor="red"))

    fig.suptitle(f"Individual Verification — {tag}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=120, bbox_inches="tight")


def plot_all(run_dir):
    ph = os.path.join(run_dir, "pose_history.json")
    if not os.path.exists(ph):
        print(f"no pose_history.json in {run_dir}")
        return []
    with open(ph) as f:
        data = json.load(f)
    out_dir = os.path.join(run_dir, "Individual_Verifications")
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for entry in data.get("points", []):
        if "pad_actual_pos_m" not in entry:
            continue
        out_png = os.path.join(out_dir, f"verify_{entry['tag']}.png")
        plot_one(entry, out_png)
        made.append(out_png)
        print(f"saved {out_png}")
    print(f"\n{len(made)} verification plots in {out_dir}")
    return made


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Paper3_Simulation/Data/gui_run")
    plot_all(rd)
