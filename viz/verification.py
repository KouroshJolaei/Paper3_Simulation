"""
verification.py — Paper-2-style desired-vs-actual motion plot for ONE move,
showing the SENSOR PAD FRAME (axes + pad rectangle) and the FULL path.

Reads move_p1_to_p2.json saved by collect_session.py and makes a figure:
  - LEFT (3D): the FULL actual pad path (every sim step), plus the pad drawn as
    a coordinate frame (3 axes) AND a rectangle at:
        START (green), DESIRED end (black), ACTUAL end (red).
    If the arm detours to home, the full path line will show it.
  - RIGHT-TOP: per-axis desired vs actual displacement (mm).
  - RIGHT-BOTTOM: banner with desired move, actual move, per-axis error,
    pad ORIENTATION change (deg, ~0 for a parallel move), and path length vs
    straight-line (to expose any detour).

Run in normal Python (PyCharm):
  python3 verification.py move_p1_to_p2.json move_p1_to_p2_verification.png
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PAD_W = 0.022   # pad short side (m)
PAD_H = 0.037   # pad long side (m)


def _draw_frame(ax, p, R, scale=0.03, label=None):
    cols = ["red", "green", "blue"]
    for i in range(3):
        d = R[:, i] * scale
        ax.plot([p[0], p[0]+d[0]], [p[1], p[1]+d[1]], [p[2], p[2]+d[2]],
                color=cols[i], linewidth=2)
    if label:
        ax.text(p[0], p[1], p[2]+scale*0.4, label, fontsize=9)


def _draw_pad_rect(ax, p, R, color):
    u = R[:, 1]; v = R[:, 2]
    corners = [
        p + ( PAD_W/2)*u + ( PAD_H/2)*v,
        p + (-PAD_W/2)*u + ( PAD_H/2)*v,
        p + (-PAD_W/2)*u + (-PAD_H/2)*v,
        p + ( PAD_W/2)*u + (-PAD_H/2)*v,
        p + ( PAD_W/2)*u + ( PAD_H/2)*v,
    ]
    c = np.array(corners)
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, linewidth=1.8)


def _orient_change_deg(R0, R1):
    dR = R0.T @ R1
    c = (np.trace(dR) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def plot_move(json_path, out_png=None, show=True):
    with open(json_path) as f:
        rec = json.load(f)

    p_start = np.array(rec["pad_start_pos"]);        R_start = np.array(rec["pad_start_R"])
    p_des   = np.array(rec["pad_end_desired_pos"]);  R_des   = np.array(rec["pad_end_desired_R"])
    p_act   = np.array(rec["pad_end_actual_pos"]);   R_act   = np.array(rec["pad_end_actual_R"])
    path    = np.array(rec.get("pad_full_path", [p_start.tolist(), p_act.tolist()]))

    des_disp = (p_des - p_start) * 1000.0
    act_disp = (p_act - p_start) * 1000.0
    err      = (p_act - p_des) * 1000.0
    dtheta   = _orient_change_deg(R_start, R_act)

    fig = plt.figure(figsize=(13, 6.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-", color="tab:blue",
                linewidth=1.5, label="actual pad path (every step)")
    _draw_frame(ax, p_start, R_start, label="START"); _draw_pad_rect(ax, p_start, R_start, "green")
    _draw_frame(ax, p_des,   R_des,   label="DESIRED"); _draw_pad_rect(ax, p_des, R_des, "black")
    _draw_frame(ax, p_act,   R_act,   label="ACTUAL"); _draw_pad_rect(ax, p_act, R_act, "red")
    ax.scatter(*p_start, color="green", s=60)
    ax.scatter(*p_des,   color="black", s=60, marker="X")
    ax.scatter(*p_act,   color="red",   s=60)
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)"); ax.set_zlabel("world Z (m)")
    ax.set_title("Pad path + frames\ngreen=start, black=desired, red=actual")
    ax.legend(loc="upper left", fontsize=8)

    allpts = np.vstack([path, p_start, p_des, p_act])
    ctr = allpts.mean(0)
    span = max((allpts.max(0) - allpts.min(0)).max(), 0.05)
    for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
        setlim(c - span/2, c + span/2)

    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(3); w = 0.35
    ax2.bar(x - w/2, des_disp, w, label="desired", color="0.6")
    ax2.bar(x + w/2, act_disp, w, label="actual",  color="tab:blue")
    ax2.set_xticks(x); ax2.set_xticklabels(["X", "Y", "Z"])
    ax2.set_ylabel("pad displacement (mm)")
    ax2.set_title("Desired vs actual pad displacement")
    ax2.axhline(0, color="k", linewidth=0.8); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 4); ax3.axis("off")
    total_err = float(np.linalg.norm(err))
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) * 1000 if len(path) > 1 else 0.0
    straight = float(np.linalg.norm(p_act - p_start)) * 1000
    txt = (
        "--- DESIRED PAD MOVE ---\n"
        f"dX={des_disp[0]:+6.2f}  dY={des_disp[1]:+6.2f}  dZ={des_disp[2]:+6.2f} mm\n\n"
        "--- ACTUAL PAD MOVE ---\n"
        f"dX={act_disp[0]:+6.2f}  dY={act_disp[1]:+6.2f}  dZ={act_disp[2]:+6.2f} mm\n\n"
        "--- ERROR (actual - desired) ---\n"
        f"X={err[0]:+6.3f}  Y={err[1]:+6.3f}  Z={err[2]:+6.3f} mm\n"
        f"TOTAL = {total_err:6.3f} mm\n\n"
        f"pad rotation during move = {dtheta:5.2f} deg  (should be ~0)\n"
        f"path length = {path_len:6.1f} mm | straight = {straight:6.1f} mm\n"
        f"(if path >> straight, the arm DETOURED)"
    )
    ax3.text(0.02, 0.98, txt, va="top", ha="left", family="monospace",
             fontsize=9.5, bbox=dict(boxstyle="round", facecolor="white", edgecolor="red"))

    fig.suptitle("Individual Move Verification: P1 -> P2 (pad frame + full path)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_png:
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"saved {out_png}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    jp = sys.argv[1] if len(sys.argv) > 1 else "move_p1_to_p2.json"
    op = sys.argv[2] if len(sys.argv) > 2 else "move_p1_to_p2_verification.png"
    plot_move(jp, op, show=True)
