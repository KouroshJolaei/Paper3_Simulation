"""
viz/pad_truth.py — SAFETY REPORT.
Per grasp: compares the pad pose the GUI DESIGNED vs the pad pose actually
reached in Isaac, and flags any mismatch in RED.

  TRUE mode   : uses 'pad_measured_pos_m' (physics read) if present -> real check.
  FK-only mode: falls back to 'pad_actual_pos_m' (FK + fixed const, BLIND to the
                finger swing) -> shown greyed, labelled, NOT a trustworthy check.

Pure PyCharm python; standalone Agg figure (never touches the GUI backend).
Run from GUI ("Check Pad Truth") or standalone:
    python3 viz/pad_truth.py <run_dir>
"""
import os, json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

THRESH_MM = 3.0   # mismatch above this = RED (adjust to taste)


def _target(p):
    """The pad pose the GUI designed."""
    for k in ("pad_gui_target_m", "pad_desired_pos_m"):
        if k in p:
            return np.array(p[k], float)
    return None


def _final(p):
    """The pad pose actually reached. Prefer the physics measurement; fall back
    to the FK-only estimate (which is blind to the finger swing)."""
    if "pad_measured_pos_m" in p:
        return np.array(p["pad_measured_pos_m"], float), "TRUE (measured)"
    if "pad_actual_pos_m" in p:
        return np.array(p["pad_actual_pos_m"], float), "FK-only (blind to swing)"
    return None, "none"


def check_run(run_dir):
    """Returns (png_path, any_bad, rows). rows = [(tag, mode, d_mm[3] or None, dist or None)]."""
    ph = os.path.join(run_dir, "pose_history.json")
    if not os.path.exists(ph):
        return None, False, []
    data = json.load(open(ph))
    rows, any_bad, any_true = [], False, False
    for p in data.get("points", []):
        tgt = _target(p)
        fin, mode = _final(p)
        if tgt is None or fin is None:
            rows.append((p.get("tag", "?"), mode, None, None))
            continue
        d_mm = (fin - tgt) * 1000.0
        dist = float(np.linalg.norm(d_mm))
        is_true = mode.startswith("TRUE")
        any_true |= is_true
        bad = is_true and dist > THRESH_MM
        any_bad |= bad
        rows.append((p.get("tag", "?"), mode, d_mm, dist))
    return _draw(run_dir, rows, any_bad, any_true), any_bad, rows


def _draw(run_dir, rows, any_bad, any_true):
    fig = Figure(figsize=(8.8, 0.42 * max(len(rows), 1) + 1.9))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    if not any_true:
        banner = ("FK-only run — physics measurement NOT yet recorded "
                  "(not a true safety check)")
        color = "#b58900"
    elif any_bad:
        banner = (f"MISMATCH > {THRESH_MM:.0f} mm — pads are NOT where the "
                  "GUI designed them")
        color = "#c00"
    else:
        banner = f"OK — all pads within {THRESH_MM:.0f} mm of GUI design"
        color = "#0a6"

    ax.text(0, 1.02, "PAD-TRUTH SAFETY REPORT", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0, 0.95, banner, fontsize=10, color=color, transform=ax.transAxes)
    ax.text(0, 0.86,
            f"{'grasp':6} {'mode':26} {'dY':>7}{'dZ':>7}{'dX':>7}  {'|d|mm':>7}",
            family="monospace", fontsize=9, transform=ax.transAxes)

    y = 0.80
    for tag, mode, d_mm, dist in rows:
        if d_mm is None:
            line = f"{tag:6} {mode:26} {'--':>7}{'--':>7}{'--':>7}  {'--':>7}"
            col = "#888"
        else:
            line = (f"{tag:6} {mode:26} "
                    f"{d_mm[1]:7.1f}{d_mm[2]:7.1f}{d_mm[0]:7.1f}  {dist:7.1f}")
            col = ("#c00" if (mode.startswith("TRUE") and dist > THRESH_MM)
                   else "#0a6" if mode.startswith("TRUE") else "#888")
        ax.text(0, y, line, family="monospace", fontsize=9, color=col,
                transform=ax.transAxes)
        y -= 0.055

    out = os.path.join(run_dir, "pad_truth_report.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    png, bad, rows = check_run(run)
    for r in rows:
        print(r)
    print("REPORT:", png, " ANY_BAD:", bad)
