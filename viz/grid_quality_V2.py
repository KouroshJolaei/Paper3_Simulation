"""grid_quality.py — what actually happened at every point of a grid sweep.

WHY THIS EXISTS
---------------
run_20260901_154153 (30x120x140 cuboid, 117 points) reported 117/117
executed, 0 dead, exec_stage complete on every point. It looked perfect.

It was not. The entire bottom row of the grid, z = +21.69 mm, closed to
0.47434 rad instead of the commanded 0.4931 and reached only 0.53 mm of
indentation against 1.23-1.41 mm everywhere else. Thirteen of 117 points,
40% of the intended squeeze, silently pooled with the good ones.

Nothing caught it because the ledger's own quality question is
"contact_target_reached", which in FIXED-ANGLE mode is null by construction.
So the ledger asked a question that could not fail, and the answer that
mattered -- did the fingers actually GET to the angle we asked for -- was
sitting in close_rad_stopped, unread.

THE ONE NUMBER THAT MATTERS
    close_rad_ceiling - close_rad_stopped
On a healthy grasp the fingers ramp to the ceiling and stall a hair short as
the pads compress: the eight good rows above sat at 2.06-2.10 mrad short,
with a spread of 0.06 mrad between them. The blocked row sat 16.8 mrad
short -- EIGHT TIMES further, and outside the good rows' spread by a factor
of 250. This is not a marginal call, which is exactly why it is worth
automating: it is obvious once plotted and invisible in a 400-line log.

Something physically stopped those fingers. The likely cause is the object's
top end jamming in the finger throat, which design_grid is supposed to
prevent -- but its throat check models EVERY object as a cylinder of the
GRASPED width (main_gui._rod_cloud), so a 30 x 120 mm box end was checked as
a 30 mm round rod and its 120 mm shoulders were invisible. That is a
separate fix. This file is the detector, and it is shape-agnostic: it will
catch the same failure on a sphere, on a mesh, or from a cause nobody has
thought of yet, because it measures the outcome rather than predicting it.

WHAT IT REPORTS
    per point   deformation at hold, close-angle shortfall, tactile peak
                sum, and (optionally) the blob axis measured vs expected
    per row     the same, aggregated along Z -- where a throat jam shows up
    per column  the same, aggregated along Y -- where an across-limit or a
                reach problem shows up
    verdict     ok / WEAK / BLOCKED / DEAD, with the points named

Aggregating by row and by column is the point. A fault that is invisible in
a 117-line list ("some points read 0.5, some read 1.4") announces itself the
moment you group by grid position and one whole row is different.

USAGE
    import grid_quality as GQ
    txt, pngs = GQ.grid_quality_and_save(run_dir)
or
    python3 grid_quality.py <run_dir>
"""

import os
import sys
import glob
import json
import numpy as np

# ---- thresholds ----------------------------------------------------------
# BLOCKED. Compared against the run's OWN median stopped angle rather than a
# hard-coded number, because the stall point depends on diameter, shape and
# pad compression. On the reference run the good points scatter by 0.06 mrad
# about their median and the bad row is 16.8 mrad away, so anything from
# ~1 to ~15 mrad separates them; 5 mrad sits in the middle of that window.
# In FIXED-ANGLE mode this is the only evidence that a close was obstructed.
BLOCK_SHORTFALL_RAD = 0.005

# WEAK. A fraction of the run's own median indentation, for the same reason.
# 0.70 flags the reference run's bad row (0.53 / 1.30 = 0.41) with room to
# spare while leaving its healthy 1.08-1.70 spread alone.
WEAK_DEFORM_FRAC = 0.70

# DEAD. Matches collect_from_config's DEAD_GRASP_FLOOR so the two agree.
DEAD_DEFORM_M = 0.0002
FRAME_RATIO_MAX = 2.0


def _load(run_dir):
    """Ledger + config for one run. Either may be missing; say which."""
    out = {}
    for key, name in (("ledger", "execution_ledger.json"),
                      ("config", "gui_config_used.json")):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    out[key] = json.load(f)
            except Exception as e:
                out[key] = None
                out.setdefault("errors", []).append(f"{name}: {e}")
        else:
            out[key] = None
            out.setdefault("errors", []).append(f"{name} not found")
    return out


def _peak_sums(run_dir, sensor="s1"):
    """Tactile peak sum and max per grasp, or {} if stitching is unavailable.

    Optional on purpose: the report must still work on a run whose CSVs have
    been moved or whose viz/ is not importable, because a partial report is
    worth far more than an exception."""
    try:
        import stitching as ST
    except Exception:
        return {}
    out = {}
    for f in sorted(glob.glob(os.path.join(
            run_dir, f"*_pt*_{sensor}_tactile_maps.csv"))):
        try:
            tag = ST._pt_key(os.path.basename(f))
            if tag is None:
                continue
            # m, _n, _p = ST.hold_average(f)
            # out[tag] = (float(np.nansum(m)), float(np.nanmax(m)))

            m, nfr, _p = ST.hold_average(f)
            out[tag] = (float(np.nansum(m)), float(np.nanmax(m)), int(nfr))
        except Exception:
            continue
    return out


def collect(run_dir, sensor="s1", want_blob=False, band_width_mm=8.0):
    """One row per grid point, joining the ledger to the grid geometry.

    Returns (rows, meta). Every row carries what was COMMANDED (y, z) and
    what was OBSERVED (angle, deformation, peak), so any disagreement is
    visible per point rather than only in aggregate."""
    d = _load(run_dir)
    led, cfg = d.get("ledger"), d.get("config")
    if led is None:
        return [], {"errors": d.get("errors", []),
                    "fatal": "no execution_ledger.json — cannot report"}

    geo = {}
    if cfg is not None:
        for q in cfg.get("points", []):
            geo[int(q["index"])] = (float(q.get("pad_offset_y_mm", np.nan)),
                                    float(q.get("pad_offset_z_mm", np.nan)))
    # peaks = _peak_sums(run_dir, sensor)

    peaks = _peak_sums(run_dir, sensor)
    peaks_o = _peak_sums(run_dir, "s2" if sensor == "s1" else "s1")



    blob = {}
    if want_blob:
        try:
            import blob_axis as BA
            brows, _bmeta = BA.measure_run(run_dir, sensor, band_width_mm)
            for r in brows:
                blob[r["grasp"]] = (r.get("measured_deg"),
                                    r.get("expected_deg"))
        except Exception:
            blob = {}

    rows = []
    for p in led.get("points", []):
        c = p.get("contact") or {}
        tag = p.get("tag") or f"pt{int(p.get('index', 0)):02d}"
        y, z = geo.get(int(p.get("index", -1)), (np.nan, np.nan))
        ceil = c.get("close_rad_ceiling")
        stop = c.get("close_rad_stopped")
        short = (None if (ceil is None or stop is None)
                 else float(ceil) - float(stop))
        # ps, pm = peaks.get(tag, (None, None))

        ps, pm, nf = peaks.get(tag, (None, None, None))
        _o1, _o2, nfo = peaks_o.get(tag, (None, None, None))

        mb, eb = blob.get(tag, (None, None))
        rows.append({
            "index": int(p.get("index", -1)), "tag": tag, "y": y, "z": z,
            "executed": bool(p.get("executed", False)),
            "stage": p.get("exec_stage"), "outcome": p.get("outcome"),
            "reason_code": p.get("reason_code"),
            "ceiling": ceil, "stopped": stop, "shortfall": short,
            "deform": c.get("deformation_raw_at_hold"),
            "deform_s1": c.get("deformation_raw_s1"),
            "deform_s2": c.get("deformation_raw_s2"),
            "target": c.get("contact_target"),
            "target_reached": c.get("contact_target_reached"),
            "dead": bool(c.get("dead_grasp", False)),
            # "peak_sum": ps, "peak_max": pm,

            "peak_sum": ps, "peak_max": pm,
            "hold_frames": nf, "hold_frames_other": nfo,

            "blob_meas": mb, "blob_exp": eb})

    cl = led.get("closing", {}) or {}
    meta = {"errors": d.get("errors", []), "sensor": sensor,
            "generated": led.get("generated"),
            "closing_mode": cl.get("mode"), "closing_signal": cl.get("signal"),
            "closing_target": cl.get("target"),
            "n_points": len(rows),
            "object": (cfg or {}).get("object", {}),
            "grid": (cfg or {}).get("grid", {}),
            "have_peaks": bool(peaks), "have_blob": bool(blob)}
    return rows, meta


def verdicts(rows):
    """Classify every point against the run's own medians. Returns (rows, ref).

    Self-scaling by design: a Ø18 cylinder and a 40 mm cuboid stall at
    completely different angles, so a fixed threshold would either miss
    faults on one or condemn healthy grasps on the other."""
    st = [r["stopped"] for r in rows if r["stopped"] is not None]
    df = [r["deform"] for r in rows if r["deform"] is not None]
    med_stop = float(np.median(st)) if st else None
    med_def = float(np.median(df)) if df else None
    for r in rows:
        tags = []
        if not r["executed"] or r["stage"] != "complete":
            tags.append("NOT-RUN")
        if r["dead"] or (r["deform"] is not None
                         and r["deform"] < DEAD_DEFORM_M):
            tags.append("DEAD")
        if (med_stop is not None and r["stopped"] is not None
                and r["stopped"] < med_stop - BLOCK_SHORTFALL_RAD):
            tags.append("BLOCKED")
        if (med_def is not None and r["deform"] is not None
                and r["deform"] < WEAK_DEFORM_FRAC * med_def):
            tags.append("WEAK")
        # if (r["target"] is not None and r["target_reached"] is False):
        #     tags.append("TARGET-MISSED")

        nf, nfo = r.get("hold_frames"), r.get("hold_frames_other")
        if nf and nfo and max(nf, nfo) > FRAME_RATIO_MAX * min(nf, nfo):
            tags.append(f"FRAMES {nf}/{nfo}")
        if (r["target"] is not None and r["target_reached"] is False):
            tags.append("TARGET-MISSED")

        r["verdict"] = "ok" if not tags else "/".join(tags)
    return rows, {"median_stopped_rad": med_stop, "median_deform_m": med_def}


def _group(rows, key, field, scale=1.0):
    """mean/min/max of `field` grouped by grid coordinate `key`."""
    g = {}
    for r in rows:
        v = r.get(field)
        if v is None or not np.isfinite(r.get(key, np.nan)):
            continue
        g.setdefault(round(float(r[key]), 2), []).append(float(v) * scale)
    return {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v))
            for k, v in sorted(g.items())}


def grid_quality_report(run_dir, sensor="s1", want_blob=False,
                        band_width_mm=8.0):
    """The whole story as text. Prints WHAT IT MEASURED, not just a verdict."""
    rows, meta = collect(run_dir, sensor, want_blob, band_width_mm)
    if not rows:
        return ("GRID QUALITY — cannot report\n  "
                + "\n  ".join(meta.get("errors", []))
                + ("\n  " + meta["fatal"] if meta.get("fatal") else ""))
    rows, ref = verdicts(rows)
    o = meta["object"] or {}

    L = ["GRID QUALITY — what happened at every point of the sweep",
         f"run    : {os.path.abspath(run_dir)}",
         f"sensor : {meta['sensor']}   points: {meta['n_points']}   "
         f"ledger written {meta.get('generated')}"]
    if o:
        L.append(f"object : {o.get('shape', '?')}  grip "
                 f"{o.get('diameter_mm', '?')} x across "
                 f"{o.get('across_mm', o.get('diameter_mm', '?'))} x along "
                 f"{o.get('length_mm', '?')} mm")
    L.append(f"closing: mode={meta['closing_mode']}  "
             f"signal={meta['closing_signal']}  target={meta['closing_target']}")
    if meta["closing_mode"] == "fixed_angle":
        L.append("         NOTE fixed-angle: contact_target_reached is null by")
        L.append("         construction, so the ledger's own pass/fail cannot")
        L.append("         fail. The close-angle shortfall below is the only")
        L.append("         evidence that a close was obstructed.")
    if ref["median_stopped_rad"] is not None:
        L.append(f"medians: stopped angle {ref['median_stopped_rad']:.5f} rad, "
                 f"indentation {1000*ref['median_deform_m']:.2f} mm "
                 f"(all thresholds are relative to these)")

    # ---- the two aggregations that make a positional fault obvious --------
    for key, label in (("z", "Z row (ALONG the pad)"),
                       ("y", "Y column (ACROSS the pad)")):
        dg = _group(rows, key, "deform", 1000.0)
        sg = _group(rows, key, "shortfall", 1000.0)
        pg = _group(rows, key, "peak_sum")
        if not dg:
            continue
        L.append("")
        L.append(f"  BY {label}")
        L.append("     coord     n   indentation mm            close shortfall"
                 " mrad     peak sum")
        L.append("                    mean   min   max          mean    max"
                 "          mean")
        for k in dg:
            dm, dlo, dhi, n = dg[k]
            sm, _slo, shi, _ = sg.get(k, (np.nan, np.nan, np.nan, 0))
            pm = pg.get(k, (np.nan,))[0]
            flag = ""
            if ref["median_deform_m"] and dm < 1000 * WEAK_DEFORM_FRAC * ref["median_deform_m"]:
                flag = "   <-- WEAK ROW" if key == "z" else "   <-- WEAK COLUMN"
            L.append(f"    {k:+7.2f}  {n:3d}   {dm:5.2f} {dlo:5.2f} {dhi:5.2f}"
                     f"        {sm:7.2f} {shi:7.2f}"
                     f"      {pm:9.0f}" + flag)

    # ---- the points that need a decision ---------------------------------
    bad = [r for r in rows if r["verdict"] != "ok"]
    L.append("")
    if not bad:
        L.append("  VERDICT: all points ok — nothing is more than "
                 f"{1000*BLOCK_SHORTFALL_RAD:.0f} mrad short of the median "
                 f"close angle and nothing is below "
                 f"{100*WEAK_DEFORM_FRAC:.0f}% of the median indentation.")
    else:
        L.append(f"  VERDICT: {len(bad)} of {len(rows)} points need a decision")
        L.append("     grasp      y      z   indent   shortfall   peak sum"
                 "   verdict")
        for r in sorted(bad, key=lambda r: r["index"]):
            dm = "  --" if r["deform"] is None else f"{1000*r['deform']:5.2f}"
            sf = "    --" if r["shortfall"] is None else f"{1000*r['shortfall']:7.2f}"
            ps = "       --" if r["peak_sum"] is None else f"{r['peak_sum']:9.0f}"
            L.append(f"     {r['tag']}  {r['y']:+6.1f} {r['z']:+6.2f}"
                     f"   {dm}    {sf}   {ps}   {r['verdict']}")
        L.append("")
        L.append("  These are DATA, not errors: they executed and wrote CSVs.")
        L.append("  Decide explicitly whether they enter the training set. A")
        L.append("  BLOCKED point was squeezed less than every other point in")
        L.append("  the run, so its map is dimmer for a reason that has")
        L.append("  nothing to do with the object.")

    if meta["have_blob"]:
        errs = [r["blob_meas"] - r["blob_exp"] for r in rows
                if r["blob_meas"] is not None and r["blob_exp"] is not None]
        if errs:
            a = np.array([(e + 90) % 180 - 90 for e in errs])
            L.append("")
            L.append(f"  BLOB AXIS: {len(a)} points with both measured and "
                     f"expected — mean error {a.mean():+.2f} deg, "
                     f"std {a.std():.2f}, |max| {np.abs(a).max():.2f}")
    elif want_blob:
        L.append("")
        L.append("  BLOB AXIS: requested but unavailable (blob_axis.py not "
                 "importable from this run).")

    L.append("")
    L.append("shortfall = close_rad_ceiling - close_rad_stopped. Small and")
    L.append("            uniform is healthy: the fingers reached the angle")
    L.append("            and stalled on pad compression. A point far above")
    L.append("            the rest was STOPPED BY SOMETHING before it got")
    L.append("            there, and its indentation will be low to match.")
    if meta.get("errors"):
        L.append("")
        L.append("notes: " + "; ".join(meta["errors"]))
    return "\n".join(L)


def grid_quality_figure(run_dir, out_dir=None, sensor="s1", want_blob=False,
                        band_width_mm=8.0):
    """Six panels: three maps over the grid, three profiles along it."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rows, meta = collect(run_dir, sensor, want_blob, band_width_mm)
    if not rows:
        return []
    rows, ref = verdicts(rows)
    out_dir = out_dir or os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)

    y = np.array([r["y"] for r in rows], float)
    z = np.array([r["z"] for r in rows], float)
    ok = np.isfinite(y) & np.isfinite(z)
    if not ok.any():
        return []

    fig = Figure(figsize=(16.5, 9.0))
    FigureCanvasAgg(fig)

    def _map(ax, vals, title, cbar_label, cmap="viridis"):
        v = np.array([np.nan if x is None else float(x) for x in vals], float)
        m = ok & np.isfinite(v)
        if not m.any():
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            return
        sc = ax.scatter(y[m], z[m], c=v[m], s=110, marker="s", cmap=cmap,
                        edgecolors="k", linewidths=0.3)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label, fontsize=8)
        ax.set_xlabel("pad offset Y (mm)", fontsize=8)
        ax.set_ylabel("pad offset Z (mm)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    _map(fig.add_subplot(2, 3, 1),
         [None if r["deform"] is None else 1000 * r["deform"] for r in rows],
         "indentation at hold", "mm", "viridis")
    _map(fig.add_subplot(2, 3, 2),
         [None if r["shortfall"] is None else 1000 * r["shortfall"]
          for r in rows],
         "close-angle SHORTFALL (ceiling - stopped)\nhigh = the fingers were "
         "stopped early", "mrad", "inferno_r")
    _map(fig.add_subplot(2, 3, 3), [r["peak_sum"] for r in rows],
         f"tactile peak sum [{sensor}]", "a.u.", "viridis")

    # verdict map — categorical, so colour carries meaning not magnitude
    ax = fig.add_subplot(2, 3, 4)
    palette = {"ok": "#2c9c4a"}
    for i, r in enumerate(rows):
        if not ok[i]:
            continue
        v = r["verdict"]
        c = palette.get(v, "#c0392b")
        ax.scatter(y[i], z[i], c=c, s=110, marker="s",
                   edgecolors="k", linewidths=0.3)
    ax.set_xlabel("pad offset Y (mm)", fontsize=8)
    ax.set_ylabel("pad offset Z (mm)", fontsize=8)
    nbad = sum(1 for r in rows if r["verdict"] != "ok")
    ax.set_title(f"verdict — green ok, red needs a decision "
                 f"({nbad}/{len(rows)} red)", fontsize=9)
    ax.tick_params(labelsize=7)

    # profiles: the aggregation that makes a positional fault jump out
    def _profile(ax, key, xlabel):
        dg = _group(rows, key, "deform", 1000.0)
        if not dg:
            return
        ks = list(dg)
        mean = [dg[k][0] for k in ks]
        lo = [dg[k][0] - dg[k][1] for k in ks]
        hi = [dg[k][2] - dg[k][0] for k in ks]
        ax.errorbar(ks, mean, yerr=[lo, hi], fmt="o-", capsize=3, lw=1.4,
                    color="#1f77b4")
        if ref["median_deform_m"]:
            ax.axhline(1000 * ref["median_deform_m"], ls="--", lw=1.0,
                       color="#888", label="run median")
            ax.axhline(1000 * WEAK_DEFORM_FRAC * ref["median_deform_m"],
                       ls=":", lw=1.2, color="#c0392b",
                       label=f"WEAK below {100*WEAK_DEFORM_FRAC:.0f}%")
            ax.legend(fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("indentation (mm)", fontsize=8)
        ax.set_title(f"indentation by {xlabel}\n(mean, whiskers = min/max)",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    _profile(fig.add_subplot(2, 3, 5), "z", "pad offset Z (mm)")
    _profile(fig.add_subplot(2, 3, 6), "y", "pad offset Y (mm)")

    o = meta["object"] or {}
    fig.suptitle(
        f"GRID QUALITY [{sensor}] — {o.get('shape', '?')} "
        f"{o.get('diameter_mm', '?')} x "
        f"{o.get('across_mm', o.get('diameter_mm', '?'))} x "
        f"{o.get('length_mm', '?')} mm, {len(rows)} points, "
        f"closing={meta['closing_mode']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out_dir, f"grid_quality_{sensor}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    return [p]


def grid_quality_and_save(run_dir, sensor="s1", want_figure=True,
                          want_blob=False, band_width_mm=8.0):
    """Report + figure, saved beside the stitch outputs. Returns (txt, pngs)."""
    txt = grid_quality_report(run_dir, sensor, want_blob, band_width_mm)
    out_dir = os.path.join(run_dir, "Stitched")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"grid_quality_{sensor}.txt"), "w") as f:
        f.write(txt + "\n")
    pngs = (grid_quality_figure(run_dir, out_dir, sensor, want_blob,
                                band_width_mm) if want_figure else [])
    return txt, pngs


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    _blob = "--blob" in sys.argv
    _run = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    for _sen in ("s1", "s2"):
        _t, _p = grid_quality_and_save(_run, _sen, want_blob=_blob)
        print(_t)
        print()
        for _f in _p:
            print("saved " + _f)
