"""run_manifest.py — make a run folder self-contained and self-describing.

WHY THIS EXISTS
---------------
Every raw input is already saved: the tactile CSVs, pose_history.json,
gui_config_used.json, execution_ledger.json. So in principle any figure can
be rebuilt months later. In practice it cannot, for three reasons.

1. THE DERIVED NUMBERS ARE ONLY IN PROSE. grid_accuracy, grid_quality,
   stitching and blob_axis each write a .txt and a .png. To replot the error
   across nine runs you would have to regex the tables back out of the
   reports -- which is exactly what had to be done to compare two closing
   modes, and it is not a thing that should ever need doing twice.

2. HALF THE INPUTS LIVE OUTSIDE THE FOLDER. The calibration entry, the
   finger-throat entry and plot_scale.json are read from Data/ at run time
   and are then free to change. Re-calibrating 30.0|cuboid on 31 August
   silently redefined what every earlier 30 mm cuboid run meant. The entry
   that was ACTUALLY USED has to be copied in, not pointed at.

3. NOTHING RECORDS WHICH CODE PRODUCED IT. Between two runs a week apart the
   stitcher's column axis was reversed, the pad centre moved 35 mm, and the
   throat model changed shape. A run folder that does not say which version
   wrote it cannot be compared with confidence to any other.

This file fixes all three, and it identifies the calibration file BY
MEASUREMENT rather than by trusting an env var: pad_offset_mm in the ledger
is TOOL_OFFSET_Z, and close_rad_ceiling is close_rad, so the entry that
matches both is the entry that was used. On run_20260901_154153 that is
30.0|cuboid in the CONTACT file (TOOL_OFFSET_Z 0.15581 -> 155.81 mm,
close_rad 0.4931), and it rules out the FIXED file, whose close_rad for the
same key is 0.519.

WHAT IT WRITES, into the run folder
    run_manifest.json   everything below, machine-readable
    run_manifest.txt    the same, readable, with a loud MISSING section
    Provenance/         copies of the external entries actually used

USAGE
    import run_manifest as RM
    RM.build(run_dir)
or
    python3 run_manifest.py <run_dir> [--project ~/Paper3_Simulation]
"""

import os
import sys
import glob
import json
import hashlib
import datetime
import shutil

DEFAULT_PROJECT = os.path.expanduser("~/Paper3_Simulation")

# What a complete run contains. required=True means its absence makes the run
# unusable rather than merely incomplete, and the report says so loudly.
#   (label, glob pattern relative to run_dir, required)
EXPECTED = [
    ("run config",            "gui_config_used.json",              True),
    ("execution ledger",      "execution_ledger.json",             True),
    ("pose history",          "pose_history.json",                 True),
    ("tactile maps s1",       "*_pt*_s1_tactile_maps.csv",         True),
    ("tactile maps s2",       "*_pt*_s2_tactile_maps.csv",         True),
    ("grid preview",          "gui_preview.png",                   False),
    ("pad truth probe",       "pad_truth_probe.json",              False),
    ("reachability report",   "reach*report*.json",                False),
    ("mesh state",            "*_mesh_state.csv",                  False),
    ("deformations",          "*_deformations.csv",                False),
    ("heatmaps",              "Heatmaps/heatmap_*.png",            False),
    ("stitched maps",         "Stitched/stitched_*.png",           False),
    ("stitch report",         "Stitched/stitch_report.txt",        False),
    ("blob axis report",      "Stitched/blob_axis_report.txt",     False),
    ("grid accuracy report",  "Stitched/grid_accuracy.txt",        False),
    ("grid quality report",   "Stitched/grid_quality_*.txt",       False),
    ("training pairs",        "**/pair_*.npz",                     False),
]

# Source files whose version changes the meaning of a run.
CODE_FILES = [
    "main_gui.py",
    "sim/collect_from_config.py",
    "viz/stitching.py",
    "viz/heatmaps.py",
    "viz/blob_axis.py",
    "viz/validation.py",
    "viz/grid_accuracy.py",
    "viz/grid_quality.py",
    "examples/probe_finger_throat.py",
]

CAL_FILES = [
    ("main",    "Data/pad_offset_calibration.json"),
    ("fixed",   "Data/pad_offset_calibration_fixed.json"),
    ("contact", "Data/pad_offset_calibration_contact.json"),
]


def _sha(path, limit=None):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
                if limit and h.block_size and f.tell() > limit:
                    break
        return h.hexdigest()[:16]
    except Exception:
        return None


def _stat(path):
    try:
        s = os.stat(path)
        return {"bytes": s.st_size,
                "mtime": datetime.datetime.fromtimestamp(
                    s.st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
    except Exception:
        return {}


def inventory(run_dir):
    """What is in the folder, against what should be."""
    out, missing = [], []
    for label, pat, required in EXPECTED:
        hits = sorted(glob.glob(os.path.join(run_dir, pat), recursive=True))
        item = {"label": label, "pattern": pat, "required": required,
                "count": len(hits)}
        if hits:
            item["total_bytes"] = sum(_stat(h).get("bytes", 0) for h in hits)
            item["newest"] = _stat(max(hits, key=os.path.getmtime))
            if len(hits) == 1:
                item["sha256_16"] = _sha(hits[0])
        else:
            missing.append((label, pat, required))
        out.append(item)
    return out, missing


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def resolve_calibration(run_dir, project):
    """Identify the calibration entry that was ACTUALLY USED, by measurement.

    The ledger records TOOL_OFFSET_Z as collision_model.pad_offset_mm and the
    commanded jaw angle as contact.close_rad_ceiling. An entry matching both
    is the entry the collector read. This beats trusting GRASP_CAL_READ,
    which is not written into the run folder at all, and it beats trusting
    the dropdown, which can be changed between designing and running."""
    led = _load(os.path.join(run_dir, "execution_ledger.json")) or {}
    cfg = _load(os.path.join(run_dir, "gui_config_used.json")) or {}
    obj = (cfg.get("config", cfg)).get("object", {})
    shape = str(obj.get("shape", "cylinder")).lower()
    width = float(obj.get("diameter_mm", 0.0) or 0.0)
    key = f"{width:.1f}" if shape == "cylinder" else f"{width:.1f}|{shape}"

    want_tool = None
    cm = led.get("collision_model") or {}
    if cm.get("pad_offset_mm") is not None:
        want_tool = float(cm["pad_offset_mm"]) / 1000.0
    want_rad = None
    for p in led.get("points", []):
        c = p.get("contact") or {}
        if c.get("close_rad_ceiling") is not None:
            want_rad = float(c["close_rad_ceiling"])
            break

    found, matches = {}, []
    for name, rel in CAL_FILES:
        doc = _load(os.path.join(project, rel))
        if not doc:
            continue
        ent = doc.get(key) or (doc.get("diameters", {}) or {}).get(key)
        if ent is None:
            continue
        found[name] = {"file": rel, "key": key, "entry": ent}
        ok_tool = (want_tool is None or ent.get("TOOL_OFFSET_Z") is None
                   or abs(float(ent["TOOL_OFFSET_Z"]) - want_tool) < 5e-5)
        ok_rad = (want_rad is None or ent.get("close_rad") is None
                  or abs(float(ent["close_rad"]) - want_rad) < 5e-5)
        if ok_tool and ok_rad:
            matches.append(name)

    return {"key": key, "shape": shape, "width_mm": width,
            "ledger_tool_offset_z_m": want_tool,
            "ledger_close_rad_ceiling": want_rad,
            "entries_found": found,
            "matched_files": matches,
            "used": (matches[0] if len(matches) == 1 else None),
            "ambiguous": len(matches) > 1}


def resolve_throat(run_dir, project):
    cfg = _load(os.path.join(run_dir, "gui_config_used.json")) or {}
    obj = (cfg.get("config", cfg)).get("object", {})
    shape = str(obj.get("shape", "cylinder")).lower()
    width = float(obj.get("diameter_mm", 0.0) or 0.0)
    key = f"{width:.1f}" if shape == "cylinder" else f"{width:.1f}|{shape}"
    doc = _load(os.path.join(project, "Data", "finger_throat.json"))
    if not doc:
        return {"key": key, "entry": None, "note": "finger_throat.json not found"}
    ent = (doc.get("diameters", {}) or {}).get(key)
    slim = None
    if ent is not None:
        # drop the point cloud: thousands of coordinates that would swamp the
        # manifest, and the scalars are what a later reader needs
        slim = {k: v for k, v in ent.items() if not isinstance(v, (list, dict))}
    return {"key": key, "entry": slim,
            "generated": doc.get("generated"),
            "fell_back_to_bare_key": (ent is None and
                                      (doc.get("diameters", {}) or {}).get(
                                          f"{width:.1f}") is not None)}


def code_versions(project):
    out = {}
    for rel in CODE_FILES:
        p = os.path.join(project, rel)
        if os.path.exists(p):
            d = _stat(p)
            d["sha256_16"] = _sha(p)
            out[rel] = d
        else:
            out[rel] = None
    return out


def derived(run_dir):
    """Run the analysis modules and keep their NUMBERS, not their prose."""
    out = {}

    try:
        import grid_accuracy as GA
        _rows, st = GA.measure(run_dir)
        st.pop("object", None)
        st.pop("grid", None)
        out["grid_accuracy"] = st
    except Exception as e:
        out["grid_accuracy"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        import grid_quality as GQ
        rows, meta = GQ.collect(run_dir, "s1")
        rows, ref = GQ.verdicts(rows)
        bad = [r for r in rows if r["verdict"] != "ok"]
        out["grid_quality"] = {
            "closing_mode": meta.get("closing_mode"),
            "closing_signal": meta.get("closing_signal"),
            "closing_target": meta.get("closing_target"),
            "n_points": len(rows),
            "median_stopped_rad": ref.get("median_stopped_rad"),
            "median_deform_m": ref.get("median_deform_m"),
            "n_flagged": len(bad),
            "flagged": [{"tag": r["tag"], "y": r["y"], "z": r["z"],
                         "deform": r["deform"], "shortfall": r["shortfall"],
                         "verdict": r["verdict"]} for r in bad],
            "per_point": [{"tag": r["tag"], "y": r["y"], "z": r["z"],
                           "deform": r["deform"], "stopped": r["stopped"],
                           "shortfall": r["shortfall"],
                           "peak_sum": r["peak_sum"],
                           "verdict": r["verdict"]} for r in rows]}
    except Exception as e:
        out["grid_quality"] = {"error": f"{type(e).__name__}: {e}"}

    # the stitcher already prints its key numbers; pull the two that matter
    sr = os.path.join(run_dir, "Stitched", "stitch_report.txt")
    if os.path.exists(sr):
        import re
        txt = open(sr).read()
        sig = {}
        for m in re.finditer(r"\[stitch (s\d)\] overlap sigma = "
                             r"([\d.]+) \((\d+)% of overlap signal\)", txt):
            sig[m.group(1)] = {"sigma": float(m.group(2)),
                               "pct_of_signal": int(m.group(3))}
        ext = {}
        for m in re.finditer(r"\[stitch (s\d)\].*?extension mm\s+up=([\d.]+) "
                             r"down=([\d.]+) left=([\d.]+) right=([\d.]+)",
                             txt):
            ext[m.group(1)] = {"up_mm": float(m.group(2)),
                               "down_mm": float(m.group(3)),
                               "left_mm": float(m.group(4)),
                               "right_mm": float(m.group(5))}
        out["stitch"] = {"overlap_sigma": sig, "extension": ext}
    else:
        out["stitch"] = {"error": "stitch_report.txt not found — run Stitch"}
    return out


def build(run_dir, project=DEFAULT_PROJECT, copy_provenance=True):
    """Write run_manifest.json + .txt, and Provenance/. Returns the dict."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    project = os.path.expanduser(project)
    inv, missing = inventory(run_dir)
    cal = resolve_calibration(run_dir, project)
    thr = resolve_throat(run_dir, project)

    man = {
        "manifest_version": 1,
        "written": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "project": project,
        "config": (_load(os.path.join(run_dir, "gui_config_used.json"))
                   or {}).get("object", {}),
        "grid": (_load(os.path.join(run_dir, "gui_config_used.json"))
                 or {}).get("grid", {}),
        "inventory": inv,
        "missing": [{"label": l, "pattern": p, "required": r}
                    for l, p, r in missing],
        "calibration": cal,
        "throat": thr,
        "plot_scale": _load(os.path.join(project, "Data", "plot_scale.json")),
        "code": code_versions(project),
        "derived": derived(run_dir),
    }

    if copy_provenance:
        pdir = os.path.join(run_dir, "Provenance")
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, "calibration_entry.json"), "w") as f:
            json.dump(cal, f, indent=2)
        with open(os.path.join(pdir, "throat_entry.json"), "w") as f:
            json.dump(thr, f, indent=2)
        for rel in ("Data/plot_scale.json",
                    "examples/scenes/ur5e_gripper.yml"):
            src = os.path.join(project, rel)
            if os.path.exists(src):
                try:
                    shutil.copy(src, os.path.join(pdir,
                                                  os.path.basename(rel)))
                except Exception:
                    pass
        man["provenance_dir"] = pdir

    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    txt = render(man)
    with open(os.path.join(run_dir, "run_manifest.txt"), "w") as f:
        f.write(txt + "\n")
    return man, txt


def render(man):
    o, g = man["config"], man["grid"]
    L = ["RUN MANIFEST — is this run self-contained?",
         f"run     : {man['run_name']}",
         f"path    : {man['run_dir']}",
         f"written : {man['written']}"]
    if o:
        L.append(f"object  : {o.get('shape', '?')}  grip {o.get('diameter_mm')} "
                 f"x across {o.get('across_mm', o.get('diameter_mm'))} "
                 f"x along {o.get('length_mm')} mm, "
                 f"tilt {o.get('tilt_deg', 0)} about {o.get('tilt_axis', 'X')}")
    if g:
        L.append(f"grid    : {g.get('n_points')} points, step "
                 f"{g.get('step_mm')} mm, centered={g.get('centered')}, "
                 f"roll {g.get('step_roll_deg', 0)} deg")

    L += ["", "  CONTENTS"]
    for it in man["inventory"]:
        mark = "  ok " if it["count"] else ("  !! " if it["required"]
                                            else "   - ")
        n = f"{it['count']:4d}" if it["count"] else "   0"
        extra = ""
        if it["count"]:
            mb = it.get("total_bytes", 0) / 1e6
            extra = f"  {mb:8.2f} MB   {it.get('newest', {}).get('mtime', '')}"
        L.append(f"{mark}{n}  {it['label']:22s}{extra}")

    req_missing = [m for m in man["missing"] if m["required"]]
    if req_missing:
        L += ["", "  !! REQUIRED AND MISSING — this run cannot be rebuilt:"]
        for m in req_missing:
            L.append(f"     {m['label']}  ({m['pattern']})")

    c = man["calibration"]
    L += ["", "  CALIBRATION — identified by MEASUREMENT, not by trusting a flag"]
    L.append(f"     key wanted            : {c['key']}")
    L.append(f"     ledger TOOL_OFFSET_Z  : "
             f"{c['ledger_tool_offset_z_m']} m")
    L.append(f"     ledger close_rad      : {c['ledger_close_rad_ceiling']}")
    L.append(f"     files holding this key: "
             f"{', '.join(c['entries_found']) or 'NONE'}")
    if c["used"]:
        e = c["entries_found"][c["used"]]["entry"]
        L.append(f"     -> USED: {c['used']}  (TOOL_OFFSET_Z "
                 f"{e.get('TOOL_OFFSET_Z')}, close_rad {e.get('close_rad')}, "
                 f"measured {e.get('measured_at')})")
    elif c["ambiguous"]:
        L.append(f"     !! AMBIGUOUS: {', '.join(c['matched_files'])} all "
                 f"match. Cannot say which was used.")
    else:
        L.append("     !! NO ENTRY MATCHES THE LEDGER. Either the calibration "
                 "file changed after the run, or the run used a manual "
                 "override. The numbers above are the truth; the files are not.")

    t = man["throat"]
    L += ["", "  FINGER THROAT"]
    if t.get("entry"):
        e = t["entry"]
        L.append(f"     {t['key']}: min_depth {e.get('min_depth_mm')} mm, "
                 f"pad separation {e.get('pad_separation_mm')} mm, "
                 f"blocked by {e.get('blocking_part_at_min_depth')}")
    else:
        L.append(f"     !! no entry for {t['key']}"
                 + ("  (a BARE-WIDTH entry exists — a cylinder's throat was "
                    "used for a non-cylinder)" if t.get("fell_back_to_bare_key")
                    else ""))

    d = man["derived"]
    L += ["", "  DERIVED NUMBERS (also in run_manifest.json, machine-readable)"]
    ga = d.get("grid_accuracy", {})
    if "error" in ga:
        L.append(f"     grid accuracy : {ga['error']}")
    else:
        L.append(f"     grid accuracy : bias ({ga['bias_y_mm']:+.3f}, "
                 f"{ga['bias_z_mm']:+.3f}) mm, scatter "
                 f"{100*ga['scatter_taxel_frac']:.2f}% of a taxel, "
                 f"worst {100*ga['max_miss_taxel_frac']:.2f}% at "
                 f"{ga['max_miss_at']}")
    gq = d.get("grid_quality", {})
    if "error" in gq:
        L.append(f"     grid quality  : {gq['error']}")
    else:
        L.append(f"     grid quality  : closing={gq['closing_mode']}, "
                 f"median indentation "
                 f"{1000*(gq['median_deform_m'] or 0):.2f} mm, "
                 f"{gq['n_flagged']} of {gq['n_points']} points flagged")
        if gq["n_flagged"]:
            L.append("                     flagged: "
                     + ", ".join(r["tag"] for r in gq["flagged"][:20])
                     + (" ..." if gq["n_flagged"] > 20 else ""))
    stt = d.get("stitch", {})
    if "error" in stt:
        L.append(f"     stitch        : {stt['error']}")
    else:
        for s, v in sorted(stt.get("overlap_sigma", {}).items()):
            e = stt.get("extension", {}).get(s, {})
            L.append(f"     stitch {s}      : overlap sigma {v['sigma']:.1f} "
                     f"({v['pct_of_signal']}%), extension "
                     f"u{e.get('up_mm')} d{e.get('down_mm')} "
                     f"l{e.get('left_mm')} r{e.get('right_mm')} mm")

    L += ["", "  CODE THAT PRODUCED THIS RUN (sha256, first 16)"]
    for rel, v in man["code"].items():
        if v is None:
            L.append(f"     {rel:38s} NOT FOUND")
        else:
            L.append(f"     {rel:38s} {v['sha256_16']}  {v['mtime']}")

    L += ["",
          "Two runs whose code hashes differ were produced by different",
          "software and should not be pooled without saying so. The",
          "calibration and throat entries above are COPIES of what was in",
          "force at run time -- re-calibrating later cannot rewrite them."]
    return "\n".join(L)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    proj = DEFAULT_PROJECT
    for i, a in enumerate(sys.argv):
        if a == "--project" and i + 1 < len(sys.argv):
            proj = sys.argv[i + 1]
    if not args:
        print(__doc__)
        sys.exit(1)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    _m, _t = build(args[0], proj)
    print(_t)
    print(f"\nsaved {os.path.join(args[0], 'run_manifest.json')}")
    print(f"saved {os.path.join(args[0], 'run_manifest.txt')}")
