#!/usr/bin/env python3
"""fix_run_config.py -- repair the object block in a run's gui_config_used.json.

WHY THIS EXISTS
---------------
heatmaps.py, stitching.py and blob_axis all learn the object's geometry from
ONE place: cfg["object"] in the run folder's gui_config_used.json. Runs whose
config was written before 2026-08-28 carry

    "shape": "cylinder"           (hardcoded, whatever the object really was)
    no "across_mm" key            (so every consumer falls back to diameter_mm)

so a 30 x 120 x 140 cuboid is described to all of them as a Ø30 cylinder:

  * the object outline is drawn 30 mm wide instead of 120 mm;
  * the EXPECTED panel predicts a chord band 2*sqrt(2.4*(30-2.4)) = 16.3 mm
    wide, centred on the object -- which a pad anchored 32 mm off centre
    never touches, so the panel comes out empty;
  * the stitched overlay outlines the wrong object.

Nothing is wrong with the collected tactile data. Only the label is wrong, so
the run does NOT need re-collecting -- rewrite the block and replot.

The three axes, per handoff v13 section 1.1:
    world X = what the JAW closes on   -> diameter_mm (grip width)
    world Y = ACROSS the pad           -> across_mm   (band + outline width)
    world Z = ALONG the pad            -> length_mm

USAGE
-----
  python3 fix_run_config.py RUN_DIR --shape cuboid --across-mm 120
  python3 fix_run_config.py RUN_DIR --shape cuboid --across-mm 120 \
          --grip-mm 30 --length-mm 140
  python3 fix_run_config.py RUN_DIR --check          (report only, no write)

The original file is copied to gui_config_used.json.bak_before_fix before any
write. Run it again with different numbers and it re-reads the CURRENT file,
so the backup is only ever made once.
"""

import argparse
import json
import os
import shutil
import sys

CFG_NAME = "gui_config_used.json"
BAK_SUFFIX = ".bak_before_fix"
VALID_SHAPES = ("cylinder", "cuboid", "cube", "sphere")


def _report(o, label):
    """Print the object block the way every consumer will read it."""
    across = o.get("across_mm", None)
    print(f"  [{label}]")
    print(f"    shape        : {o.get('shape', '(missing)')}")
    print(f"    diameter_mm  : {o.get('diameter_mm', '(missing)')}   (grip, world X)")
    if across is None:
        print(f"    across_mm    : (MISSING -> consumers fall back to "
              f"diameter_mm = {o.get('diameter_mm', '?')})")
    else:
        print(f"    across_mm    : {across}   (world Y -- band + outline)")
    print(f"    length_mm    : {o.get('length_mm', '(missing)')}   (along, world Z)")
    print(f"    centre       : {o.get('center_world_mm', '(missing)')}")
    print(f"    tilt         : {o.get('tilt_deg', 0.0)} deg about "
          f"{o.get('tilt_axis', 'X')}")


def _predicted_band(o):
    """What heatmaps._band_width_mm will return for this block, in mm."""
    import math
    indent = 2.4                       # heatmaps.INDENT_MM
    d = float(o.get("across_mm", o.get("diameter_mm", 26.0)))
    shape = str(o.get("shape", "cylinder")).lower()
    if shape in ("cuboid", "cube"):
        return d, f"whole {d:.0f} mm face"
    if d > 2 * indent:
        return 2.0 * math.sqrt(indent * (d - indent)), f"chord from D{d:.0f}"
    return 8.0, "fallback"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run folder containing " + CFG_NAME)
    ap.add_argument("--shape", choices=VALID_SHAPES,
                    help="what the object actually is")
    ap.add_argument("--across-mm", type=float,
                    help="extent ACROSS the pad, world Y (cuboid: the D field)")
    ap.add_argument("--grip-mm", type=float,
                    help="what the jaw closes on, world X (default: leave as is)")
    ap.add_argument("--length-mm", type=float,
                    help="extent ALONG the pad, world Z (default: leave as is)")
    ap.add_argument("--check", action="store_true",
                    help="report the current block and exit without writing")
    a = ap.parse_args()

    path = os.path.join(os.path.expanduser(a.run_dir), CFG_NAME)
    if not os.path.exists(path):
        sys.exit(f"ERROR: no {CFG_NAME} in {a.run_dir}")

    with open(path) as f:
        cfg = json.load(f)
    o = cfg.get("object")
    if not isinstance(o, dict) or "center_world_mm" not in o:
        sys.exit(f"ERROR: {path} has no usable 'object' block")

    print(f"file: {path}")
    _report(o, "BEFORE")
    bw, how = _predicted_band(o)
    print(f"    -> expected band: {bw:.1f} mm ({how})")

    if a.check:
        return
    if a.shape is None and a.across_mm is None and a.grip_mm is None \
            and a.length_mm is None:
        sys.exit("\nNothing to change. Pass --shape / --across-mm "
                 "(or --check to report only).")

    if not os.path.exists(path + BAK_SUFFIX):
        shutil.copy(path, path + BAK_SUFFIX)
        print(f"\nbackup written: {os.path.basename(path)}{BAK_SUFFIX}")
    else:
        print(f"\nbackup already exists, kept: "
              f"{os.path.basename(path)}{BAK_SUFFIX}")

    if a.shape is not None:
        o["shape"] = a.shape
    if a.grip_mm is not None:
        o["diameter_mm"] = float(a.grip_mm)
    if a.length_mm is not None:
        o["length_mm"] = float(a.length_mm)
    # across_mm must ALWAYS end up present: its absence is exactly the fault
    # this tool repairs. When it is not given, fall it back to the grip width,
    # which is correct for anything round and explicit for everything else.
    o["across_mm"] = float(a.across_mm if a.across_mm is not None
                           else o.get("across_mm", o.get("diameter_mm", 26.0)))
    o["_repaired_by"] = "fix_run_config.py"

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

    _report(o, "AFTER")
    bw, how = _predicted_band(o)
    print(f"    -> expected band: {bw:.1f} mm ({how})")
    print("\nNow replot this run (heatmaps + stitching). No re-collection "
          "is needed: the tactile CSVs were never wrong.")


if __name__ == "__main__":
    main()
