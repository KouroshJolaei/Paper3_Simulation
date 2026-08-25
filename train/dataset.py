#!/usr/bin/env python3
"""
dataset.py — find, VALIDATE and load Paper-3 training pairs.

The point of this file is the validation. A run folder can contain a
training_pair.npz that loads perfectly and is still useless or actively
harmful to train on, and none of those cases announce themselves:

  * a 1-point "grid", where input == target and there is nothing to predict
  * a pair written before the canvas was pinned (74x74, 87x48, 99x99 world
    frame), which cannot be batched with anything current
  * a pair whose initial grasp was a SUBSTITUTE for the designed one
  * a real-rig pair whose TOOL_OFFSET_Z was never measured on hardware, so
    every pad position in it carries an unverified constant offset

Each of those trains happily and produces a plausible loss curve. So they are
rejected here, by name, with the reason printed — and a run that is rejected
is listed rather than silently skipped.

THE MASK. target_mask marks the cells a pad actually visited. Everything
outside it is not "no contact", it is "nobody looked", and grading it would
teach the model the shape of the sweep instead of the shape of the object.
Every loss and every metric in this project takes the mask.

Standalone:
    python3 dataset.py --scan                  # what have I got?
    python3 dataset.py --scan --verbose        # and why was each one rejected
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

PROJECT = os.path.expanduser("~/Paper3_Simulation")
SIM_DIRS = [os.path.join(PROJECT, "Data", "gui_run", "SIM"),
            os.path.join(PROJECT, "Data", "gui_run")]
REAL_DIRS = [os.path.join(PROJECT, "Data", "gui_run", "Real")]

# The pinned canvas. Anything else cannot go in the same batch.
# Mirrors stitching.INITIAL_GRASP. Used only to skip the anchored copy of
# the pt00 pair, which duplicates training_pair.npz exactly -- including it
# would put one sample in the set twice and weight that grasp double.
INITIAL_GRASP = "pt00"

EXPECT_SHAPE = (96, 96)
EXPECT_RES_MM = 1.0
EXPECT_FRAME = "pad"

# A pair must have measured meaningfully more than the initial pad, or the
# model is being asked to predict something already in its input. The pad
# alone paints ~851 cells; 1.25x is a deliberately loose floor that still
# catches the degenerate 1-point case (ratio exactly 1.00).
MIN_TARGET_OVER_INPUT = 1.25


def _run_name(path):
    """The RUN a pair belongs to, whatever depth it sits at.

        <run>/Stitched/training_pair.npz          -> <run>
        <run>/Stitched/pairs/pair_pt07_*.npz      -> <run>

    Taking dirname twice works for the first and returns "Stitched" for the
    second. That is not cosmetic: split_by_run groups on this name, so every
    anchored pair would have been filed under a run called "Stitched" and the
    anchors of one sweep could land on BOTH sides of the train/val split --
    the exact leak the by-run split exists to prevent, and one that would
    have shown up only as a suspiciously good validation score.
    """
    d = os.path.dirname(os.path.abspath(path))
    if os.path.basename(d) == "pairs":
        d = os.path.dirname(d)
    if os.path.basename(d) == "Stitched":
        d = os.path.dirname(d)
    return os.path.basename(d)


class Pair:
    """One validated (input, target, mask) triple plus where it came from."""

    def __init__(self, path, sensor, meta, rig):
        self.path, self.sensor, self.meta, self.rig = path, sensor, meta, rig
        self.run = _run_name(path)

    def load(self):
        d = np.load(self.path, allow_pickle=True)
        s = self.sensor
        return (d[f"input_{s}"].astype(np.float32),
                d[f"target_{s}"].astype(np.float32),
                d[f"target_mask_{s}"].astype(bool))

    def __repr__(self):
        return f"<Pair {self.run}/{self.sensor} rig={self.rig}>"


def _rig_of(run_dir):
    """sim or real, from the run's own files rather than the folder name."""
    for name in ("reachability_report.json", "pose_history.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    d = json.load(f)
                if str(d.get("rig", "")).lower() == "real":
                    return "real"
            except Exception:
                pass
    return "real" if os.sep + "Real" + os.sep in run_dir + os.sep else "sim"


def _cal_source(run_dir):
    for name in ("reachability_report.json", "pose_history.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    v = json.load(f).get("calibration_source")
                if v:
                    return str(v)
            except Exception:
                pass
    return None


def inspect_pair(npz_path, allow_sim_fallback=False):
    """-> (pairs, notes). pairs is [] when the file is unusable; notes always
    explains what was found, whether it passed or not."""
    notes, pairs = [], []
    run_dir = os.path.dirname(os.path.dirname(npz_path))
    rig = _rig_of(run_dir)

    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return [], [f"unreadable: {e}"]

    try:
        meta = json.loads(str(d["meta"]))
    except Exception as e:
        return [], [f"no readable meta: {e}"]

    # --- canvas convention -------------------------------------------
    res = float(meta.get("res_mm", -1))
    frame = str(meta.get("canvas_frame", "?"))
    if abs(res - EXPECT_RES_MM) > 1e-9:
        notes.append(f"res_mm={res} (need {EXPECT_RES_MM}) — exported before "
                     f"the canvas was pinned")
    if frame != EXPECT_FRAME:
        notes.append(f"canvas_frame={frame!r} (need {EXPECT_FRAME!r}) — a "
                     f"world-frame pair puts the same contact at a different "
                     f"orientation depending on the wrist")

    # --- calibration provenance (real runs only) ----------------------
    src = _cal_source(run_dir)
    if rig == "real" and src and src != "real":
        msg = (f"calibration_source={src} — TOOL_OFFSET_Z was not measured on "
               f"the real rig, so every pad position carries an unverified "
               f"constant offset")
        if allow_sim_fallback:
            notes.append("WARNING: " + msg)
        else:
            notes.append(msg)

    hard_fail = [n for n in notes if not n.startswith("WARNING")]

    for s in ("s1", "s2"):
        need = [f"input_{s}", f"target_{s}", f"target_mask_{s}"]
        if any(k not in d.files for k in need):
            notes.append(f"{s}: missing arrays")
            continue
        inp, tgt, msk = d[f"input_{s}"], d[f"target_{s}"], d[f"target_mask_{s}"]
        if inp.shape != EXPECT_SHAPE:
            notes.append(f"{s}: shape {inp.shape} (need {EXPECT_SHAPE})")
            continue
        st = str(meta.get(f"initial_status_{s}", "?"))
        if st != "designed":
            notes.append(f"{s}: initial_status={st!r} — the input frame is "
                         f"NOT the grasp the grid was designed around")
            continue
        n_in = int((d[f"input_mask_{s}"]).sum()) if f"input_mask_{s}" in d.files \
            else int((inp != 0).sum())
        n_tg = int(msk.sum())
        if n_in == 0 or n_tg == 0:
            notes.append(f"{s}: empty mask")
            continue
        ratio = n_tg / max(n_in, 1)
        if ratio < MIN_TARGET_OVER_INPUT:
            notes.append(f"{s}: target covers {n_tg} cells vs input {n_in} "
                         f"(x{ratio:.2f}) — nothing to predict beyond the "
                         f"input itself")
            continue
        if not np.isfinite(inp).all() or not np.isfinite(tgt).all():
            notes.append(f"{s}: non-finite values")
            continue
        if hard_fail:
            continue
        pairs.append(Pair(npz_path, s, meta, rig))
        notes.append(f"{s}: OK — {meta.get(f'n_grasps_{s}', '?')} grasps, "
                     f"input {n_in} cells, target {n_tg} cells (x{ratio:.2f}), "
                     f"max in {float(inp.max()):.0f} / tgt {float(tgt.max()):.0f}")
    return pairs, notes


def survey_runs(roots):
    """EVERY run folder found, and what state it is in.

    find_pairs below only ever saw folders that already contained a
    training_pair.npz. A run that was never exported, or whose export was
    REFUSED, simply did not appear anywhere in the output -- so a drive with
    35 run folders reported on 7 and said nothing at all about the other 28.
    That is the same silent gap this project has removed everywhere else: an
    absence that looks like a clean result.

    Returns [(run_dir, state, detail)] with state one of
        "pair"        an npz exists (its contents are judged by inspect_pair)
        "no_pair"     grasps were collected but no npz was written
        "no_grasps"   the folder has no per-grasp tactile CSVs at all
    """
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d in sorted(glob.glob(os.path.join(root, "**", "run_*"),
                                  recursive=True)):
            if not os.path.isdir(d):
                continue
            npz = os.path.join(d, "Stitched", "training_pair.npz")
            n_csv = len(glob.glob(os.path.join(d, "*_pt*_s1_tactile_maps.csv")))
            if os.path.exists(npz):
                out.append((d, "pair", f"{n_csv} grasp CSVs"))
            elif n_csv > 0:
                # Why was it not exported? The commonest reasons leave
                # evidence in the folder, so say which rather than "no pair".
                why = f"{n_csv} grasp CSVs but no training_pair.npz"
                led = os.path.join(d, "execution_ledger.json")
                if os.path.exists(led):
                    try:
                        with open(led) as f:
                            L = json.load(f)
                        if L.get("aborted"):
                            why += " — run ABORTED"
                        tags = {p.get("tag") for p in L.get("points", [])
                                if p.get("executed")}
                        if tags and "pt00" not in tags:
                            why += " — pt00 never executed, so export refuses"
                    except Exception:
                        pass
                if not os.path.isdir(os.path.join(d, "Stitched")):
                    why += " — never stitched"
                out.append((d, "no_pair", why))
            else:
                out.append((d, "no_grasps", "no per-grasp tactile CSVs"))
    return out


def find_pairs(rigs=("sim",), allow_sim_fallback=False, verbose=False,
               roots=None, anchors="none"):
    """Scan the run trees. Returns (pairs, rejected) where rejected is
    [(path, notes)] so nothing disappears without explanation."""
    seen, cand = set(), []
    dirs = list(roots) if roots else (
        (list(SIM_DIRS) if "sim" in rigs else []) +
        (list(REAL_DIRS) if "real" in rigs else []))
    for root in dirs:
        if not os.path.isdir(root):
            continue
        # training_pair.npz (the pt00 pair) plus, when asked for, the
        # anchored pairs in Stitched/pairs/. The pt00 file keeps its own
        # name and place so every earlier scan and export stays valid.
        _pats = [os.path.join(root, "**", "Stitched", "training_pair.npz")]
        if anchors != "none":
            _pats.append(os.path.join(root, "**", "Stitched", "pairs",
                                      "pair_*.npz"))
        _found = []
        for _pat in _pats:
            _found += glob.glob(_pat, recursive=True)
        for p in sorted(_found):
            # pair_pt00_interior.npz duplicates training_pair.npz exactly;
            # keeping both would put the same sample in twice and, worse,
            # weight that one grasp double in the loss.
            _b = os.path.basename(p)
            if _b.startswith("pair_") and f"_{INITIAL_GRASP}_" in _b:
                continue
            if anchors == "interior" and _b.startswith("pair_") \
                    and "_edge" in _b:
                continue
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                cand.append(p)

    keep, rejected = [], []
    for p in cand:
        got, notes = inspect_pair(p, allow_sim_fallback=allow_sim_fallback)
        got = [g for g in got if g.rig in rigs]
        if got:
            keep.extend(got)
        else:
            rejected.append((p, notes))
        if verbose:
            run = _run_name(p)
            print(f"\n{run}")
            for n in notes:
                print(f"    {n}")
    return keep, rejected


def split_by_run(pairs, val_frac=0.25, seed=0):
    """Train/val split BY RUN, never by pair.

    s1 and s2 of one grasp see the same object at the same pose, and (later)
    several anchors will come from one sweep. Splitting by pair would put
    near-duplicates on both sides and the validation score would measure
    memorisation."""
    runs = sorted({p.run for p in pairs})
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    n_val = max(1, int(round(len(runs) * val_frac))) if len(runs) > 1 else 0
    val_runs = set(runs[:n_val])
    tr = [p for p in pairs if p.run not in val_runs]
    va = [p for p in pairs if p.run in val_runs]
    return tr, va, sorted(val_runs)


def stack(pairs):
    """-> X (N,1,96,96), Y (N,1,96,96), M (N,1,96,96) float32/bool."""
    X, Y, M = [], [], []
    for p in pairs:
        i, t, m = p.load()
        X.append(i[None]); Y.append(t[None]); M.append(m[None])
    if not X:
        z = np.zeros((0, 1) + EXPECT_SHAPE, np.float32)
        return z, z, z.astype(bool)
    return np.stack(X), np.stack(Y), np.stack(M)


def norm_stats(pairs):
    """Scale for the network input. Taken from the TRAIN split only — using
    all data would leak the validation runs' scale into training.

    Sim and real are in different units (sim tactile counts ~0-1400, real raw
    capacitance ~10k-36k), which is why they are not mixed without this."""
    vals = []
    for p in pairs:
        i, t, m = p.load()
        vals.append(float(np.percentile(t[m], 99)) if m.any() else 0.0)
    v = float(np.median(vals)) if vals else 1.0
    return max(v, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rig", default="sim", choices=["sim", "real", "both"])
    ap.add_argument("--allow-sim-fallback", action="store_true")
    ap.add_argument("--roots", nargs="*", default=None)
    a = ap.parse_args()
    rigs = ("sim", "real") if a.rig == "both" else (a.rig,)

    pairs, rejected = find_pairs(rigs=rigs,
                                 allow_sim_fallback=a.allow_sim_fallback,
                                 verbose=a.verbose, roots=a.roots)

    # EVERY run folder is accounted for, not only the ones that happen to
    # have an npz. A folder that was never exported is a fact about the data
    # set; leaving it out of the report makes the set look cleaner than it is.
    roots = a.roots if a.roots else (
        (list(SIM_DIRS) if "sim" in rigs else []) +
        (list(REAL_DIRS) if "real" in rigs else []))
    survey = survey_runs(roots)
    with_pair = [r for r in survey if r[1] == "pair"]
    no_pair = [r for r in survey if r[1] == "no_pair"]
    no_grasp = [r for r in survey if r[1] == "no_grasps"]
    print(f"\n=== RUN FOLDERS FOUND: {len(survey)} ===")
    print(f"  {len(with_pair):>3} have a training_pair.npz")
    print(f"  {len(no_pair):>3} have grasps but NO pair")
    print(f"  {len(no_grasp):>3} have no grasp data at all")
    if no_pair:
        print(f"\n=== NO PAIR EXPORTED ({len(no_pair)}) ===")
        for d, _st, why in no_pair:
            print(f"  {os.path.basename(d):<44} {why}")
    if no_grasp and a.verbose:
        print(f"\n=== NO GRASP DATA ({len(no_grasp)}) ===")
        for d, _st, why in no_grasp:
            print(f"  {os.path.basename(d):<44} {why}")
    elif no_grasp:
        print(f"\n  ({len(no_grasp)} folders hold no grasp data; --verbose "
              f"lists them)")

    print(f"\n=== USABLE: {len(pairs)} pairs from "
          f"{len({p.run for p in pairs})} runs ===")
    for p in pairs:
        print(f"  {p.run:<44} {p.sensor}  {p.rig}")
    if rejected:
        print(f"\n=== REJECTED: {len(rejected)} ===")
        for path, notes in rejected:
            run = _run_name(path)
            print(f"  {run}")
            for n in notes:
                print(f"      {n}")
    if pairs:
        tr, va, vruns = split_by_run(pairs)
        print(f"\nsplit BY RUN: {len(tr)} train / {len(va)} val")
        print(f"  validation runs: {vruns}")
        print(f"  input scale (train only): {norm_stats(tr):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
