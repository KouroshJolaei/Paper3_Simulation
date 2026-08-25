#!/usr/bin/env python3
"""
train.py — train the contact-completion model on validated pairs.

    python3 train.py --scan-only          # what data is usable, no training
    python3 train.py --epochs 200
    python3 train.py --rig both --allow-sim-fallback

WHAT IT WRITES, into out/<timestamp>/
    best.pt          weights at the best validation loss
    history.csv      per-epoch train/val loss and metrics
    report.txt       the run's settings, data, and final numbers
    curves.png       loss and metric curves
    preds.png        input | target | prediction | error, for val pairs

TWO BASELINES ARE ALWAYS REPORTED, and they matter more than the model's
own number in isolation:

    zero    predict nothing anywhere
    copy    predict the input, unchanged, everywhere

"copy" is the honest one to beat. A large part of any target IS the initial
imprint, so a model that learns to reproduce its input and stop already
scores well. Until the model beats copy by a clear margin it has not learned
to extrapolate — it has learned to echo. With a handful of pairs that is the
most likely outcome, and it should be visible rather than hidden behind an
impressive-looking SSIM.

Everything is scored on MEASURED cells only (see model.masked_l1).
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as DS                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rig", default="sim", choices=["sim", "real", "both"])
ap.add_argument("--allow-sim-fallback", action="store_true")
ap.add_argument("--anchors", default="none",
                choices=["none", "interior", "all"],
                help="none: the pt00 pair only (default). interior: also "
                     "pairs anchored on grasps that have measured target on "
                     "all sides. all: also the rim anchors, whose target is "
                     "lopsided. Extra anchors are NOT extra objects -- they "
                     "reuse one sweep, so they augment rather than enlarge "
                     "the set, and split_by_run keeps them together.")
ap.add_argument("--roots", nargs="*", default=None)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--base", type=int, default=32)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--val-frac", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out"))
ap.add_argument("--scan-only", action="store_true")
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()

RIGS = ("sim", "real") if args.rig == "both" else (args.rig,)


def _nanmean(a):
    """np.nanmean warns and returns nan for an all-nan slice. An all-nan TC
    is a real outcome — no map had contact above threshold — so it is
    returned quietly rather than as a warning in the middle of training."""
    a = np.asarray(a, float)
    good = a[np.isfinite(a)]
    return float(good.mean()) if good.size else float("nan")


def main():
    pairs, rejected = DS.find_pairs(rigs=RIGS,
                                    allow_sim_fallback=args.allow_sim_fallback,
                                    roots=args.roots, anchors=args.anchors)

    # Account for every run folder, not just the ones with an npz. A folder
    # that was never exported is part of the picture: without this line a
    # drive holding 35 runs reported on 7 and said nothing about the rest.
    _roots = args.roots if args.roots else (
        (list(DS.SIM_DIRS) if "sim" in RIGS else []) +
        (list(DS.REAL_DIRS) if "real" in RIGS else []))
    _sv = DS.survey_runs(_roots)
    _np_ = [r for r in _sv if r[1] == "no_pair"]
    print(f"run folders  : {len(_sv)} found — "
          f"{len([r for r in _sv if r[1] == 'pair'])} with a pair, "
          f"{len(_np_)} with grasps but no pair, "
          f"{len([r for r in _sv if r[1] == 'no_grasps'])} with no grasps")
    for d, _st, why in _np_:
        print(f"  no pair  {os.path.basename(d)}: {why}")

    _nanch = sum(1 for p in pairs
                 if os.path.basename(p.path).startswith("pair_"))
    print(f"usable pairs : {len(pairs)} from "
          f"{len({p.run for p in pairs})} runs"
          + (f"  ({_nanch} of them anchored, --anchors={args.anchors})"
             if _nanch else ""))
    for path, notes in rejected:
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        print(f"  rejected {run}: {notes[0] if notes else '?'}")
    if not pairs:
        print("\nnothing usable. Run `python3 dataset.py --scan --verbose` "
              "to see why each candidate was refused.")
        return 1

    tr, va, val_runs = DS.split_by_run(pairs, args.val_frac, args.seed)

    # VALIDATION IS ALWAYS pt00-ONLY, whatever --anchors is set to.
    #
    # Anchored pairs are AUGMENTATION: they re-centre the canvas on a
    # different grasp of the SAME sweep. Letting them into validation changes
    # the exam, not just the study material -- and the proof is that `copy`,
    # a fixed rule that cannot get better or worse, scored 0.2094 / 0.461 with
    # anchors=none and 0.2198 / 0.338 with anchors=all. Three models were
    # being marked against three different papers, so no comparison between
    # them meant anything. Holding validation to the pt00 pairs makes the runs
    # comparable, and is the standard rule anyway: never augment a test set.
    _anch = lambda p: os.path.basename(p.path).startswith("pair_")
    _va_all = list(va)
    va = [p for p in va if not _anch(p)]
    _dropped = len(_va_all) - len(va)
    print(f"split BY RUN : {len(tr)} train / {len(va)} val   "
          f"(val runs: {val_runs})")
    if _dropped:
        print(f"  validation held to pt00 only: {_dropped} anchored pair(s) "
              f"removed from val so every --anchors setting is graded on the "
              f"same exam")
    if not va and _va_all:
        print("  WARNING: the validation runs have no pt00 pair, so there is "
              "nothing left to validate on. Re-export them or change --seed.")
    if not va:
        print("  NOTE: only one run — no validation split is possible, so "
              "every number below is a TRAINING number and says nothing "
              "about generalisation.")
    if len({p.run for p in tr}) < 3:
        print(f"  NOTE: {len({p.run for p in tr})} training run(s). This is "
              f"enough to prove the pipeline runs end to end and NOT enough "
              f"to conclude anything about the method.")

    if args.scan_only:
        return 0

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import model as M

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                       else "cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Scale from the TRAIN split only: taking it from everything would leak
    # the validation runs' magnitude into training.
    scale = DS.norm_stats(tr)
    print(f"input scale  : {scale:.1f}  (99th pct of train targets)")

    def tens(ps):
        X, Y, Mk = DS.stack(ps)
        return (torch.from_numpy(X / scale), torch.from_numpy(Y / scale),
                torch.from_numpy(Mk))

    Xtr, Ytr, Mtr = tens(tr)
    Xva, Yva, Mva = tens(va) if va else (Xtr[:0], Ytr[:0], Mtr[:0])
    dl = DataLoader(TensorDataset(Xtr, Ytr, Mtr),
                    batch_size=min(args.batch, len(tr)), shuffle=True)

    net = M.UNet(base=args.base, depth=args.depth).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20,
                                                       factor=0.5)
    print(f"model        : UNet base={args.base} depth={args.depth}, "
          f"{n_par/1e6:.2f} M params, device {dev}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The mode goes in the FOLDER NAME, not just inside a file: three runs an
    # hour apart are otherwise three timestamps you have to open to tell
    # apart, and that is how the wrong pair of numbers ends up compared.
    _tag = {"none": "pt00only", "interior": "interior",
            "all": "interior+edge"}[args.anchors]
    out_dir = os.path.join(args.out, f"{stamp}_{_tag}")
    # ...and every file inside carries it too. Folder names are lost the
    # moment a report or a figure is copied into a slide or an email, which
    # is exactly when two runs get compared without noticing they were
    # trained on different data.
    _sfx = f"_{_tag}"
    os.makedirs(out_dir, exist_ok=True)

    def evaluate(X, Y, Mk):
        if len(X) == 0:
            return {}
        net.eval()
        with torch.no_grad():
            x, y, m = X.to(dev), Y.to(dev), Mk.to(dev)
            p = net(x)
            return {"l1": float(M.masked_l1(p, y, m)),
                    "mse": float(M.masked_mse(p, y, m)),
                    "ssim": float(M.masked_ssim(p, y, m)),
                    "tc_mm": _nanmean(
                        M.tactile_centroid_error_mm(p, y, m).cpu().numpy())}

    def baselines(X, Y, Mk):
        if len(X) == 0:
            return {}
        x, y, m = X.to(dev), Y.to(dev), Mk.to(dev)
        z = torch.zeros_like(y)
        return {"zero_l1": float(M.masked_l1(z, y, m)),
                "copy_l1": float(M.masked_l1(x, y, m)),
                "zero_ssim": float(M.masked_ssim(z, y, m)),
                "copy_ssim": float(M.masked_ssim(x, y, m))}

    base_tr = baselines(Xtr, Ytr, Mtr)
    base_va = baselines(Xva, Yva, Mva) if len(Xva) else {}
    print(f"\nbaselines (train): zero L1 {base_tr['zero_l1']:.4f}   "
          f"copy L1 {base_tr['copy_l1']:.4f}")
    if base_va:
        print(f"baselines (val)  : zero L1 {base_va['zero_l1']:.4f}   "
              f"copy L1 {base_va['copy_l1']:.4f}")
    print("  ^ the model must beat COPY to have learned anything beyond "
          "echoing its input\n")

    hist, best, best_ep = [], float("inf"), -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        tot = nb = 0
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            opt.zero_grad()
            loss = M.masked_l1(net(xb), yb, mb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        tr_l = tot / max(nb, 1)
        ev_tr = evaluate(Xtr, Ytr, Mtr)
        ev_va = evaluate(Xva, Yva, Mva)
        watch = ev_va.get("l1", ev_tr["l1"])
        sched.step(watch)
        hist.append({"epoch": ep, "train_loss": tr_l,
                     **{f"tr_{k}": v for k, v in ev_tr.items()},
                     **{f"va_{k}": v for k, v in ev_va.items()}})
        if watch < best:
            best, best_ep = watch, ep
            torch.save({"model": net.state_dict(), "scale": scale,
                        "args": vars(args)},
                       os.path.join(out_dir, f"best{_sfx}.pt"))
        if ep % 10 == 0 or ep == 1:
            msg = (f"ep {ep:>4}  train {tr_l:.4f}  "
                   f"tr_ssim {ev_tr['ssim']:.3f}")
            if ev_va:
                msg += (f"  |  val {ev_va['l1']:.4f}  "
                        f"val_ssim {ev_va['ssim']:.3f}  "
                        f"val_TC {ev_va['tc_mm']:.2f} mm")
            print(msg)

    dt = time.time() - t0
    with open(os.path.join(out_dir, f"history{_sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)

    final = hist[-1]
    ref = base_va if base_va else base_tr
    got = final.get("va_l1", final["tr_l1"])
    verdict = ("BEATS copy" if got < ref["copy_l1"] else
               "DOES NOT beat copy — the model is echoing its input, not "
               "extrapolating")

    lines = [
        "PAPER 3 — contact completion, training report",
        f"finished   : {datetime.now():%Y-%m-%d %H:%M:%S}  ({dt/60:.1f} min)",
        f"rig        : {args.rig}   pairs {len(pairs)} from "
        f"{len({p.run for p in pairs})} runs",
        f"anchors    : {args.anchors}  ({_tag})   "
        f"train {len(tr)} pairs ({sum(1 for p in tr if _anch(p))} anchored), "
        f"val {len(va)} pairs (pt00 only, always)",
        f"split      : {len(tr)} train / {len(va)} val, BY RUN "
        f"(val: {val_runs})",
        f"model      : UNet base={args.base} depth={args.depth}, "
        f"{n_par/1e6:.2f} M params",
        f"epochs     : {args.epochs}   best at {best_ep} (loss {best:.4f})",
        f"input scale: {scale:.1f}",
        "",
        "MEASURED CELLS ONLY. ~60% of the canvas was never visited; those "
        "cells are 'nobody looked', not 'no contact', and are excluded from "
        "every number below.",
        "",
        f"{'':<12}{'L1':>10}{'SSIM':>10}{'TC mm':>10}",
        f"{'zero':<12}{ref['zero_l1']:>10.4f}{ref['zero_ssim']:>10.3f}"
        f"{'-':>10}",
        f"{'copy input':<12}{ref['copy_l1']:>10.4f}{ref['copy_ssim']:>10.3f}"
        f"{'-':>10}",
        f"{'model':<12}{got:>10.4f}"
        f"{final.get('va_ssim', final['tr_ssim']):>10.3f}"
        f"{final.get('va_tc_mm', final['tr_tc_mm']):>10.2f}",
        "",
        f"VERDICT: {verdict}",
    ]
    if not va:
        lines += ["", "NO VALIDATION SPLIT: one run only, so the numbers "
                      "above are training numbers and say nothing about "
                      "generalisation."]
    if len({p.run for p in tr}) < 3:
        lines += ["", f"{len({p.run for p in tr})} training run(s): this "
                      f"proves the pipeline runs, not that the method works."]
    rep = "\n".join(lines)
    with open(os.path.join(out_dir, f"report{_sfx}.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep)

    try:
        _plots(out_dir, hist, net, Xva if len(Xva) else Xtr,
               Yva if len(Xva) else Ytr, Mva if len(Xva) else Mtr, dev,
               scale, tag=_tag, sfx=_sfx)
    except Exception as e:
        print(f"(plots skipped: {e})")
    print(f"\nwrote {out_dir}")
    return 0


def _plots(out_dir, hist, net, X, Y, Mk, dev, scale, tag="",
           sfx=""):
    import torch
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    ep = [h["epoch"] for h in hist]
    fig = Figure(figsize=(11, 3.4)); FigureCanvasAgg(fig)
    for i, (keys, ttl) in enumerate(
            ((("train_loss", "va_l1"), "masked L1"),
             (("tr_ssim", "va_ssim"), "masked SSIM"),
             (("tr_tc_mm", "va_tc_mm"), "tactile centroid error (mm)"))):
        ax = fig.add_subplot(1, 3, i + 1)
        for k in keys:
            v = [h.get(k) for h in hist]
            if any(x is not None for x in v):
                ax.plot(ep, v, lw=1.2, label=k)
        ax.set_xlabel("epoch"); ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3, lw=0.4); ax.legend(fontsize=7)
    if tag:
        fig.suptitle(f"training curves — anchors: {tag}", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"curves{sfx}.png"), dpi=120,
                bbox_inches="tight")

    net.eval()
    with torch.no_grad():
        P = net(X.to(dev)).cpu().numpy()
    n = min(4, len(X))
    fig = Figure(figsize=(13, 3.2 * n)); FigureCanvasAgg(fig)
    for r in range(n):
        m = Mk[r, 0].numpy().astype(bool)
        for c, (img, ttl) in enumerate((
                (X[r, 0].numpy(), "input (pt00)"),
                (Y[r, 0].numpy(), "target (stitched)"),
                (P[r, 0], "prediction"),
                (np.abs(P[r, 0] - Y[r, 0].numpy()), "|error|"))):
            ax = fig.add_subplot(n, 4, r * 4 + c + 1)
            # masked cells hidden: showing the model's output where nothing
            # was measured invites reading it as if it had been checked
            show = np.ma.masked_where(~m, img) if c in (1, 3) else img
            ax.imshow(show, cmap="jet", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ttl, fontsize=9)
    if tag:
        fig.suptitle(f"validation predictions (pt00 only) — "
                     f"trained with anchors: {tag}", y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"preds{sfx}.png"), dpi=110,
                bbox_inches="tight")


if __name__ == "__main__":
    sys.exit(main())
