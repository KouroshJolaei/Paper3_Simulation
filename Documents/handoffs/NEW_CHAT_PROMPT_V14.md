# Starter prompt for the next Claude chat

Copy everything below into the first message of the new chat, and upload the
files listed at the bottom alongside it.

---

I'm Kourosh, PhD candidate at CoRo Lab, ÉTS Montréal, working on Paper 3: a
learned tactile contact-completion model trained on Isaac Sim data, replacing
the hand-crafted shape-classifier extrapolation of Papers 1 and 2.

**Read PAPER3_HANDOFF_v14_0.pdf FIRST and treat it as authoritative** —
especially §7 (what is left), §10 (what changed since v13.0) and §11 (how I
work). PAPER3_HANDOFF_v13_0.md is the immediate predecessor and is still
useful for the hold-window bug, the residual guard, the calibration redo and
the real-robot collector; where the two disagree, v14.0 wins. Note the
numbering on my disk drifted: `PAPER3_HANDOFF_v11_2.pdf` is internally titled
"v13.0" — go by file name.

**Where I am.** Block 2 (stitching) is closed. Calibration is redone on one
uniform basis: 7 keys, deformation target 1.40 mm. Twelve upright sweeps are
collected (6 cuboid, 6 cylinder) and **not yet analysed**. A rolled 90°
cylinder batch on 5 diameters was started and may be finished. The real UR5e
is a working collector in free air — verified frame, pad-to-pad motion,
measured arrival guard, constructed vertical home, tilt gate, cuboid support,
and a Real Robot tab in the GUI — but real tactile data is blocked because one
pad is missing from the lab, and no real-rig TOOL_OFFSET_Z has been measured.

**I've been away for two weeks doing Paper 2 reviewer responses**, so
expect me to have forgotten details. Block 3 (U-Net training) has still not
started and is the priority.

**Before trusting anything about the code**, verify file freshness by grepping
for the signatures in §2 of the handoff.

**Two things I'd like to start with:** analysing the collected sweeps as a
set, then Block 3. But ask me what actually finished while I was away before
assuming anything.

**How I work.** One verifiable step at a time; measure, never guess; distances
in mm; complete files, never diffs; commands on one line; short answers,
usually one line per question unless I ask for more. When my read of the Isaac
Sim window contradicts a diagnostic, interrogate the diagnostic — that
instinct has been right every time. Bad data must refuse loudly rather than
degrade silently. And compute numbers before asserting them rather than
working them out in your head.

**Two questions to start: do you have the full picture, and is there anything
else you need to see? One line each.**

---

## Files to upload with that message

**Handoffs** — the current one plus history:

- `PAPER3_HANDOFF_v14_0.pdf` (current, authoritative)
- `PAPER3_HANDOFF_v13_0.md`
- `PAPER3_HANDOFF_v12_0.pdf`
- `PAPER3_HANDOFF_v11_2.pdf`
- `PAPER3_HANDOFF_v10_0.md`
- `PAPER3_HANDOFF_v9_0.pdf`

**Papers** — needed for Block 3 and Block 4 framing:

- `Paper_1.pdf`
- `Paper_2.pdf`
- `Berith_s_paper.pdf` (the simulated TSF-85 sensor and its CNN)

**Sim code** — the current versions:

- `main_gui.py`
- `sim/collect_from_config.py`
- `viz/stitching.py`
- `viz/grid_quality.py`
- `viz/grid_accuracy.py`
- `viz/heatmaps.py`
- `viz/blob_axis.py`
- `viz/run_manifest.py`
- `examples/run_batch.py`

**Real-robot code** — only if the conversation turns to the real rig:

- `Real_Robot/collect_real.py`
- `Real_Robot/Manual.odt` (lab cell start-up)

**Block 3 code** — upload these when you start training, not before:

- `dataset.py`, `model.py`, `train.py`

**Calibration and provenance** — small and worth having:

- `Data/pad_offset_calibration_contact.json`
- `Data/pad_offset_calibration_fixed.json`
- `Data/finger_throat.json`

**Run outputs** — share per run, when asking about a specific sweep, rather
than all at once:

- `gui_config_used.json`, `design_grid_report.txt`, `execution_ledger.json`,
  `pose_history.json`
- `stitch_report.txt`, `stitched_s1.png`
- `grid_accuracy.txt`, `grid_quality_s1.txt`, `grid_quality_s2.txt`
- `run_manifest.txt`
- two heatmaps (pt00 and one edge-straddling point)
- one training-pair `.npz` from an edge anchor
