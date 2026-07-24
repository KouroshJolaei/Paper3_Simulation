"""
diag7_show_contact.py — model-free raw contact dump.

For each grasp in a run folder, prints the RAW 7x4 hold-average map for
s1 and s2 (NO mirroring, NO flips, NO model) plus the hot row / hot
column index. Reason from the numbers.

Usage (normal python, NOT isaac):
  python3 diag7_show_contact.py <run_dir>
"""
import sys, os, glob
import numpy as np

def hold_average(csv_path):
    """Mean 7x4 map over frames where taxel sum >= 0.5 * peak sum."""
    rows = []
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 30:
                continue
            try:
                rows.append([float(v) for v in parts[2:30]])
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.array(rows)                  # (frames, 28)
    sums = arr.sum(axis=1)
    peak = sums.max()
    hold = arr[sums >= 0.5 * peak]
    return hold.mean(axis=0).reshape(7, 4)   # NEVER transpose

def main(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "*_pt*_s1_tactile_maps.csv")))
    if not files:
        print(f"no *_pt*_s1_tactile_maps.csv in {run_dir}")
        return
    for f1 in files:
        f2 = f1.replace("_s1_", "_s2_")
        tag = os.path.basename(f1).split("_s1_")[0]
        print("=" * 60)
        print(tag)
        for name, path in (("s1", f1), ("s2", f2)):
            if not os.path.exists(path):
                print(f"  {name}: FILE MISSING"); continue
            m = hold_average(path)
            if m is None:
                print(f"  {name}: no data"); continue
            col_sums = m.sum(axis=0)      # 4 columns
            row_sums = m.sum(axis=1)      # 7 rows
            print(f"  {name}  RAW 7x4 (row 0 = first row in csv):")
            for r in range(7):
                print("    " + " ".join(f"{v:7.1f}" for v in m[r]))
            print(f"  {name}  col sums: " +
                  " ".join(f"{v:8.1f}" for v in col_sums) +
                  f"   HOT COL = {int(np.argmax(col_sums))}")
            print(f"  {name}  row sums hot ROW = {int(np.argmax(row_sums))}")
        print()

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    main(run)