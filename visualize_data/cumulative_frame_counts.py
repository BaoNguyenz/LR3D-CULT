#!/usr/bin/env python3
"""
Compute cumulative distribution of frames per object.

For k = 2..180, count how many objects have num_frames >= k.

Usage:
    python cumulative_frame_stats.py "E:/LET_ME_COOK/Captone/PixelNerf_finetuning/dataset_pottery" --out "E:/reports"

Outputs:
  - CSV: cumulative_frame_counts_2_180.csv with columns: threshold,num_objects
  - CSV: frame_counts_per_object.csv with each object's exact frame count
  - Prints a small pretty table to stdout
"""
import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv

def find_transforms(root: Path):
    tf_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden dirs
        if any(p.startswith('.') for p in Path(dirpath).parts):
            continue
        for fn in filenames:
            if fn.lower().startswith("transforms") and fn.lower().endswith(".json"):
                tf_paths.append(Path(dirpath) / fn)
    return tf_paths

def read_num_frames(tf_path: Path):
    try:
        with tf_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"ERR_LOAD:{e}"
    frames = data.get("frames")
    if frames is None:
        return None, "NO_FRAMES_KEY"
    return len(frames), None

def main(root_dir: str, out_dir: str = None, min_threshold=2, max_threshold=180, verbose=True):
    root = Path(root_dir)
    if not root.exists():
        print("Root path not found:", root)
        return 2

    out = Path(out_dir) if out_dir else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    tf_files = find_transforms(root)
    if verbose:
        print(f"Found {len(tf_files)} transforms*.json files under {root}")

    # We aggregate by object folder; if multiple transforms files in same folder, take max frames
    folder_to_max = {}
    errors = []

    for tf in tf_files:
        n, err = read_num_frames(tf)
        folder = str(tf.parent)
        if err:
            errors.append((str(tf), err))
            continue
        prev = folder_to_max.get(folder)
        if prev is None or n > prev:
            folder_to_max[folder] = n

    # Build per-object list
    per_obj = sorted(folder_to_max.items(), key=lambda x: (-x[1], x[0]))  # (folder, num_frames)
    # Save per-object CSV
    per_obj_csv = out / "frame_counts_per_object.csv"
    with per_obj_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "num_frames"])
        for obj, n in per_obj:
            writer.writerow([obj, n])
    if verbose:
        print("Saved per-object frame counts to", per_obj_csv)

    # Build counts dict: exact counts -> number of objects
    counts = Counter([n for (_obj, n) in per_obj])
    total_objects = len(per_obj)

    # Compute cumulative for thresholds min_threshold..max_threshold
    thresholds = list(range(min_threshold, max_threshold+1))
    cumulative_results = []
    # To compute efficiently, we can precompute prefix sums:
    # For k from high to low, maintain running sum of counts for >=k
    max_seen = max(counts.keys()) if counts else 0
    running = 0
    # create array of counts up to max(max_threshold, max_seen)
    top = max(max_threshold, max_seen)
    counts_arr = [counts.get(i, 0) for i in range(top+1)]  # index = frames
    # compute cumulative from top down
    cum_arr = [0] * (top+1)
    s = 0
    for i in range(top, -1, -1):
        s += counts_arr[i]
        cum_arr[i] = s
    for k in thresholds:
        num_objs_ge_k = cum_arr[k] if k <= top else 0
        cumulative_results.append((k, num_objs_ge_k))

    # Save cumulative CSV (2..180)
    cum_csv = out / f"cumulative_frame_counts_{min_threshold}_{max_threshold}.csv"
    with cum_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "num_objects_ge_threshold", "total_objects"])
        for k, v in cumulative_results:
            writer.writerow([k, v, total_objects])
    if verbose:
        print("Saved cumulative CSV to", cum_csv)

    # Print a compact table to stdout
    if verbose:
        print("\nCumulative counts (threshold -> objects with >= threshold frames):")
        print(f"{'threshold':>10}  {'num_objects':>12}  {'percent_of_dataset':>18}")
        for k, v in cumulative_results:
            pct = (100.0 * v / total_objects) if total_objects > 0 else 0.0
            print(f"{k:10d}  {v:12d}  {pct:17.2f}%")

    # If any errors, print short sample
    if errors:
        print(f"\nNote: {len(errors)} transforms files had errors (showing up to 10):")
        for p, e in errors[:10]:
            print(" ", p, e)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root folder containing objects (each object: has transforms.json or transforms_*.json)")
    parser.add_argument("--out", "-O", help="Output folder for CSV (default cwd)", default="./stats_output")
    parser.add_argument("--min", type=int, default=2, help="Minimum threshold (default 2)")
    parser.add_argument("--max", type=int, default=180, help="Maximum threshold (default 180)")
    parser.add_argument("--no-print", dest="verbose", action="store_false", help="Do not print table to stdout")
    args = parser.parse_args()
    exit(main(args.root, args.out, min_threshold=args.min, max_threshold=args.max, verbose=args.verbose))
