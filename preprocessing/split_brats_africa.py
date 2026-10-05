#!/usr/bin/env python3
"""Shuffle and 50/50 train/test split of BraTS-Africa, each class folder split independently.

Always writes <out>/split.csv. With --mode symlink|copy, also builds <out>/{train,test}/<class>/<case>.
Usage:
    python split_brats_africa.py
    python split_brats_africa.py --root "/path/to/BraTS-Africa" --seed 42 --mode symlink
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

#write path to dataset
DEFAULT_ROOT = Path.home() / "software/BraTs Africa/BraTS-Africa Dataset/BraTS-Africa"
CLASS_FOLDERS = ["51_OtherNeoplasms", "95_Glioma"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, help="default: <root>/../BraTS-Africa_split")
    ap.add_argument("--seed", type=int, default=42) #keep seed for reproducibility
    ap.add_argument("--mode", choices=["symlink", "copy"])
    args = ap.parse_args()

    out = args.out or args.root.parent / "BraTS-Africa_split"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []
    for cls in CLASS_FOLDERS:
        cases = sorted(p for p in (args.root / cls).iterdir() if p.is_dir())
        rng.shuffle(cases)
        half = len(cases) // 2
        print(f"{cls}: {half} train / {len(cases) - half} test")

        for split, split_cases in (("train", cases[:half]), ("test", cases[half:])):
            for case in split_cases:
                rows.append([split, cls, case.name, case])
                dest = out / split / cls / case.name
                if args.mode and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if args.mode == "symlink":
                        dest.symlink_to(case.resolve())
                    else:
                        shutil.copytree(case, dest)

    with open(out / "split.csv", "w", newline="") as f:
        csv.writer(f).writerows([["split", "class", "case_id", "path"], *rows])
    print(f"Wrote {out / 'split.csv'} (seed={args.seed})")


if __name__ == "__main__":
    main()
