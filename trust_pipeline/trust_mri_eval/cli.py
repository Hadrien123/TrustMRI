"""Command line entry point.

    python -m trust_mri_eval.cli --synthetic
    python -m trust_mri_eval.cli --synthetic --shape 96 96 64 --set n_patients=6 split=[3,1,2]
    python -m trust_mri_eval.cli --data-dir /path/to/patients --regions ET TC WT
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .config import Config
from .pipeline import run


def build_config(argv: list[str] | None = None) -> Config:
    """Parse arguments into a :class:`Config`."""
    ap = argparse.ArgumentParser(prog="trust_mri_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="run on generated synthetic patients")
    src.add_argument("--data-dir", type=Path, help="folder with one sub-folder per patient (see data/io.py)")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--run-id", default="", help="output sub-folder name (default: timestamp)")
    ap.add_argument("--shape", type=int, nargs=3, metavar=("X", "Y", "Z"), help="synthetic volume shape")
    ap.add_argument("--regions", nargs="+", help="tumour regions to evaluate (default: ET)")
    ap.add_argument("--no-maps", action="store_true", help="do not write NIfTI maps")
    ap.add_argument("--no-figures", action="store_true", help="do not write PNG figures")
    ap.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="override any Config field, value parsed as JSON (e.g. Cs=[0.1,1] penalty=l1)")
    args = ap.parse_args(argv)

    cfg = Config(synthetic=args.synthetic, data_dir=args.data_dir, output_dir=args.output_dir, run_id=args.run_id,
                 save_maps=not args.no_maps, save_figures=not args.no_figures)
    if args.shape:
        cfg.volume_shape = tuple(args.shape)
    if args.regions:
        cfg.regions = tuple(args.regions)
    return cfg.override(args.set)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    out = run(build_config(argv))
    logging.getLogger("trust_mri_eval").info("done in %.0fs -> %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
