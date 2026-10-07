"""Command line entry point.

    python -m trust_mri_eval.cli --data-dir /path/to/cases
    python -m trust_mri_eval.cli --data-dir /path/to/cases --set modality=t1c region=TC save_maps=false
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
    ap.add_argument("--data-dir", type=Path, required=True, help="folder searched for the case files (see data/io.py)")
    ap.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="override any Config field, value parsed as JSON (e.g. Cs=[0.1,1] penalty=l1)")
    args = ap.parse_args(argv)

    return Config(data_dir=args.data_dir).override(args.set)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    out = run(build_config(argv))
    logging.getLogger("trust_mri_eval").info("done in %.0fs -> %s", time.time() - t0, out)


if __name__ == "__main__":
    main()
