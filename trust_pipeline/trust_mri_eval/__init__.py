"""TRUST-MRI evaluation of imputation + segmentation.

The imputation codebase, BraSyn_tutorial (https://github.com/WinstonHuTiger/BraSyn_tutorial), is imported from
its clone in ``BRASYN_REPO`` (environment variable; default ``../external/BraSyn_tutorial`` from the run folder).
"""
import os
import sys
from pathlib import Path

BRASYN_REPO = Path(os.environ.get("BRASYN_REPO", "../external/BraSyn_tutorial"))
if not (BRASYN_REPO / "project").is_dir():
    raise ImportError(f"BraSyn_tutorial not found in {BRASYN_REPO.resolve()}: "
                      f"git clone https://github.com/WinstonHuTiger/BraSyn_tutorial {BRASYN_REPO}")
sys.path.insert(0, str(BRASYN_REPO.resolve() / "project"))
