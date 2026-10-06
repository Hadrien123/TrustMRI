"""All pipeline parameters in one dataclass.

Metric, uncertainty and model functions receive plain arrays and numbers; only ``pipeline.py``
and ``cli.py`` read the :class:`Config` object. Override fields from the CLI with ``--set name=value``.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Shape3 = tuple[int, int, int]


@dataclass
class Config:
    # ---------------------------------------------------------------- run / paths
    data_dir: Path | None = None           # folder searched for the case files (see data/io.py)
    modality: str | None = None            # imputed sequence (t1c, t1n, t2f, t2w); None = the only one imputed
    region: str = "WT"                     # tumour region of the segmentations: WT, TC or ET
    run_id: str = ""                       # empty -> timestamp
    output_dir: Path = Path("outputs")
    seed: int = 0

    # ---------------------------------------------------------------- preprocessing
    roi_dilation: int = 5
    seg_threshold: float = 0.5

    # ---------------------------------------------------------------- quality (one output)
    eval_imputation: int = 0               # the imputed image n evaluated against the ground truth
    eval_prompt: int = 0                   # the segmentation (n, k) evaluated against the reference
    patch_size: Shape3 = (3, 3, 3)

    # ---------------------------------------------------------------- trust (all N x K outputs)
    anova_chunk: int = 200_000
    min_patch_coverage: float = 0.5        # ROI fraction of a patch scored by the confidence model
    label_tolerance: float = 1.0           # voxels within this distance of the GT boundary count as correct
    label_min_fraction: float = 0.9        # a patch is correct if this fraction of its voxels is correct
    test_fraction: float = 0.25            # patients held out to evaluate the confidence model
    penalty: str = "l2"                    # "l2" or "l1"
    Cs: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    cv_folds: int = 5

    # ---------------------------------------------------------------- outputs
    save_maps: bool = True
    save_figures: bool = True

    # ----------------------------------------------------------------
    def override(self, assignments: list[str]) -> "Config":
        """Return a copy with ``name=value`` overrides (values parsed as JSON when possible)."""
        types = {f.name: f.type for f in dataclasses.fields(self)}
        changes: dict[str, Any] = {}
        for item in assignments:
            name, _, raw = item.partition("=")
            name = name.strip()
            if name not in types:
                raise KeyError(f"unknown config field: {name}")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            changes[name] = _coerce(getattr(self, name), value, "Path" in str(types[name]))
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the configuration."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(self).items()}


def _coerce(current: Any, value: Any, is_path: bool) -> Any:
    """Cast an override to the type of the field (tuples, paths, numbers)."""
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if is_path and isinstance(value, str):
        return Path(value)
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, float) and isinstance(value, int):
        return float(value)
    return value
