"""Non-overlapping patch grid shared by local metrics, features and labels.

A :class:`PatchGrid` tiles a box of a *parent* volume with patches of size ``patch``. The box start
is snapped to a multiple of the patch size, so grids built over different boxes of the same parent
are aligned on one global lattice. Patches that stick out of the parent are padded; padded voxels
never count (zero weight).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Shape3 = tuple[int, int, int]


@dataclass(frozen=True)
class PatchGrid:
    """Patch lattice over ``box`` of a volume of shape ``parent_shape``."""

    parent_shape: Shape3
    patch: Shape3
    start: Shape3
    n: Shape3

    @classmethod
    def over(cls, parent_shape: Shape3, patch: Shape3,
             box: tuple[slice, slice, slice] | None = None) -> "PatchGrid":
        """Grid covering ``box`` (default: the whole parent), aligned to the global lattice."""
        lo = [0, 0, 0] if box is None else [s.start or 0 for s in box]
        hi = list(parent_shape) if box is None else [s.stop for s in box]
        start = tuple((a // p) * p for a, p in zip(lo, patch))
        n = tuple(max(1, math.ceil((b - s) / p)) for b, s, p in zip(hi, start, patch))
        return cls(tuple(parent_shape), tuple(patch), start, n)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ geometry
    @property
    def box(self) -> tuple[slice, slice, slice]:
        """Slices of the parent covered by the grid (clipped to the parent)."""
        return tuple(slice(s, min(s + k * p, d))  # type: ignore[return-value]
                     for s, k, p, d in zip(self.start, self.n, self.patch, self.parent_shape))

    @property
    def box_shape(self) -> Shape3:
        return tuple(b.stop - b.start for b in self.box)  # type: ignore[return-value]

    @property
    def voxels_per_patch(self) -> int:
        return int(np.prod(self.patch))

    def crop(self, vol: np.ndarray) -> np.ndarray:
        """Box view of a parent-shaped array (leading dimensions are kept)."""
        return vol[(..., *self.box)]

    # ------------------------------------------------------------------ reductions
    def _as_box(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[-3:] == self.box_shape:
            return arr
        if arr.shape[-3:] == self.parent_shape:
            return self.crop(arr)
        raise ValueError(f"array shape {arr.shape} matches neither box {self.box_shape} "
                         f"nor parent {self.parent_shape}")

    def _blocks(self, arr: np.ndarray) -> np.ndarray:
        arr = self._as_box(arr)
        pad = [(0, k * p - b) for k, p, b in zip(self.n, self.patch, self.box_shape)]
        if any(after for _, after in pad):
            arr = np.pad(arr, [(0, 0)] * (arr.ndim - 3) + pad)
        lead = arr.shape[:-3]
        (nx, ny, nz), (px, py, pz) = self.n, self.patch
        return arr.reshape(*lead, nx, px, ny, py, nz, pz)

    def sum(self, arr: np.ndarray) -> np.ndarray:
        """Per-patch sum, shape ``(*lead, nx, ny, nz)``."""
        b = self._blocks(arr)
        return b.sum(axis=(-5, -3, -1), dtype=np.float64)

    def coverage(self, mask: np.ndarray) -> np.ndarray:
        """Fraction of each patch's voxels inside ``mask`` (padding counts as outside)."""
        return (self.sum(mask.astype(bool)) / self.voxels_per_patch).astype(np.float32)

    def mean(self, arr: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Per-patch (weighted) mean; NaN where the weights sum to zero.

        ``weights`` defaults to "inside the parent", so padded voxels are ignored.
        """
        if weights is None:
            weights = np.ones(self.box_shape, dtype=np.float32)
        w = self._as_box(weights).astype(np.float32)
        num = self.sum(self._as_box(arr) * w)
        den = self.sum(w)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        return np.where(den > 0, out, np.nan).astype(np.float32)

    # ------------------------------------------------------------------ back to voxels
    def expand(self, values: np.ndarray, fill: float = 0.0, dtype: np.dtype = np.float32) -> np.ndarray:
        """Paint per-patch ``values`` (nx, ny, nz) back into a parent-shaped volume."""
        if values.shape != self.n:
            raise ValueError(f"expected patch values of shape {self.n}, got {values.shape}")
        full = values
        for axis, p in enumerate(self.patch):
            full = np.repeat(full, p, axis=axis)
        full = full[tuple(slice(0, s) for s in self.box_shape)]
        out = np.full(self.parent_shape, fill, dtype=dtype)
        out[self.box] = np.nan_to_num(full, nan=fill)
        return out


def bounding_box(mask: np.ndarray, margin: int = 0) -> tuple[slice, slice, slice]:
    """Smallest box containing ``mask`` (whole volume if empty), grown by ``margin`` voxels."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return tuple(slice(0, d) for d in mask.shape)  # type: ignore[return-value]
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def grow_box(box: tuple[slice, ...], margin: int, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Grow a box by ``margin`` voxels on each side, clipped to ``shape``."""
    return tuple(slice(max(s.start - margin, 0), min(s.stop + margin, d)) for s, d in zip(box, shape))
