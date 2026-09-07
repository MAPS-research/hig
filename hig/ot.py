"""Histogram matching as an optimal-transport problem.

The constraint HIG enforces is a *count*: bin ``j`` must hold exactly
``target[j]`` pixels. Given the image's current counts, the cheapest way to get
there is a transport plan -- and because the network simplex minimises total
movement, the image changes as little as the constraint allows. That is the
whole reason the intervention survives the remaining denoising steps instead of
being sanded off as damage.

Integer marginals matter. The LP is totally unimodular, so integer supplies and
demands give an integer plan, which is what lets the resulting histogram match
the target exactly rather than approximately -- the property the information
embedding application depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot as pot

from hig.binning import Binning

__all__ = ["to_counts", "solve_plan", "HistogramMatcher"]


def to_counts(weights: np.ndarray, total: int) -> np.ndarray:
    """Non-negative weights -> integer counts summing to exactly ``total``.

    Largest-remainder apportionment: floor everything, then hand the leftover
    units to the largest fractional parts. Ties break by index so the result is
    a deterministic function of the input.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"expected a 1-D histogram, got shape {w.shape}")
    if np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("weights sum to zero; nothing to distribute")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")

    exact = w / w.sum() * total
    counts = np.floor(exact).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder:
        frac = exact - counts
        # stable sort keeps ties in index order -> deterministic
        winners = np.argsort(-frac, kind="stable")[:remainder]
        counts[winners] += 1
    assert counts.sum() == total
    return counts


def solve_plan(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    cost: np.ndarray,
    num_iter_max: int = 1_000_000,
) -> np.ndarray:
    """Exact OT plan between two integer histograms. ``(n_src, n_tgt) int64``."""
    src = np.asarray(source_counts, dtype=np.int64)
    tgt = np.asarray(target_counts, dtype=np.int64)
    if src.sum() != tgt.sum():
        raise ValueError(f"marginals must agree: source has {src.sum()}, target {tgt.sum()}")
    if cost.shape != (len(src), len(tgt)):
        raise ValueError(f"cost must be {(len(src), len(tgt))}, got {cost.shape}")

    plan = pot.emd(
        src.astype(np.float64),
        tgt.astype(np.float64),
        np.ascontiguousarray(cost, dtype=np.float64),
        numItermax=num_iter_max,
    )
    # Totally unimodular LP + integer marginals => integer vertex solution.
    # Round, then insist on it: a silent fractional plan would quietly break the
    # exactness guarantee downstream.
    rounded = np.rint(plan).astype(np.int64)
    if np.abs(plan - rounded).max() > 1e-6:
        raise RuntimeError("OT solver returned a fractional plan; marginals may be malformed")
    if not (rounded.sum(axis=1) == src).all() or not (rounded.sum(axis=0) == tgt).all():
        raise RuntimeError("OT plan does not satisfy its marginals; try raising num_iter_max")
    return rounded


@dataclass
class HistogramMatcher:
    """Transport an image's colour histogram onto ``target``.

    ``target`` is a weight vector over the binning's bins; it is apportioned to
    the image's pixel count on every call, so the same matcher works on any
    resolution.

    ``seed`` fixes which pixels within a bin get moved. Reproducibility is a
    hard requirement for the embedding application, so the choice goes through
    an explicit generator and never touches global RNG state.
    """

    binning: Binning
    target: np.ndarray
    seed: int = 0

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float64)
        if self.target.shape != (self.binning.num_bins,):
            raise ValueError(
                f"target must have {self.binning.num_bins} bins, got {self.target.shape}"
            )

    def integer_target(self, n_pixels: int) -> np.ndarray:
        """The exact per-bin counts this matcher enforces at a given pixel count."""
        return to_counts(self.target, n_pixels)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        binning = self.binning
        units = binning.units(image)
        n_pixels = units.size

        source = np.bincount(units, minlength=binning.num_units).astype(np.int64)
        occupied = np.flatnonzero(source)
        target = self.integer_target(n_pixels)

        plan = solve_plan(source[occupied], target, binning.cost(occupied))

        # Rows come back in ascending source order, so repeating each target bin
        # by its transported count lines the plan up with the pixels once they
        # are grouped by source unit.
        _, cols = np.nonzero(plan)
        moved = plan[plan > 0]
        assignment = np.repeat(cols.astype(np.int32), moved)

        # Shuffle first, then a stable sort by unit: pixels end up grouped by
        # unit with a seeded random order inside each group.
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(n_pixels)
        order = shuffled[np.argsort(units[shuffled], kind="stable")]

        target_bin = np.empty(n_pixels, dtype=np.int32)
        target_bin[order] = assignment
        return binning.apply(image, units, target_bin)
