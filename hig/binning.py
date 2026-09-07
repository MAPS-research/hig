"""Bin definitions: what a histogram counts, and what it costs to move mass.

A binning answers three questions for the OT layer:

* which bin does each pixel currently occupy (``units``),
* what does it cost to move a pixel from one place to a target bin (``cost``),
* which colour do we write when that move is made (``paint``).

Two schemes ship with HIG, differing in whether a bin is a single colour range
or a set of interchangeable candidates:

``GridBinning``
    Uniform quantisation of the selected channels. Bin == colour cube, so a
    reassigned pixel takes that cube's centre. Source and target live on the
    same grid.

``MultiOptionBinning``
    Full 8-bit resolution on two channels, randomly partitioned into ``num_bins``
    groups of ``k`` colours. Only the per-bin total is constrained, so a
    reassigned pixel takes whichever of its target bin's ``k`` candidates is
    closest to what it already was -- far less visible distortion for the same
    histogram constraint. Source units are the fine colours, so the cost matrix
    is rectangular.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Binning", "GridBinning", "MultiOptionBinning"]

_CHANNELS = "rgb"


def _channel_indices(channels: str) -> list[int]:
    if not channels or any(c not in _CHANNELS for c in channels):
        raise ValueError(f"channels must be a non-empty subset of 'rgb', got {channels!r}")
    if len(set(channels)) != len(channels):
        raise ValueError(f"channels must not repeat, got {channels!r}")
    return [_CHANNELS.index(c) for c in channels]


def _as_pixels(image: np.ndarray, channels: list[int]) -> np.ndarray:
    """(H, W, 3) uint8 image -> (H*W, len(channels)) uint8 view of the channels
    that participate in the histogram."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {image.dtype}")
    return image.reshape(-1, 3)[:, channels]


class Binning(ABC):
    """Bins over the colour space, plus the transport cost between them."""

    num_bins: int
    num_units: int
    channels: str

    @property
    def channel_indices(self) -> list[int]:
        return _channel_indices(self.channels)

    @abstractmethod
    def units(self, image: np.ndarray) -> np.ndarray:
        """Source unit index per pixel, flattened row-major. ``(H*W,) int32``."""

    @abstractmethod
    def cost(self, unit_ids: np.ndarray) -> np.ndarray:
        """``(len(unit_ids), num_bins) float64`` transport cost.

        Only the requested rows are built: units carrying no mass cannot appear
        in the plan, and dropping them shrinks the LP without changing its
        solution.
        """

    @abstractmethod
    def paint(self, unit_ids: np.ndarray, bin_ids: np.ndarray) -> np.ndarray:
        """Channel values to write when unit ``u`` is reassigned to bin ``j``.
        ``(N, len(channels)) uint8``."""

    @abstractmethod
    def unit_to_bin(self, unit_ids: np.ndarray) -> np.ndarray:
        """Which bin each source unit belongs to. The identity for
        :class:`GridBinning`; the random partition for
        :class:`MultiOptionBinning`."""

    # -- shared helpers ----------------------------------------------------

    def histogram(self, image: np.ndarray) -> np.ndarray:
        """Pixel count per source unit. ``(num_units,) int64``."""
        return np.bincount(self.units(image), minlength=self.num_units).astype(np.int64)

    def bin_histogram(self, image: np.ndarray) -> np.ndarray:
        """Pixel count per *bin* -- the quantity HIG actually constrains.
        ``(num_bins,) int64``."""
        bins = self.unit_to_bin(self.units(image))
        return np.bincount(bins, minlength=self.num_bins).astype(np.int64)

    def apply(self, image: np.ndarray, unit_ids: np.ndarray, bin_ids: np.ndarray) -> np.ndarray:
        """Write ``paint(unit_ids, bin_ids)`` back into a copy of ``image``.

        ``unit_ids``/``bin_ids`` are per-pixel and flattened row-major.
        Channels outside ``self.channels`` are left untouched -- that is what
        keeps a two-channel constraint from flattening the image.
        """
        out = image.copy()
        flat = out.reshape(-1, 3)
        flat[:, self.channel_indices] = self.paint(unit_ids, bin_ids)
        return out


class GridBinning(Binning):
    """Uniform ``levels``-way quantisation per channel; bin == colour cube."""

    def __init__(self, channels: str = "rgb", levels: int = 16):
        idx = _channel_indices(channels)
        if 256 % levels or levels < 2:
            raise ValueError(f"levels must be a power-of-two divisor of 256, got {levels}")
        self.channels = channels
        self.levels = levels
        self.num_bins = levels ** len(idx)
        self.num_units = self.num_bins
        self._idx = idx
        self._step = 256 // levels
        # radix weights, most significant channel first, matching `channels`
        self._radix = np.array(
            [levels ** (len(idx) - 1 - k) for k in range(len(idx))], dtype=np.int64
        )
        # digit -> the 8-bit value written for that digit, chosen so that
        # re-quantising the written value lands back on the same digit
        self._value = np.round(np.arange(levels) * 255.0 / (levels - 1)).astype(np.uint8)
        digits = np.stack(
            np.meshgrid(*([np.arange(levels)] * len(idx)), indexing="ij"), axis=-1
        ).reshape(-1, len(idx))
        self._digits = digits.astype(np.float64)

    def units(self, image: np.ndarray) -> np.ndarray:
        q = _as_pixels(image, self._idx).astype(np.int64) // self._step
        return (q * self._radix).sum(axis=1).astype(np.int32)

    def unit_to_bin(self, unit_ids: np.ndarray) -> np.ndarray:
        return np.asarray(unit_ids, dtype=np.int32)

    def cost(self, unit_ids: np.ndarray) -> np.ndarray:
        src = self._digits[np.asarray(unit_ids, dtype=np.int64)]
        diff = src[:, None, :] - self._digits[None, :, :]
        return np.einsum("ijk,ijk->ij", diff, diff)

    def paint(self, unit_ids: np.ndarray, bin_ids: np.ndarray) -> np.ndarray:
        digits = self._digits[np.asarray(bin_ids, dtype=np.int64)].astype(np.int64)
        return self._value[digits]


class MultiOptionBinning(Binning):
    """``num_bins`` random groups of ``k`` full-resolution colours on two channels.

    Only the per-bin total is constrained, so the transport cost from a colour
    to a bin is the distance to that bin's *nearest* member, and a reassigned
    pixel moves to exactly that member. With the candidates scattered uniformly
    over the colour square, every bin is cheaply reachable from anywhere, which
    is what keeps an exactly-matched histogram from looking quantised.
    """

    def __init__(
        self,
        channels: str = "rg",
        num_bins: int = 4096,
        unit_bits: int = 8,
        seed: int = 0,
    ):
        idx = _channel_indices(channels)
        if len(idx) != 2:
            raise ValueError(f"multi-option binning is defined on two channels, got {channels!r}")
        if not 1 <= unit_bits <= 8:
            raise ValueError(f"unit_bits must be in 1..8, got {unit_bits}")
        side = 1 << unit_bits
        num_units = side * side
        if num_units % num_bins:
            raise ValueError(
                f"{num_units} colours do not divide evenly into {num_bins} bins; "
                "pick a power-of-two num_bins"
            )

        self.channels = channels
        self.num_bins = num_bins
        self.num_units = num_units
        self.options = num_units // num_bins
        self.unit_bits = unit_bits
        self.seed = seed
        self._idx = idx
        self._shift = 8 - unit_bits
        self._side = side

        # A fixed permutation is the whole design: bins must be scattered, and
        # they must be reproducible from the seed alone so nothing large ships
        # with the repo.
        perm = np.random.default_rng(seed).permutation(num_units).astype(np.int32)
        self._unit_to_bin = np.empty(num_units, dtype=np.int32)
        self._unit_to_bin[perm] = np.repeat(np.arange(num_bins, dtype=np.int32), self.options)
        # (num_bins, options) unit ids, and their channel values
        self._bin_to_unit = perm.reshape(num_bins, self.options)
        self._unit_value = np.stack(
            [np.arange(num_units) // side, np.arange(num_units) % side], axis=-1
        ).astype(np.int16)
        self._bin_value = self._unit_value[self._bin_to_unit]  # (num_bins, options, 2)

    def units(self, image: np.ndarray) -> np.ndarray:
        px = _as_pixels(image, self._idx).astype(np.int32) >> self._shift
        return (px[:, 0] * self._side + px[:, 1]).astype(np.int32)

    def unit_to_bin(self, unit_ids: np.ndarray) -> np.ndarray:
        return self._unit_to_bin[np.asarray(unit_ids, dtype=np.int64)]

    # every L1 distance on a 256-wide colour grid is at most 2 * 255
    _UNREACHABLE = 1000

    def cost(self, unit_ids: np.ndarray) -> np.ndarray:
        """L1 distance from each requested colour to its nearest member of each bin."""
        table = self._distance_table()
        return table[:, np.asarray(unit_ids, dtype=np.int64)].T.astype(np.float64)

    def _distance_table(self) -> np.ndarray:
        """``(num_bins, num_units) int16`` distance to each bin's nearest member.

        Computing this pair by pair costs ``units x bins x options``, which is
        minutes at 8-bit resolution. But "distance to the nearest of a set of
        seeds" is a multi-source L1 distance transform on the colour grid, and
        the L1 transform is separable -- one sweep per axis, both of them a
        running minimum. Same answer, two orders of magnitude cheaper.

        Held on the instance: it depends only on (seed, num_bins, unit_bits) and
        costs ~0.5 GB at 8-bit, so it is built once and reused across steps.
        """
        if getattr(self, "_table", None) is None:
            side, bins = self._side, self.num_bins
            grid = np.full((bins, side, side), self._UNREACHABLE, dtype=np.int16)
            seeds = self._bin_value  # (bins, options, 2)
            grid[
                np.repeat(np.arange(bins), self.options),
                seeds[..., 0].ravel(),
                seeds[..., 1].ravel(),
            ] = 0
            for axis in (2, 1):
                grid = self._sweep(grid, axis)
            self._table = grid.reshape(bins, -1)
        return self._table

    @staticmethod
    def _sweep(grid: np.ndarray, axis: int) -> np.ndarray:
        """One exact 1-D L1 pass along ``axis``.

        ``h[i] = min_k (h[k] + |i - k|)`` splits into a forward and a backward
        half, and each half is a running minimum once the index is folded in:
        ``min_{k<=i} (h[k] + i - k) = i + min-accumulate(h - k)``. No Python
        loop over the axis.
        """
        n = grid.shape[axis]
        shape = [1] * grid.ndim
        shape[axis] = n
        index = np.arange(n, dtype=grid.dtype).reshape(shape)
        reverse = tuple(
            slice(None, None, -1) if d == axis else slice(None) for d in range(grid.ndim)
        )

        forward = np.minimum.accumulate(grid - index, axis=axis)
        forward += index
        backward = np.minimum.accumulate((grid + index)[reverse], axis=axis)[reverse]
        backward -= index
        return np.minimum(forward, backward, out=forward)

    def paint(self, unit_ids: np.ndarray, bin_ids: np.ndarray) -> np.ndarray:
        """Nearest candidate in the target bin -- computed for the pairs that
        actually occur, so no (units x bins) table is ever materialised."""
        src = self._unit_value[np.asarray(unit_ids, dtype=np.int64)]  # (N, 2)
        options = self._bin_value[np.asarray(bin_ids, dtype=np.int64)]  # (N, options, 2)
        pick = np.abs(options - src[:, None, :]).sum(axis=-1).argmin(axis=1)
        chosen = np.take_along_axis(options, pick[:, None, None], axis=1)[:, 0, :]
        return chosen.astype(np.uint8) << self._shift
