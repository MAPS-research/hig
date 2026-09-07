"""Soft prompt <-> colour histogram.

The information-embedding application tunes a single soft prompt token so that
an LLM greedily reproduces a target sentence, then stores that vector in an
image's colour histogram. The two live in the same space by construction: the
histogram has one bin per hidden dimension.

Encoding is a softmax. Decoding is its inverse -- but a softmax destroys one
degree of freedom, since ``emb`` and ``emb + c`` give the same distribution. The
constant is recovered by taking the *minimum-norm* representative of the shift
class, which is what training targets anyway: the tuning loop reprojects the
vector to a fixed norm at every step, and among all shifts of the same
distribution the shortest is the centred one.

That minimiser is available in closed form. Minimising ``||log(s*p)||^2`` over
``log s`` gives ``log s = -mean(log p)``, so decoding is just centring the log
histogram -- no search over a hard-coded range of scales, and nothing to retune
when the hidden size or the norm changes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["weights_from_embedding", "embedding_from_histogram", "recovery_error"]


def weights_from_embedding(embedding: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Soft prompt -> bin weights. A temperature-scaled softmax."""
    emb = np.asarray(embedding, dtype=np.float64).ravel()
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    shifted = emb / tau
    shifted = shifted - shifted.max()  # standard overflow guard; softmax is shift-invariant
    weights = np.exp(shifted)
    return weights / weights.sum()


def embedding_from_histogram(histogram: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Bin counts -> the soft prompt that produced them.

    Returns the minimum-norm member of the shift class, i.e. the centred log
    histogram. Empty bins would send this to negative infinity, so the histogram
    must have mass everywhere -- which it does, because the target came from a
    softmax and the transport plan reproduces it exactly.
    """
    counts = np.asarray(histogram, dtype=np.float64).ravel()
    if np.any(counts <= 0):
        empty = int((counts <= 0).sum())
        raise ValueError(
            f"{empty} of {counts.size} bins are empty; the embedding cannot be "
            "recovered from a histogram with zero mass. Raise the resolution or "
            "lower tau so every bin receives at least one pixel."
        )
    log_p = np.log(counts / counts.sum())
    return tau * (log_p - log_p.mean())


def recovery_error(original: np.ndarray, recovered: np.ndarray) -> dict:
    """How far a round trip moved the soft prompt, reported the way the
    embedding application cares about it."""
    a = np.asarray(original, dtype=np.float64).ravel()
    b = np.asarray(recovered, dtype=np.float64).ravel()
    centred = a - a.mean()
    return {
        "mean_abs": float(np.abs(a - b).mean()),
        "max_abs": float(np.abs(a - b).max()),
        "relative_norm": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        # how much of the error is just the shift the softmax cannot carry
        "shift": float(a.mean() - b.mean()),
        "relative_norm_after_centring": float(
            np.linalg.norm(centred - (b - b.mean())) / np.linalg.norm(centred)
        ),
    }
