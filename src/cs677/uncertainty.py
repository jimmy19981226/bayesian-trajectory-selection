"""Uncertainty in nats. No ML dependencies are needed for the numerical core."""

import numpy as np


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] == 0:
        raise ValueError("A nonempty vocabulary axis is required")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(p.sum(axis=-1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    return -(p * logs).sum(axis=-1)


def decompose(probabilities):
    """Accept [passes, tokens, vocabulary], return total, expected, MI [tokens]."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or min(p.shape) == 0:
        raise ValueError("Expected nonempty [passes, tokens, vocabulary]")
    expected = entropy(p).mean(axis=0)
    total = entropy(p.mean(axis=0))
    return {"total_entropy": total, "expected_entropy": expected,
            "mutual_information": np.maximum(total - expected, 0)}


def top_m_mean(values, m=50):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite token scores")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    m = min(m, values.size)
    return float(np.partition(values, values.size - m)[-m:].mean())
