"""Paper selection and a separately labeled reconstruction of the proposal rule."""

from collections import defaultdict
import numpy as np


def select_winners(rows, expected_candidates=5):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (str(row["problem_id"]), str(row["candidate_id"]))
        if key in seen:
            raise ValueError(f"Duplicate candidate: {key}")
        seen.add(key)
        for metric in ("total_entropy", "mutual_information"):
            if not np.isfinite(row[metric]):
                raise ValueError(f"Nonfinite {metric} for {key}")
        groups[key[0]].append(row)
    if not groups:
        raise ValueError("No candidates")
    baseline, epistemic = [], []
    for pid, candidates in sorted(groups.items()):
        if len(candidates) != expected_candidates:
            raise ValueError(f"{pid}: expected {expected_candidates} candidates, got {len(candidates)}")
        if len({r['problem'] for r in candidates}) != 1:
            raise ValueError(f"Inconsistent problem text for {pid}")
        # Lexicographic candidate ID breaks ties reproducibly, independent of input order.
        for metric, output in [("total_entropy", baseline), ("mutual_information", epistemic)]:
            output.append(min(candidates, key=lambda r: (-r[metric], str(r['candidate_id']))))
    changed = sum(a['candidate_id'] != b['candidate_id'] for a, b in zip(baseline, epistemic))
    return baseline, epistemic, {"problems": len(groups), "different_winners": changed,
                                "disagreement_percent": 100 * changed / len(groups)}


def chi_squared_weights(utilities, regularization=1.0, prior=None):
    """Maximize q.u - lambda/2 * sum((q-p)^2/p) on the probability simplex.

    This is an explicit optional reconstruction; the notebook does not preserve
    its original objective or normalization, and the paper uses argmax instead.
    """
    u = np.asarray(utilities, dtype=float)
    if u.ndim != 1 or not u.size or not np.isfinite(u).all():
        raise ValueError("Utilities must be a nonempty finite vector")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be positive")
    p = np.full(u.size, 1 / u.size) if prior is None else np.asarray(prior, dtype=float)
    if p.shape != u.shape or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("Prior must be a strictly positive probability vector")
    # Center first to avoid large common offsets. Solve the KKT threshold.
    u = u - u.max()
    lo, hi = u.min(), regularization
    for _ in range(100):
        eta = (lo + hi) / 2
        q = p * np.maximum(0, 1 + (u - eta) / regularization)
        if q.sum() > 1:
            lo = eta
        else:
            hi = eta
    q = p * np.maximum(0, 1 + (u - (lo + hi) / 2) / regularization)
    return q / q.sum()
