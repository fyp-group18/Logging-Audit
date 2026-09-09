"""Paired bootstrap significance testing for retrieval metric comparison."""

from __future__ import annotations

import random


def paired_bootstrap(
    values_a: list[float],
    values_b: list[float],
    n_iterations: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap test.

    Resamples query-level metric pairs together (preserving pairing)
    and returns a p-value: fraction of bootstrap samples where the
    resampled mean delta is at least as extreme as the observed delta.

    Args:
        values_a: Per-query metric values for system A (e.g., hybrid).
        values_b: Per-query metric values for system B (e.g., vector).
        n_iterations: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        p-value in [0, 1].
    """
    n = len(values_a)
    if n != len(values_b):
        raise ValueError(f"Paired lists must have equal length: {n} vs {len(values_b)}")
    if n == 0:
        return 1.0

    # Compute paired differences and center under H0: E[d] = 0
    diffs = [a - b for a, b in zip(values_a, values_b)]
    observed_delta = sum(diffs) / n
    mean_diff = observed_delta
    centered = [d - mean_diff for d in diffs]

    rng = random.Random(seed)
    extreme_count = 0
    for _ in range(n_iterations):
        boot_mean = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(boot_mean) >= abs(observed_delta):
            extreme_count += 1

    return extreme_count / n_iterations
