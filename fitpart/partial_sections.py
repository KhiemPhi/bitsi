"""Observed-only cross-section profiles and gap-aware Chebyshev partitioning.

Missing angular bins are NaN, never zero or interpolated measurements. Coordinates
remain slab positions even across gaps. This is an experimental alternative to
partitioning the Fourier amplitudes of independently completed cross-sections.
"""
import numpy as np
from numpy.polynomial import chebyshev as Ch


def observed_radii(rho, phi, slab, n_slabs, n_bins=24, min_points=2):
    bins = np.clip(((phi + np.pi) * n_bins / (2 * np.pi)).astype(int), 0, n_bins - 1)
    key = slab * n_bins + bins
    order = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order], return_index=True)
    Y = np.full((n_slabs, n_bins), np.nan)
    for k, idx in zip(keys, np.split(order, starts[1:])):
        if len(idx) >= min_points:
            Y.flat[k] = np.quantile(rho[idx], 0.9)
    return Y


def masked_standardize(Y, noise_floor):
    """Estimate noise only from observed consecutive triples in the same bin."""
    out = Y.copy()
    for j in range(Y.shape[1]):
        y = Y[:, j]
        d = (y[2:] - 2 * y[1:-1] + y[:-2]) / np.sqrt(6)
        d = d[np.isfinite(d)]
        sigma = max(noise_floor, 1.4826 * np.median(np.abs(d - np.median(d)))) if len(d) else noise_floor
        seen = np.isfinite(y)
        if seen.any():
            out[seen, j] = (y[seen] - np.median(y[seen])) / sigma
    return out


def masked_piece_cost(Y, a, b, max_deg=8):
    """BIC in unit-noise coordinates; unobserved entries contribute no likelihood."""
    total = 0.0
    for y in Y[a:b].T:
        ok = np.isfinite(y)
        m = int(ok.sum())
        if not m:
            continue
        x = np.linspace(-1, 1, b - a)[ok]
        Q, _ = np.linalg.qr(Ch.chebvander(x, min(max_deg, m - 1)))
        rss = np.maximum(0, np.sum(y[ok] ** 2) - np.cumsum((Q.T @ y[ok]) ** 2))
        total += np.min(rss + np.arange(1, len(rss) + 1) * np.log(max(m, 2)))
    return float(total)


def masked_partition(Y, min_len=3, penalty=10.0, max_deg=8, min_common_bins=3):
    """DP with cuts allowed only between slabs sharing measured angular support.

    No boundary is inferred inside or next to an unobserved slab. Each piece also
    needs min_len measured slabs; missing rows cannot pay this sample requirement.
    """
    S = len(Y)
    if not S:
        return [], 0.0
    seen = np.isfinite(Y)
    valid = seen.any(1)
    allowed = np.zeros(S, bool)
    allowed[1:] = (seen[:-1] & seen[1:]).sum(1) >= min_common_bins
    count = np.r_[0, np.cumsum(valid)]
    F = np.full(S + 1, np.inf)
    back = np.zeros(S + 1, int)
    F[0] = 0
    for b in range(1, S + 1):
        if b != S and not allowed[b]:
            continue
        for a in range(b):
            if not np.isfinite(F[a]) or (a and not allowed[a]):
                continue
            if count[b] - count[a] < min_len and (a != 0 or b != S):
                continue
            cost = F[a] + masked_piece_cost(Y, a, b, max_deg)
            cost += penalty * np.log(max(int(valid.sum()), 2)) if a else 0
            if cost < F[b]:
                F[b], back[b] = cost, a
    cuts, b = [], S
    while b:
        a = back[b]
        if a:
            cuts.append(int(a - 1))
        b = a
    return sorted(cuts), float(F[S])
