"""Multi-scale cut detection from Chebyshev approximations (Sec. 4.3).

A profile g(t) on [-1, 1] is fit by a truncated Chebyshev series at degrees
n_1 < ... < n_L. At each level, cut candidates are the sign-changing real roots of
p'' (computed as colleague-matrix eigenvalues by numpy's `chebroots`). A cut's
persistence l(c) is the coarsest level where it appears (matched across levels
within eta), and its scale-aware strength is

    s(c) = (L - l(c) + 1) / L * |p'(t_c)| / max_t |p'(t)|      (at level l(c))

The ITSI jump-threshold and Chebyshev-extrema detectors are included for the
ablations in Sec. 5.6.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev as C


@dataclass
class Cut:
    t: float          # location in profile units (same units as the profile's t)
    strength: float   # s(c) in (0, 1]
    level: int        # persistence l(c), 1 = coarsest
    slope: float      # normalized |p'| at the persisting level
    raw_slope: float = 0.0  # |dp/dt| in profile units per unit t (for cross-direction normalization)
    levels: list = field(default_factory=list)  # every level it was found at


def to_unit(t):
    lo, hi = t.min(), t.max()
    return 2 * (t - lo) / max(hi - lo, 1e-12) - 1, lo, hi


def from_unit(x, lo, hi):
    return lo + (x + 1) * 0.5 * (hi - lo)


def chebfit(x, y, deg):
    return C.chebfit(x, y, deg)


def chop_degree(x, y, max_deg=64, plateau_factor=3.0, min_deg=4):
    """Pick n_L by truncating where coefficients reach the noise plateau.

    A least-squares analogue of Chebfun's chopping rule: fit at max_deg, form the
    monotone envelope e_k = max_{j >= k} |c_j|, estimate the plateau as the median
    envelope over the upper half of the coefficients, and return the smallest k
    whose envelope is within `plateau_factor` of that plateau.
    """
    max_deg = int(min(max_deg, len(x) // 2))
    if max_deg <= min_deg:
        return max(max_deg, 1)
    c = np.abs(chebfit(x, y, max_deg))
    if c.max() == 0:
        return min_deg
    env = np.maximum.accumulate(c[::-1])[::-1] / c.max()
    plateau = np.median(env[max_deg // 2:])
    k = int(np.argmax(env <= plateau_factor * plateau))
    return int(np.clip(k, min_deg, max_deg))


def bic_degree(x, y, max_deg=64, min_deg=2):
    """Degree minimizing BIC (the rule used by `main.fit_best_chebyshev`)."""
    n = len(x)
    max_deg = int(min(max_deg, n - 2))
    best, best_deg = np.inf, min_deg
    for deg in range(min_deg, max_deg + 1):
        rss = np.sum((C.chebval(x, chebfit(x, y, deg)) - y) ** 2)
        bic = n * np.log(rss / n + 1e-18) + (deg + 1) * np.log(n)
        if bic < best:
            best, best_deg = bic, deg
    return best_deg


def degree_schedule(n_top, fractions=(0.25, 0.5, 0.75, 1.0), min_deg=3):
    degs = sorted({max(min_deg, int(round(f * n_top))) for f in fractions})
    return degs


def sign_change_roots(coeffs, order, dense=None):
    """Real roots in [-1, 1] of the order-th derivative where it changes sign."""
    dc = C.chebder(coeffs, order)
    if len(dc) < 2:
        return np.array([])
    r = C.chebroots(dc)
    r = r[np.abs(r.imag) < 1e-7].real
    r = r[(r > -1) & (r < 1)]
    if len(r) == 0:
        return r
    d = 1e-4
    keep = C.chebval(r - d, dc) * C.chebval(r + d, dc) < 0
    return np.sort(r[keep])


def level_cuts(coeffs, order=2, n_dense=2048, margin=0.0):
    """Cut candidates at one level: (x, normalized slope, raw slope, max raw slope).

    Roots within `margin` of either end of [-1, 1] are dropped and slopes are
    normalized over the interior only: polynomial fits ring at the domain ends, and a
    cut there would only shave off a sliver.
    """
    roots = sign_change_roots(coeffs, order)
    roots = roots[np.abs(roots) <= 1 - margin]
    d = C.chebder(coeffs, 1 if order == 2 else 2)  # extrema ablation: curvature magnitude
    xs = np.linspace(-1 + margin, 1 - margin, n_dense)
    max_slope = np.abs(C.chebval(xs, d)).max() + 1e-12
    raw = np.abs(C.chebval(roots, d))
    return roots, raw / max_slope, raw, max_slope


def multiscale_cuts(t, values, degree_rule="chop", max_deg=64, fractions=(0.25, 0.5, 0.75, 1.0),
                    eta=None, single_scale=False, order=2):
    """Detect cuts on one profile.

    t, values : profile (slab centres and values). Empty slabs should carry 0.
    eta       : match tolerance in t units (default: 2 slab spacings).
    single_scale : only the top degree n_L (ablation "single Chebyshev degree").
    order     : 2 = inflection points (ours), 1 = extrema (ablation).

    Returns (cuts, info) with cuts sorted by location.
    """
    t = np.asarray(t, float)
    y = np.asarray(values, float)
    if len(t) < 6 or np.ptp(y) == 0:
        return [], dict(degrees=[], coeffs=[])
    x, lo, hi = to_unit(t)
    y0, yr = y.min(), np.ptp(y)
    yn = (y - y0) / yr
    n_top = chop_degree(x, yn, max_deg) if degree_rule == "chop" else bic_degree(x, yn, max_deg)
    degs = [n_top] if single_scale else degree_schedule(n_top, fractions)
    L = len(degs)
    spacing = (hi - lo) / max(len(t) - 1, 1)
    eta = 2 * spacing if eta is None else eta
    eta_x = eta * 2 / max(hi - lo, 1e-12)
    end_margin = 2 * 2.0 / max(len(t) - 1, 1)  # two slabs, in unit coordinates

    cuts = []  # in unit coordinates while matching
    all_coeffs = []
    max_raw = 0.0
    for li, deg in enumerate(degs, start=1):
        c = chebfit(x, yn, deg)
        all_coeffs.append(c)
        roots, slope, raw, mx = level_cuts(c, order, margin=end_margin)
        # back to raw profile units (y = y0 + yr * yn); slope stays per unit x so short
        # sweep directions are not favoured over long ones
        scale = yr  # raw profile units per unit of normalized position x in [-1, 1]
        max_raw = max(max_raw, mx * scale)
        for r, s, rs in zip(roots, slope, raw * scale):
            match = None
            if cuts:
                dists = [abs(cu.t - r) for cu in cuts]
                j = int(np.argmin(dists))
                if dists[j] < eta_x:
                    match = cuts[j]
            if match is None:
                cuts.append(Cut(t=r, strength=(L - li + 1) / L * s, level=li, slope=s,
                                raw_slope=float(rs), levels=[li]))
            else:
                match.levels.append(li)
    for cu in cuts:
        cu.t = float(from_unit(cu.t, lo, hi))
    cuts.sort(key=lambda c: c.t)
    return cuts, dict(degrees=degs, coeffs=all_coeffs, lo=lo, hi=hi, n_top=n_top, y0=y0, yr=yr,
                      max_raw_slope=max_raw, n_levels=L)


def jump_threshold_cuts(t, values, delta):
    """ITSI-style cuts: between consecutive slabs whose normalized values jump >= delta."""
    y = np.asarray(values, float)
    if np.ptp(y) == 0:
        return []
    yn = (y - y.min()) / np.ptp(y)
    jumps = np.abs(np.diff(yn))
    idx = np.where(jumps >= delta)[0]
    mid = 0.5 * (np.asarray(t)[idx] + np.asarray(t)[idx + 1])
    return [Cut(t=float(m), strength=1.0, level=1, slope=float(j), levels=[1])
            for m, j in zip(mid, jumps[idx])]
