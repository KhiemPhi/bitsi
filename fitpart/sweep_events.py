"""Event detection on multi-channel sweep profiles, with ideas borrowed from SIFT and HOG.

  * significance (SIFT's contrast rejection): every candidate cut gets a z-score, the step across
    it divided by the channel's noise (robustly estimated from first differences); cuts with
    z > z_min (3) are kept. One statistical scale for every channel, no hand-set thresholds.
  * HOG-style histogram channels: per slab, a histogram of surface-normal orientation relative
    to the sweep axis and a CIELAB colour histogram; the profile is the Hellinger distance between
    the k slabs before and after each position; peaks are scored by robust z (median / MAD).
  * scalar channels (BITC, beta_0, beta_1, mean L, a, b): multi-scale Chebyshev inflections
    (`cheb.multiscale_cuts`), re-scored by the significance test.
  * orientation selection (SIFT's orientation assignment): candidate sweep directions are the
    PCA axes, hemisphere samples and the principal axes of the surface-normal tensor; a direction
    is scored by its total significant evidence.
"""
import numpy as np

from .cheb import Cut, multiscale_cuts
from .profiles import fibonacci_hemisphere, multichannel_profiles, pca_axes


def robust_noise(y):
    """Noise sigma of a profile from first differences.

    MAD-based (robust to the steps we are looking for), floored at half the plain standard
    deviation of the differences: piecewise-constant channels (beta counts, BITC plateaus) have
    MAD = 0, which would make every jump look infinitely significant.
    """
    d = np.diff(np.asarray(y, float))
    if len(d) < 3:
        return 1e-9
    mad = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    return max(mad, 0.5 * d.std() / np.sqrt(2), 1e-9)


def step_z(y, i_cut, k, sigma):
    """Two-window step statistic at slab boundary i_cut (cut between slab i_cut and i_cut+1)."""
    lo, hi = max(0, i_cut - k + 1), min(len(y), i_cut + 1 + k)
    left, right = y[lo:i_cut + 1], y[i_cut + 1:hi]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    se = sigma * np.sqrt(1.0 / len(left) + 1.0 / len(right))
    return abs(right.mean() - left.mean()) / se


def normal_tensor_axes(normals):
    n = np.asarray(normals, float)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    T = n.T @ n
    _, vecs = np.linalg.eigh(T)
    return vecs.T[::-1]  # dominant normal direction first


def candidate_directions(points, normals=None, n_extra=9, min_angle_deg=15.0):
    axes = pca_axes(points)
    cands = [a for a in axes]
    if normals is not None:
        cands += list(normal_tensor_axes(normals))
    cands += [h @ axes for h in fibonacci_hemisphere(n_extra)]
    cos_min = np.cos(np.deg2rad(min_angle_deg))
    out = []
    for d in cands:
        d = d / np.linalg.norm(d)
        if all(abs(d @ e) < cos_min for e in out):
            out.append(d)
    return np.array(out)


def _hist_profiles(prof, points, normals, colors, d, k):
    """Hellinger distance between k-slab windows of per-slab histograms (normals, colour)."""
    slab_of = prof["slab_of"]
    S = len(prof["t"])
    chans = {}
    if normals is not None:
        n = np.asarray(normals, float)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        ang = np.abs(n @ d)  # |cos| of normal vs sweep axis
        b = np.minimum((ang * 6).astype(int), 5)
        H = np.zeros((S, 6))
        np.add.at(H, (slab_of, b), 1)
        chans["hist_normal"] = H
    if colors is not None:
        from skimage.color import rgb2lab
        c = np.asarray(colors, float)
        if c.max() > 1.5:
            c = c / 255.0
        lab = rgb2lab(c[None])[0]
        bl = np.clip((lab[:, 0] / 25).astype(int), 0, 3)
        ba = np.clip(((lab[:, 1] + 40) / 20).astype(int), 0, 3)
        bb = np.clip(((lab[:, 2] + 40) / 20).astype(int), 0, 3)
        H = np.zeros((S, 12))
        np.add.at(H, (slab_of, bl), 1)
        np.add.at(H, (slab_of, 4 + ba), 1)
        np.add.at(H, (slab_of, 8 + bb), 1)
        chans["hist_color"] = H
    out = {}
    for nm, H in chans.items():
        C = np.vstack([np.zeros((1, H.shape[1])), np.cumsum(H, 0)])
        D = np.zeros(S - 1)
        for i in range(S - 1):  # boundary between slab i and i+1
            lo, hi = max(0, i - k + 1), min(S, i + 1 + k)
            L = C[i + 1] - C[lo]
            R = C[hi] - C[i + 1]
            if L.sum() < 3 or R.sum() < 3:
                continue
            p, q = L / L.sum(), R / R.sum()  # block (window) normalization
            D[i] = np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q))))
        out[nm] = D
    return out


def events_along(points, d, normals=None, colors=None, channels=("geo", "topo", "color", "hist"),
                 k=3, z_min=3.0, max_deg=16):
    """Significant cut events along direction d. Returns (cuts, prof); cut.strength = z."""
    scal = tuple(c for c in channels if c in ("geo", "topo", "color"))
    prof = multichannel_profiles(points, d, colors if "color" in channels else None, channels=scal or ("geo",))
    t = prof["t"]
    S = len(t)
    spacing = prof["spacing"]
    cand = []
    if scal:
        for nm, y in prof["channels"].items():
            if nm == "geo" and "geo" not in channels:
                continue
            sigma = robust_noise(y)
            cuts, _ = multiscale_cuts(t, y, max_deg=max_deg)
            for c in cuts:
                i = int(np.clip(np.searchsorted(t, c.t) - 1, 0, S - 2))
                z = step_z(y, i, k, sigma)
                cand.append((c.t, z, nm))
    if "hist" in channels:
        for nm, D in _hist_profiles(prof, points, normals, colors if colors is not None else None, d, k).items():
            if len(D) < 5:
                continue
            med = np.median(D)
            mad = 1.4826 * np.median(np.abs(D - med)) + 1e-9
            for i in range(1, len(D) - 1):
                if D[i] > 0 and D[i] == D[max(0, i - k):i + k + 1].max():
                    z = (D[i] - med) / mad
                    cand.append((0.5 * (t[i] + t[i + 1]), z, nm))
    # pool: merge within 2 slabs, keep max z; keep significant
    cand.sort(key=lambda x: -x[1])
    pooled = []
    for tc, z, nm in cand:
        if z < z_min:
            break
        hit = [p for p in pooled if abs(p.t - tc) < 2 * spacing]
        if hit:
            hit[0].channels.add(nm)
            continue
        c = Cut(t=float(tc), strength=float(z), level=1, slope=0.0, levels=[1])
        c.channels = {nm}
        pooled.append(c)
    pooled.sort(key=lambda c: c.t)
    return pooled, prof


def direction_score(cuts, cap=10.0):
    return float(sum(min(c.strength, cap) for c in cuts))


def detect_events(points, normals=None, colors=None, channels=("geo", "topo", "color", "hist"),
                  directions="auto", **kw):
    """Events along every candidate direction: list of (d, prof, cuts, score), best first."""
    dirs = candidate_directions(points, normals) if directions == "auto" else np.asarray(directions)
    out = []
    for d in dirs:
        cuts, prof = events_along(points, d, normals, colors, channels, **kw)
        out.append((d, prof, cuts, direction_score(cuts)))
    out.sort(key=lambda x: -x[3])
    return out
