"""Cuts along a spine without a part count: significant events in the primitive's own channels.

section channels   steps in the cross-section amplitudes a0(s), |c_1|(s), ..., |c_n|(s)
                   (a lid or cap is a step in a0; a body turning box-like is a step in |c_4|):
                   two-window step statistic divided by the channel's robust noise
histogram channels Hellinger distance between k-slab windows of per-slab histograms of the
                   normal angle to the local tangent (6 bins) and CIELAB colour (12 bins),
                   standardized by median / MAD over boundaries (the "HOG" detector)
Candidates are local maxima within k slabs; candidates within 2 slabs are pooled (max z);
events with z > z_min are cuts.
"""
import numpy as np

from ..sweep_events import robust_noise, step_z


def _hellinger_profile(H, k):
    S = len(H)
    C = np.vstack([np.zeros((1, H.shape[1])), np.cumsum(H, 0)])
    D = np.zeros(S - 1)
    for i in range(S - 1):
        lo, hi = max(0, i - k + 1), min(S, i + 1 + k)
        L, R = C[i + 1] - C[lo], C[hi] - C[i + 1]
        if L.sum() < 3 or R.sum() < 3:
            continue
        p, q = L / L.sum(), R / R.sum()
        D[i] = np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q))))
    return D


def _local_max(z, k):
    return [i for i in range(len(z)) if z[i] > 0 and z[i] == z[max(0, i - k):i + k + 1].max()]


def spine_events(spine, section=None, normals=None, colors=None, k=3, z_min=3.0,
                 channels=("section", "hist")):
    S = spine.n_slabs
    if S < 4:
        return [], []
    cand = []
    if ("section" in channels or "section0" in channels) and section is not None:
        A = section.amplitudes()
        for j in range(A.shape[1] if "section" in channels else 1):  # section0: mean radius only
            y = A[:, j]
            sig = robust_noise(y)
            z = np.array([step_z(y, i, k, sig) for i in range(S - 1)])
            cand += [(i, z[i], f"sec{j}") for i in _local_max(z, k)]
    if "hist" in channels:
        sl = spine.slab
        hs = []
        if normals is not None:
            n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
            ang = np.abs(np.sum(n * spine.tangents[sl], 1))
            H = np.zeros((S, 6))
            np.add.at(H, (sl, np.minimum((ang * 6).astype(int), 5)), 1)
            hs.append(("hist_normal", H))
        if colors is not None:
            from skimage.color import rgb2lab
            c = np.asarray(colors, float)
            c = c / 255.0 if c.max() > 1.5 else c
            lab = rgb2lab(c[None])[0]
            H = np.zeros((S, 12))
            np.add.at(H, (sl, np.clip((lab[:, 0] / 25).astype(int), 0, 3)), 1)
            np.add.at(H, (sl, 4 + np.clip(((lab[:, 1] + 40) / 20).astype(int), 0, 3)), 1)
            np.add.at(H, (sl, 8 + np.clip(((lab[:, 2] + 40) / 20).astype(int), 0, 3)), 1)
            hs.append(("hist_color", H))
        for nm, H in hs:
            D = _hellinger_profile(H, k)
            if len(D) < 5:
                continue
            med = np.median(D)
            mad = 1.4826 * np.median(np.abs(D - med)) + 1e-9
            z = (D - med) / mad
            cand += [(i, z[i], nm) for i in _local_max(D, k)]
    cand.sort(key=lambda x: -x[1])
    pooled = []
    for i, z, nm in cand:
        if z < z_min:
            break
        if any(abs(i - p[0]) < 2 for p in pooled):
            continue
        pooled.append((i, z, nm))
    pooled.sort()
    return [p[0] for p in pooled], pooled
