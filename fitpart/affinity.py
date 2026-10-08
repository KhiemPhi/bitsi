"""Affinities between superpoints: geometric (Sec. 4.3) and semantic (Sec. 4.2)."""
import numpy as np

from .cheb import multiscale_cuts, jump_threshold_cuts
from .profiles import bitc_profile, sweep_directions


def detect_all_cuts(points, n_extra_dirs=9, axis_only=False, detector="cheb", delta=0.2,
                    norm="object", extent_weight=False, profile_kwargs=None, cut_kwargs=None):
    """Profiles + cuts for every sweep direction.

    detector: "cheb" (inflections, ours), "extrema" (Chebyshev extrema ablation),
              "single" (single Chebyshev degree ablation), "jump" (ITSI threshold delta).
    norm: "object" divides every cut's slope by the steepest fitted slope over all
          directions of this object, so flat, noisy directions yield weak cuts;
          "profile" keeps the per-profile normalization of s(c) as written in the draft.
    extent_weight: also scale strengths by extent_d / max extent, so cuts along short
          directions (across the object's thin dimension) count less.
    Returns a list of (direction, profile dict, cuts).
    """
    profile_kwargs = profile_kwargs or {}
    cut_kwargs = cut_kwargs or {}
    dirs = sweep_directions(points, n_extra=0 if axis_only else n_extra_dirs)
    out = []
    for d in dirs:
        prof = bitc_profile(points, d, **profile_kwargs)
        if detector == "jump":
            cuts = jump_threshold_cuts(prof["t"], prof["values"], delta)
        else:
            cuts, _ = multiscale_cuts(prof["t"], prof["values"],
                                      order=1 if detector == "extrema" else 2,
                                      single_scale=detector == "single", **cut_kwargs)
        out.append((d, prof, cuts))
    if detector != "jump" and norm == "object":
        g = max((c.raw_slope for _, _, cuts in out for c in cuts), default=0.0)
        if g > 0:
            for _, _, cuts in out:
                for c in cuts:  # keep the persistence factor, swap the slope normalizer
                    c.strength = c.strength / max(c.slope, 1e-12) * (c.raw_slope / g)
    if extent_weight:
        ext = np.array([np.ptp(prof["coord"]) for _, prof, _ in out])
        for (_, _, cuts), e in zip(out, ext / ext.max()):
            for c in cuts:
                c.strength *= e
    return out


def geometric_affinity(points, sp_labels, pairs, cut_sets):
    """A_geo_ij = 1 - max_{c separates i, j} s(c), for each pair in `pairs`.

    A cut at t along d separates superpoints i and j when their median coordinates
    along d lie on opposite sides of t.
    """
    M = sp_labels.max() + 1
    strongest = np.zeros(len(pairs))
    for d, prof, cuts in cut_sets:
        if not cuts:
            continue
        coord = prof["coord"]
        med = np.array([np.median(coord[sp_labels == i]) for i in range(M)])
        a, b = med[pairs[:, 0]], med[pairs[:, 1]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        for c in cuts:
            sep = (lo < c.t) & (c.t < hi)
            strongest[sep] = np.maximum(strongest[sep], c.strength)
    return 1.0 - strongest


def mask_cooccurrence(view_sp_mask):
    """A_mask_ij from per-view majority mask labels.

    view_sp_mask: (V, M) int array, mask label of superpoint i in view v, -1 if not visible
                  (or not covered by any mask).
    Returns dense (M, M) affinity and (M, M) co-visibility counts.
    """
    V, M = view_sp_mask.shape
    same = np.zeros((M, M))
    both = np.zeros((M, M))
    for v in range(V):
        lab = view_sp_mask[v]
        vis = lab >= 0
        idx = np.where(vis)[0]
        l = lab[idx]
        both[np.ix_(idx, idx)] += 1
        same[np.ix_(idx, idx)] += (l[:, None] == l[None, :])
    aff = np.where(both > 0, same / np.maximum(both, 1), np.nan)
    return aff, both


def cosine_affinity(feats):
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return f @ f.T


def normal_smoothness(normals, sp_labels, pairs):
    """Mean-normal agreement |n_i . n_j| between adjacent superpoints (a weak base affinity)."""
    M = sp_labels.max() + 1
    n = np.zeros((M, 3))
    np.add.at(n, sp_labels, normals)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.abs(np.einsum("ij,ij->i", n[pairs[:, 0]], n[pairs[:, 1]]))


COLOR_SLOPE_FLOOR = {"L": 30.0, "a": 30.0, "b": 30.0}


def detect_all_cuts_mc(points, colors=None, channels=("geo", "topo", "color"), n_extra_dirs=9,
                       axis_only=False, max_deg=16, cut_kwargs=None, gref=None, return_gmax=False):
    """Multi-channel version of `detect_all_cuts`: per direction, every channel profile gets the
    multi-scale Chebyshev analysis; strengths are normalized per channel by that channel's
    steepest fitted slope over all directions (so channels with different units are
    comparable); cuts from all channels are pooled, merging cuts within eta (2 slabs) and keeping
    the strongest. Each pooled cut remembers which channels produced it (`c.channels`).
    """
    from .cheb import Cut
    from .profiles import multichannel_profiles
    cut_kwargs = dict(cut_kwargs or {})
    cut_kwargs.setdefault("max_deg", max_deg)
    dirs = sweep_directions(points, n_extra=0 if axis_only else n_extra_dirs)
    per_dir = []
    gmax = {}
    for d in dirs:
        prof = multichannel_profiles(points, d, colors, channels)
        chcuts = {}
        for nm, vals in prof["channels"].items():
            cuts, _ = multiscale_cuts(prof["t"], vals, **cut_kwargs)
            chcuts[nm] = cuts
            gmax[nm] = max([gmax.get(nm, 0.0)] + [c.raw_slope for c in cuts])
        per_dir.append((d, prof, chcuts))
    # absolute floors so a nearly uniform channel cannot produce strong cuts from noise:
    # CIELAB units per unit of normalized sweep position
    for nm, floor in COLOR_SLOPE_FLOOR.items():
        if nm in gmax:
            gmax[nm] = max(gmax[nm], floor)
    if gref is not None:  # normalize by the root object's slopes so pieces are comparable
        gmax = {nm: gref.get(nm, g) for nm, g in gmax.items()}
    out = []
    for d, prof, chcuts in per_dir:
        spacing = prof["spacing"]
        pooled = []
        for nm, cuts in chcuts.items():
            g = gmax.get(nm, 0.0)
            if g <= 0:
                continue
            for c in cuts:
                s = c.strength / max(c.slope, 1e-12) * (c.raw_slope / g)
                hit = [p for p in pooled if abs(p.t - c.t) < 2 * spacing]
                if hit:
                    p = hit[0]
                    p.channels.add(nm)
                    if s > p.strength:
                        p.strength, p.t = s, c.t
                else:
                    pc = Cut(t=c.t, strength=s, level=c.level, slope=c.slope, raw_slope=c.raw_slope,
                             levels=list(c.levels))
                    pc.channels = {nm}
                    pooled.append(pc)
        pooled.sort(key=lambda c: c.t)
        out.append((d, prof, pooled))
    return (out, gmax) if return_gmax else out
