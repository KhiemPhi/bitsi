"""Task screws inferred from BITSI's output (no labels): which part to grasp, and the screw
(axis, point, pitch, magnitude) the grasped part should follow.

analyze() runs BITSI v4 (gravity axis offered, robot setting) and returns the parts BITSI found:
coaxial segments along the root spine (ordered along an up-pointing spine) and off-axis children
(with the point where each attaches to its parent). Each task rule below uses only those parts.

A screw follows Lynch & Park: rotation by theta about `axis` through `point`, plus pitch * theta
translation along the axis; pitch = inf is a pure translation of `theta` metres along `axis`.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

UP = np.array([0, 0, 1.0])
JAW = 0.085      # Robotiq 2F-85 opening (m)
RECEIVER = np.array([0.0, -1.0, 0.0])   # hand-over target direction (robot's right)


@dataclass
class TaskScrew:
    task: str
    grasp: np.ndarray            # boolean mask over the cloud: the part to grasp (press: the part to push)
    axis: np.ndarray
    point: np.ndarray
    pitch: float = 0.0
    theta: float = 0.0           # rad, or metres if pitch is inf
    lift: float = 0.0            # straight lift before the screw (rigid tools)
    push: bool = False           # press: fingers closed, no grasp
    across: bool = True          # grasp across a bar-like part
    info: dict = field(default_factory=dict)


@dataclass
class Part:
    idx: np.ndarray
    kind: str                    # "segment" | "child"
    order: int = 0               # position along the root spine (segments)
    attach: np.ndarray = None    # where a child meets its parent
    n_attach: int = 0
    protrude: float = 0.0        # how far the child stands out of the body's box (box children)

    def centroid(self, W):
        return W[self.idx].mean(0)


def obb(P):
    """(axes as columns sorted by extent, extents sorted descending)."""
    c = P - P.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ext = np.ptp(c @ vt.T, 0)
    o = np.argsort(-ext)
    return vt[o].T, ext[o]


def analyze(W, N):
    from fitpart.bitsi4 import BITSI4
    M = BITSI4(up=UP)
    lab, root = M.fit(W, N)
    a = root.spine.tangents[0] / np.linalg.norm(root.spine.tangents[0])
    if a @ UP < 0:
        a = -a
    s = (W - W.mean(0)) @ a
    parts = []
    body = root.body
    seg = root.seg_labels
    for k, sl in enumerate(np.unique(seg)):
        idx = body[seg == sl]
        parts.append(Part(idx, "segment", order=0))
    parts.sort(key=lambda p: s[p.idx].mean())
    for i, p in enumerate(parts):
        p.order = i
    tb = cKDTree(W[body]) if len(body) else None
    for ch in root.children:
        d, j = tb.query(W[ch.idx]) if tb is not None else (np.full(len(ch.idx), np.inf), None)
        near = ch.idx[d < 2.5 * M.h]
        att = W[near].mean(0) if len(near) else W[ch.idx][np.argmin(d)]
        parts.append(Part(np.asarray(ch.idx), "child", attach=att, n_attach=ch.n_attach))
    if not root.children:
        parts += box_children(W, M.h, parts)
    return dict(parts=parts, axis=a, h=M.h, labels=lab, spine_kind=root.spine.kind)


def box_children(W, h, parts, margin_h=3.0, min_pts=30, max_frac=0.2):
    """Rectangular-section primitive: the Fourier cross-section (order <= 4) cannot follow a box's
    corners, so small protrusions on box-like bodies (handles, knobs, buttons) are lost in the corner
    residual. Fit the body's robust oriented box (5-95% extent along its principal axes: a face holds
    far more than 5% of the points, a knob or button far less) and take
    points more than margin_h * h outside it, grouped into connected components, as children; the
    attach point is where a component meets the box. Their points are removed from the segments."""
    from fitpart.bitsi2.tree import _components
    c = W.mean(0)
    R, _ = obb(W)
    X = (W - c) @ R
    lo, hi = np.percentile(X, 5, axis=0), np.percentile(X, 95, axis=0)
    out = np.maximum(lo - X, X - hi).max(1)
    cand = np.flatnonzero(out > margin_h * h)
    if len(cand) == 0:
        return []
    nc, comp = _components(W[cand], 3.0 * h)
    kids = []
    for k in range(nc):
        idx = cand[comp == k]
        if len(idx) < min_pts or len(idx) > max_frac * len(W):
            continue
        base = idx[out[idx] < np.quantile(out[idx], 0.2)]
        kids.append(Part(idx, "child", attach=W[base].mean(0), n_attach=1, protrude=float(np.quantile(out[idx], 0.9))))
    taken = np.concatenate([p.idx for p in kids]) if kids else np.zeros(0, int)
    for p in parts:
        p.idx = np.setdiff1d(p.idx, taken)
    return kids


def mask(n, idx):
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def horiz(v):
    v = np.asarray(v, float) - (np.asarray(v, float) @ UP) * UP
    return v / (np.linalg.norm(v) + 1e-12)


def handle_part(W, A):
    """Bar-like part that fits the jaw: narrow width <= JAW and length >= 2 x width; longest wins."""
    best = None
    for p in A["parts"]:
        if len(p.idx) < 30:
            continue
        _, e = obb(W[p.idx])
        if e[1] <= JAW and e[0] >= 2 * e[1]:
            if best is None or e[0] > best[1]:
                best = (p, e[0])
    return None if best is None else best[0]


def far_part(W, A, ref):
    """The part whose centroid is farthest from ref (a tool's head, opposite its handle)."""
    others = [p for p in A["parts"] if p is not ref and len(p.idx) >= 30]
    return max(others, key=lambda p: np.linalg.norm(p.centroid(W) - ref.centroid(W))) if others else None


# ---------------------------------------------------------------------------------------- rules
def turn(W, A):
    segs = [p for p in A["parts"] if p.kind == "segment"]
    top = max(segs, key=lambda p: p.order)
    a = A["axis"]
    if len(segs) == 1:  # no cut: top 15% along the axis
        s = W @ a
        top = Part(np.flatnonzero(s > np.quantile(s, 0.85)), "segment")
    c = W[top.idx].mean(0)
    return TaskScrew("turn", mask(len(W), top.idx), a, c, 0.0, np.pi, across=False,
                     info=dict(cap_points=len(top.idx)))


def hinge(W, A):
    """Lid hinge: the knob (a child near the top) marks the lid's free edge; the hinge is the top edge on
    the opposite side of the object; opening lifts the knob."""
    kids = [p for p in A["parts"] if p.kind == "child"]
    c = W.mean(0)
    knob = max(kids, key=lambda p: p.centroid(W)[2] + 1e-3 * len(p.idx)) if kids else None
    rest = np.ones(len(W), bool)
    if knob is not None:
        rest[knob.idx] = False
    # the lid's top face (flat: many points share the top height, so compare with >=, knob excluded)
    top = rest & (W[:, 2] >= np.quantile(W[rest, 2], 0.9) - 1e-4)
    if kids:
        u = horiz(c - knob.centroid(W))            # from the knob across the object: the hinge side
        g = knob.idx
    else:  # no knob: hinge on the far edge from the robot, grasp the near top edge
        u = horiz(c)
        st = (W - c) @ u
        g = np.flatnonzero(top & (st < np.quantile(st[top], 0.15)))
    st = (W[top] - c) @ u
    edge = W[top][st > np.quantile(st, 0.97)].mean(0)
    axis = np.cross(UP, u)
    return TaskScrew("hinge", mask(len(W), g), axis / np.linalg.norm(axis), edge, 0.0, np.deg2rad(70),
                     across=False, info=dict(knob=bool(kids)))


def slide(W, A):
    kids = [p for p in A["parts"] if p.kind == "child"]
    body = np.concatenate([p.idx for p in A["parts"] if p.kind == "segment"])
    if not kids:
        return None
    bc = W[body].mean(0)
    big = [p for p in kids if len(p.idx) >= 50] or kids
    # the handle stands out of the body's box the furthest (spurious pieces sit at its faces/corners)
    hd = max(big, key=lambda p: p.protrude)
    o = horiz(hd.centroid(W) - bc)
    return TaskScrew("slide", mask(len(W), hd.idx), o, hd.centroid(W), np.inf, 0.12, across=True)


def press(W, A):
    kids = [p for p in A["parts"] if p.kind == "child"]
    if not kids:
        return None
    bt = max(kids, key=lambda p: p.centroid(W)[2])
    n = bt.centroid(W) - bt.attach
    n = n / (np.linalg.norm(n) + 1e-12)
    top = W[bt.idx][np.argmax(W[bt.idx] @ n)]
    return TaskScrew("press", mask(len(W), bt.idx), -n, top, np.inf, 0.007, push=True)


def squeeze(W, A):
    """Two arms joined at one end: the pivot is the end (along the long axis) where the raised arm comes
    closest to the lower one; close the raised arm toward the lower one about a horizontal axis there.
    (Plain geometry on the whole cloud; BITSI's parts are not needed here.)"""
    R, _ = obb(W)
    t = horiz(R[:, 0])
    s = (W - W.mean(0)) @ t
    zmin = W[:, 2].min()
    hi = W[:, 2] > zmin + 0.5 * (np.quantile(W[:, 2], 0.98) - zmin)      # raised arm
    bins = np.linspace(s.min(), s.max(), 11)
    gaps = []
    for k in range(10):
        m = (s >= bins[k]) & (s < bins[k + 1])
        gaps.append(np.quantile(W[m & hi, 2], 0.5) if (m & hi).sum() > 5 else np.inf)
    end = 1 if np.argmin(gaps) >= 5 else -1                 # pivot end along t
    piv = W.mean(0) + t * (s.max() if end > 0 else s.min())
    piv[2] = np.quantile(W[hi, 2], 0.05)
    d = -end * t                                             # from the pivot toward the free end
    axis = np.cross(UP, d)
    axis /= np.linalg.norm(axis)
    # closing moves the free end down: choose the sign of theta accordingly
    v = np.cross(axis, d)
    th = np.deg2rad(20) * (1 if v[2] < 0 else -1)
    sd = (W - piv) @ d
    g = hi & (sd > np.quantile(sd[hi], 0.5))
    return TaskScrew("squeeze", g, axis, piv, 0.0, th, across=True)


def pivot(W, A, robot=np.zeros(3)):
    c = W.mean(0)
    u = horiz(c - robot)                  # tilt away from the robot
    s = (W - c) @ u
    zmin = W[:, 2].min()
    edge = c + u * s.max()
    edge[2] = zmin
    axis = np.cross(UP, u)
    top = W[:, 2] > np.quantile(W[:, 2], 0.8)
    return TaskScrew("pivot", top, axis / np.linalg.norm(axis), edge, 0.0, np.deg2rad(50), across=False)


def lift(W, A):
    h, _ = handle_run(W)
    m = h if h is not None else np.ones(len(W), bool)
    return TaskScrew("lift", m, UP, W.mean(0), np.inf, 0.15, across=h is not None)


def width_profile(W, n=40, which="min"):
    """Sweep the tool along its long (PCA) axis: per slice, the narrower ("min") or wider ("max")
    cross-section width."""
    c = W.mean(0)
    R, _ = obb(W)
    t = R[:, 0]
    s = (W - c) @ t
    edges = np.linspace(s.min(), s.max() + 1e-9, n + 1)
    sl = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n - 1)
    wid = np.full(n, np.inf)
    for k in range(n):
        m = sl == k
        if m.sum() >= 5:
            X = (W[m] - c) @ R[:, 1:]
            w2 = (np.subtract(*np.percentile(X[:, 0], [98, 2])), np.subtract(*np.percentile(X[:, 1], [98, 2])))
            wid[k] = min(w2) if which == "min" else max(w2)
    return t, s, sl, wid


def handle_run(W, thin=0.045):
    """The handle = the longest contiguous run of slices narrow in BOTH directions (wider cross-section
    width <= thin: a head, bowl or whisk is wide in at least one direction); the head is the part beyond
    the far end of that run (the side with more of the remaining points)."""
    t, s, sl, wid = width_profile(W, which="max")
    ok = wid <= thin
    best, cur = (0, -1, -1), None
    for k in range(len(ok) + 1):
        if k < len(ok) and ok[k]:
            cur = k if cur is None else cur
        elif cur is not None:
            if k - cur > best[0]:
                best = (k - cur, cur, k - 1)
            cur = None
    if best[0] == 0:
        return None, None
    a, b = best[1], best[2]
    h = (sl >= a) & (sl <= b)
    lo, hi = sl < a, sl > b
    head = hi if hi.sum() >= lo.sum() else lo
    return h, head


def _tool(W, A, task):
    h, head = handle_run(W)
    if h is None:
        return None, None, None
    hc = W[h].mean(0)
    tc = W[head].mean(0) if head is not None and head.any() else W.mean(0)
    return Part(np.flatnonzero(h), "segment"), (Part(np.flatnonzero(head), "segment") if head is not None and head.any() else None), tc - hc


def swing(W, A):
    hd, head, t = _tool(W, A, "swing")
    if hd is None:
        return None
    axis = np.cross(horiz(t), UP)
    axis /= np.linalg.norm(axis)
    q = hd.centroid(W)
    # sign: the head must move down
    v = np.cross(axis, (q + t) - q)
    th = np.deg2rad(40) * (1 if v[2] < 0 else -1)
    return TaskScrew("swing", mask(len(W), hd.idx), axis, q, 0.0, th, lift=0.15, info=dict(head=t.tolist()))


def scoop(W, A):
    hd, head, t = _tool(W, A, "scoop")
    if hd is None:
        return None
    axis = np.cross(horiz(t), UP)
    axis /= np.linalg.norm(axis)
    q = hd.centroid(W) + t                 # rotate about the head (bowl)
    return TaskScrew("scoop", mask(len(W), hd.idx), axis, q, 0.0, np.deg2rad(35), lift=0.15)


def stir(W, A):
    hd, head, t = _tool(W, A, "stir")
    if hd is None:
        return None
    q = hd.centroid(W) + t + 0.03 * horiz(np.cross(UP, t))
    return TaskScrew("stir", mask(len(W), hd.idx), UP, q, 0.0, np.deg2rad(120), lift=0.15)


def thick_end(W, frac=0.3):
    """Width profile along the tool: (handle mask = the thicker end, other-end mask, handle->tip dir)."""
    t, s, sl, wid = width_profile(W)
    w = np.where(np.isfinite(wid), wid, np.nan)
    n = len(w)
    k = max(int(frac * n), 1)
    handle_hi = np.nanmean(w[-k:]) > np.nanmean(w[:k])
    hm = sl >= n - k if handle_hi else sl < k
    om = sl < k if handle_hi else sl >= n - k
    return hm, om, horiz(W[om].mean(0) - W[hm].mean(0))


def cut(W, A):
    """Knife: the handle is the thicker end (the blade is thin all along); stroke 10 cm from the handle
    toward the tip after a small lift."""
    hm, om, d = thick_end(W)
    return TaskScrew("cut", hm, d, W.mean(0), np.inf, 0.10, lift=0.08)


def handover(W, A):
    """Grasp the tool away from its handle so the handle can be offered: the handle is the thicker end of
    the tool's width profile; grasp the opposite 30%, then rotate about the vertical so the handle
    points at the receiver."""
    t, s, sl, wid = width_profile(W)
    w = np.where(np.isfinite(wid), wid, np.nan)
    n = len(w)
    lo_w, hi_w = np.nanmean(w[: int(0.3 * n)]), np.nanmean(w[-int(0.3 * n):])
    handle_hi = hi_w > lo_w
    g = sl < int(0.3 * n) if handle_hi else sl >= n - int(0.3 * n)
    hside = sl >= n - int(0.3 * n) if handle_hi else sl < int(0.3 * n)
    hdir = horiz(W[hside].mean(0) - W[g].mean(0))
    ang = np.arctan2(np.cross(hdir, RECEIVER) @ UP, hdir @ RECEIVER)
    return TaskScrew("handover", g, UP, W[g].mean(0), 0.0, ang, lift=0.15, across=True,
                     info=dict(handle_dir=hdir.tolist()))


RULES = dict(turn=turn, hinge=hinge, slide=slide, press=press, squeeze=squeeze, pivot=pivot, lift=lift,
             swing=swing, scoop=scoop, stir=stir, cut=cut, handover=handover)


def infer(task, W, N):
    A = analyze(W, N)
    ts = RULES[task](W, A)
    if ts is not None:
        ts.info.update(spine_kind=A["spine_kind"], n_parts=len(A["parts"]),
                       n_children=sum(p.kind == "child" for p in A["parts"]))
    return ts, A
