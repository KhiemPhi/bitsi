"""BITSI v4: BITSI v3 with children, applicability and child cuts decided by code length.

Changes from v3 (RESULTS.md section 20):

  * Children by description length instead of thresholds. v3 kept a residual piece as a child if it
    had >= 1% of the points and covered <= 50% of the angle about the spine (and its points were
    > 3h outside the envelope). v4 takes every connected piece > `cand_margin_h` h outside the body
    envelope (2h: thin wire handles stick out less than 3h) and asks which description is shorter:
      H0  absorb: the body's cross-section, refit per slab WITHOUT one-sided rejection, explains the
          piece too (the envelope bends outward to cover it; every angle bin of those slabs pays
          its squared residual, so inflating the envelope is not free);
      H1  child: the body's slabs are refit without the piece, and the piece gets its own swept
          primitive (geodesic spine: 3 numbers per slab; cross-section: its own Fourier fit).
    Costs are in nats: residuals^2 / (2 sigma^2) per angle bin + 0.5 log(#bins) per Fourier
    coefficient (BIC) + 0.5 log(n) per spine coordinate. sigma = the robust scatter of the body's
    bin radii about its envelope (floored at h). The piece is a child when H1 is shorter. A full
    ring is absorbed by a0, so no angle-coverage rule is needed; tiny pieces cannot pay for a spine,
    so no size rule beyond `min_child_pts` (a spine needs points).

  * Applicability gate. A swept primitive is the wrong model for many objects (chairs, cabinets).
    u = fraction of points neither inside the body's fitted envelope (+ margin) nor in an accepted
    child. When u > gate, the object falls back to sweep events along the longest PCA axis (the
    best method on non-swept categories so far). gate = inf disables it.

  * Child cuts: children are swept with the same MDL Chebyshev partition as the root (v3 never cut
    children).
"""
import numpy as np

from .bitsi2.section import _basis, fit_section
from .bitsi2.tree import Node, _components
from .bitsi3 import BITSI3
from scipy.spatial import cKDTree


def section_code(rho, phi, sl, slabs, sigma, n_max=4, nb=24, q=0.9):
    """Code length (nats) of the outer radius per angle bin, slab by slab, under the best Fourier
    order <= n_max (plain least squares, no rejection: what is not fit is paid for)."""
    bc = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
    pb = np.clip(((phi + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    total = 0.0
    order = np.argsort(sl, kind="stable")
    sls = sl[order]
    for k in slabs:
        idx = order[np.searchsorted(sls, k):np.searchsorted(sls, k, side="right")]
        if len(idx) < 3:
            continue
        cnt = np.bincount(pb[idx], minlength=nb)
        ok = cnt >= 1
        y = np.array([np.quantile(rho[idx][pb[idx] == b], q) for b in np.flatnonzero(ok)])
        best = np.inf
        for n in range(int(min(n_max, max(0, (len(y) - 1) // 3))) + 1):
            A = _basis(bc[ok], n)
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            rss = float(np.sum((y - A @ c) ** 2))
            best = min(best, rss / (2 * sigma ** 2) + 0.5 * (2 * n + 1) * np.log(max(len(y), 2)))
        total += best
    return total


class BITSI4(BITSI3):
    # Defaults = the variant selected on dev + ooddev by the section-20 rule: children by code
    # length, no child cuts, no applicability gate (the gate and child cuts lowered the objective).
    def __init__(self, children="mdl", gate=np.inf, cand_margin_h=2.0, min_child_pts=20, max_cands=12,
                 child_cuts=False, up=None, up_prior=0.05, up_cone=0.0, **kw):
        super().__init__(child_cuts=child_cuts, **kw)
        # up_prior (robot setting, needs up; default 0.05 = selected in RESULTS.md section 32): None = the
        # gravity axis is a third candidate that must beat
        # the others; a number delta = PREFER the gravity-aligned axis unless another candidate leaves a
        # residual fraction smaller by more than delta (inf = always gravity-aligned).
        # up_cone (deg): the gravity-aligned candidate is the best symmetry axis (normal-line score)
        # within this cone around up (0 = exactly vertical).
        self.up_prior, self.up_cone = up_prior, up_cone
        # up: gravity direction known to a robot (the table normal). When given, a vertical axis
        # through the point the normal lines pass closest to is offered as a third root-spine
        # candidate (still chosen by least residual). Off by default: on canonical full clouds it
        # would leak the dataset's up axis.
        self.up = None if up is None else np.asarray(up, float) / np.linalg.norm(up)
        self.children_mode, self.gate = children, gate
        self.cand_margin_h, self.min_child_pts, self.max_cands = cand_margin_h, min_child_pts, max_cands
        self.child_log = []

    # ------------------------------------------------------------------ children by code length
    def _sigma(self, se, sl):
        """Robust scatter of the body's angle-bin outer radii about its fitted envelope."""
        nb = 24
        pb = np.clip(((se.phi + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
        key = sl * nb + pb
        r = se.rho - se.rho_hat
        # per (slab, bin): residual of the bin's outer radius = high quantile of point residuals
        o = np.argsort(key, kind="stable")
        ks, starts = np.unique(key[o], return_index=True)
        vals = np.array([np.quantile(v, 0.9) for v in np.split(r[o], starts[1:])])
        vals = vals[vals < np.quantile(vals, 0.9)]  # protrusions are what we are testing
        mad = 1.4826 * np.median(np.abs(vals - np.median(vals))) if len(vals) > 5 else self.h
        return max(mad, self.h)

    def _mdl_children(self, node, P, N):
        se, sp = node.section, node.spine
        r = se.rho - se.rho_hat
        cand = np.flatnonzero(r > self.cand_margin_h * self.h)
        if len(cand) == 0:
            return []
        ncomp, comp = _components(P[cand], self.graph_h * self.h)
        sizes = np.bincount(comp, minlength=ncomp)
        sigma = self._sigma(se, sp.slab)
        kids = []
        for c in np.argsort(-sizes)[:self.max_cands]:
            if sizes[c] < self.min_child_pts:
                break
            m = cand[comp == c]
            slabs = np.unique(sp.slab[m])
            in_sl = np.isin(sp.slab, slabs)
            body = in_sl.copy()
            body[m] = False
            # H0: one cross-section per slab covers body + piece
            h0 = section_code(se.rho[in_sl], se.phi[in_sl], sp.slab[in_sl], slabs, sigma, self.n_max)
            # H1: body alone + the piece as its own swept primitive
            h1 = section_code(se.rho[body], se.phi[body], sp.slab[body], slabs, sigma, self.n_max)
            try:
                csp, cse = self._fit_primitive(P[m], N[m], self.child_spine)
            except Exception:
                continue
            h1 += section_code(cse.rho, cse.phi, csp.slab, np.unique(csp.slab), sigma, self.n_max)
            h1 += 0.5 * 3 * csp.n_slabs * np.log(len(m))
            self.child_log.append(dict(n=len(m), h0=h0, h1=h1, depth=node.depth))
            if h1 < h0:
                kids.append(m)
        return kids

    def _up_axis(self, P, N):
        """The gravity-aligned candidate axis: exactly up, or the best symmetry axis within up_cone."""
        if not self.up_cone:
            return self.up
        from .axes import _revolution_score
        rng = np.random.default_rng(0)
        u = np.cross(self.up, [1.0, 0, 0] if abs(self.up[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(self.up, u)
        th = np.deg2rad(self.up_cone) * np.sqrt(rng.uniform(0, 1, 300))
        ph = rng.uniform(0, 2 * np.pi, 300)
        dirs = np.vstack([self.up, np.cos(th)[:, None] * self.up + np.sin(th)[:, None] * (np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v)])
        Nn = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
        sc, _ = _revolution_score(P - P.mean(0), Nn, dirs, 0.02 * np.linalg.norm(np.ptp(P, 0)))
        return dirs[int(np.argmax(sc))]

    def _fit_primitive(self, P, N, kind):
        if kind != "auto" or self.up is None:
            return super()._fit_primitive(P, N, kind)
        from .bitsi2.section import fit_section
        from .bitsi2.spine import straight_spine
        a_up = self._up_axis(P, N)
        upc = straight_spine(P, a_up, vertical_centre(P, N, a_up), self.h * self.spacing_h, kind="revolution")
        others = [self._spine(P, N, k) for k in ("revolution", "pca")]
        fits = []
        for sp in [upc] + others:
            se = fit_section(P, sp, self.h, self.n_max)
            fits.append((sp, se, float(np.mean(se.rho - se.rho_hat > self.margin_h * self.h))))
        if self.up_prior is not None:   # prefer the gravity-aligned axis unless clearly worse
            alt = min(fits[1:], key=lambda f: f[2])
            best = alt if alt[2] < fits[0][2] - self.up_prior else fits[0]
            return best[0], best[1]
        best = None                     # the earlier rule: gravity axis as a third candidate
        for sp, se, rf in fits[1:] + fits[:1]:
            if best is None or rf < best[2] - 0.01:
                best = (sp, se, rf)
        return best[0], best[1]

    # ------------------------------------------------------------------ one node (BITSI2._fit with
    # the child test swapped; everything else unchanged)
    def _fit(self, idx, depth, kind, parent_body=None):
        if self.children_mode != "mdl":
            return super()._fit(idx, depth, kind, parent_body)
        P, N = self.P[idx], self.N[idx]
        node = Node(idx=idx, depth=depth)
        node.spine, node.section = self._fit_primitive(P, N, kind)
        r = node.section.rho - node.section.rho_hat
        node.residual_frac = float(np.mean(r > self.margin_h * self.h))
        child_local = self._mdl_children(node, P, N) if depth < self.max_depth else []
        own = np.ones(len(idx), bool)
        for m in child_local:
            own[m] = False
        node.body = idx[own]
        if (depth == 0 or self.child_cuts) and own.sum() >= 2 * self.min_len:
            from .bitsi2.tree import _sub_spine
            sp, se = _sub_spine(node.spine, own), node.section
            cuts, ev = self._cuts(sp, se, P[own], N[own], self.C[idx][own] if self.C is not None else None, own)
            node.cuts, node.events = cuts, ev
            seg = np.searchsorted(np.asarray(cuts, int), sp.slab, side="left")
        else:
            seg = np.zeros(own.sum(), int)
        node.seg_labels = self.next_label + seg
        self.next_label += int(seg.max()) + 1 if len(seg) else 0
        self.labels[node.body] = node.seg_labels
        tb = cKDTree(P[own]) if own.any() else None
        for m in child_local:
            gi = idx[m]
            if tb is not None:
                d, _ = tb.query(self.P[gi])
                contact = gi[d < 2.0 * self.h]
            else:
                contact = gi[:0]
            size = np.linalg.norm(np.ptp(self.P[gi], 0))
            nc, cc = _components(self.P[contact], max(self.graph_h * self.h, 0.2 * size))
            n_att = int((np.bincount(cc, minlength=nc) >= 3).sum()) if nc else 0
            child = self._fit(gi, depth + 1, self.child_spine)
            child.n_attach = n_att
            node.children.append(child)
        return node

    # ------------------------------------------------------------------ public
    def fit(self, points, normals, colors=None):
        self.child_log = []
        labels, root = super().fit(points, normals, colors)
        in_child = np.zeros(len(self.P), bool)
        for ch in root.children:
            in_child[ch.idx] = True
        r = root.section.rho - root.section.rho_hat
        self.unexplained = float(np.mean(~in_child & (r > self.margin_h * self.h)))
        self.fallback = bool(self.unexplained > self.gate)
        if self.fallback:
            labels = hog_labels(self.P, self.N, self.C)
        return labels, root


def vertical_centre(P, N, up, iters=3):
    """Point on a vertical axis that the horizontal projections of the normal lines pass closest to
    (trimmed least squares)."""
    u = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    p2 = np.stack([P @ u, P @ v], 1)
    n2 = np.stack([N @ u, N @ v], 1)
    ln = np.linalg.norm(n2, axis=1)
    ok = ln > 0.3                                   # normals that are mostly horizontal
    c2 = p2.mean(0)
    if ok.sum() >= 10:
        A = np.stack([-n2[ok, 1], n2[ok, 0]], 1) / ln[ok, None]
        b = np.sum(A * p2[ok], 1)
        w = np.ones(len(b), bool)
        for _ in range(iters):
            c2 = np.linalg.lstsq(A[w], b[w], rcond=None)[0]
            r = np.abs(A @ c2 - b)
            w = r <= np.quantile(r, 0.7)
    return c2[0] * u + c2[1] * v + float(np.median(P @ up)) * up


def hog_labels(P, N, C):
    from .profiles import pca_axes
    from .sweep_events import events_along
    ev, prof = events_along(P, pca_axes(P)[0], N, C, ("hist",))
    return np.searchsorted(np.sort([c.t for c in ev]), prof["coord"])
