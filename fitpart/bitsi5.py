"""BITSI v5 experiments (not a validated replacement for v4).

Partial-cloud cut candidate: BITSI5(axis_rule="residual", spine_coding="free",
up=(0, 0, 1), partial_cuts=True). This keeps v4's gravity prior and child test,
but fits observed angular radii with masked, gap-aware MDL; see partial_sections.py.
The historical defaults below are preserved for reproducibility.

Earlier experiments: BITSI v4 with the root-axis choice and the child test made robust to incomplete
cross-sections (axis_rule "normals"), or with a gripper-aware root axis (axis_rule "gripper": the
bounding-box axis along which a parallel-jaw gripper can wrap the object, RESULTS.md section 25).
Original motivation of v5:
cross-sections (eye-in-hand partial views), RESULTS.md section 24.

  * Root axis by normal-line consistency. v4 keeps the straight-spine candidate (symmetry axis, PCA
    axis, optionally the gravity axis) that leaves the fewest points outside its fitted envelope. On a
    one-sided partial surface a wrong axis can leave fewer points outside (the envelope only has to
    cover the half that was seen). On a swept surface every observed normal line passes near the axis
    no matter how much of the surface is missing, so v5 scores each candidate by the fraction of
    points whose normal line passes within tol of it (axes._revolution_score) and keeps the best one
    when that fraction is >= axis_tau; otherwise (no candidate is a credible sweep axis: boxes,
    furniture) it falls back to v4's least-residual rule.

  * Child spine coded as a smooth curve. v4 charged a child's spine 3 free numbers per slab
    (0.5 log n each), which for a long thin handle was most of the child's code length and made the
    child test reject handles. v5 codes the spine centres as one Chebyshev series per coordinate
    (degree <= 8 by BIC, residuals in units of the same sigma as the cross-sections): a smooth handle
    costs a few coefficients, an irregular body fragment still pays for its wiggles.
"""
import numpy as np
from numpy.polynomial import chebyshev as Ch

from .axes import _revolution_score
from .bitsi2.section import fit_section
from .bitsi2.spine import straight_spine
from .bitsi4 import BITSI4, section_code, vertical_centre
from .bitsi2.tree import _components


def obb_axes(P):
    """Columns = axes of the minimum-volume oriented bounding box (PCA axes if Qhull fails)."""
    import open3d as o3d
    try:
        box = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float))).get_minimal_oriented_bounding_box()
        return np.asarray(box.R)
    except Exception:
        c = P - P.mean(0)
        return np.linalg.svd(c, full_matrices=False)[2].T


def spine_code(centres, sigma, max_deg=8):
    """Code length (nats) of a spine's slab centres as one Chebyshev series per coordinate."""
    S = len(centres)
    if S < 2:
        return 0.0
    x = np.linspace(-1, 1, S)
    best = np.inf
    for k in range(int(min(max_deg, S - 1)) + 1):
        V = Ch.chebvander(x, k)
        c, *_ = np.linalg.lstsq(V, centres, rcond=None)
        rss = float(np.sum((centres - V @ c) ** 2))
        best = min(best, rss / (2 * sigma ** 2) + 0.5 * 3 * (k + 1) * np.log(max(S, 2)))
    return best


class BITSI5(BITSI4):
    def __init__(self, axis_rule="normals", axis_tau=0.25, spine_coding="cheb", gripper_opening=0.08, scale=1.0,
                 n_wrap_slabs=24, partial_cuts=False, partial_bin_pooling="mean", **kw):
        super().__init__(**kw)
        self.axis_rule, self.axis_tau, self.spine_coding = axis_rule, axis_tau, spine_coding
        self.gripper_opening, self.scale, self.n_wrap_slabs = gripper_opening, scale, n_wrap_slabs
        self.partial_cuts = partial_cuts
        if partial_bin_pooling not in ("mean", "sum"):
            raise ValueError("partial_bin_pooling must be mean or sum")
        self.partial_bin_pooling = partial_bin_pooling

    def fit(self, points, normals, colors=None):
        self.trace = []
        self.axis_scores = []
        return super().fit(points, normals, colors)

    def _cuts(self, sp, se, P, N, C, own=None):
        if not self.partial_cuts:
            return super()._cuts(sp, se, P, N, C, own)
        if tuple(self.channels) != ("section",):
            raise ValueError("partial_cuts currently supports section channels only")
        from .partial_sections import observed_radii, masked_standardize, masked_partition
        rho = se.rho if own is None else se.rho[own]
        phi = se.phi if own is None else se.phi[own]
        Y = observed_radii(rho, phi, sp.slab, sp.n_slabs)
        Ys = masked_standardize(Y, self.h)
        # Angular bins observe the same surface and are strongly correlated. Do
        # not count a coherent radius change 24 times as independent evidence.
        # A fixed divisor avoids rescaling the signal when visibility changes.
        if self.partial_bin_pooling == "mean":
            Ys /= np.sqrt(Y.shape[1])
        cuts, code = masked_partition(Ys, self.min_len, self.penalty, self.max_deg)
        self.trace.append(dict(Y=Y, observed=np.isfinite(Y), cuts=cuts,
                               code=code, edges=sp.edges, mode="observed_radii"))
        return cuts, [(c, 0.0, "observed_mdl") for c in cuts]

    # ------------------------------------------------------------------ root axis
    def _fit_primitive(self, P, N, kind):
        if kind == "auto" and self.axis_rule == "gripper":
            return self._gripper_axis(P, N)
        if kind != "auto" or self.axis_rule != "normals":
            return super()._fit_primitive(P, N, kind)
        cands = [self._spine(P, N, k) for k in ("revolution", "pca")]
        if self.up is not None:
            cands.append(straight_spine(P, self.up, vertical_centre(P, N, self.up), self.h * self.spacing_h,
                                        kind="revolution"))
        Nn = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
        c0 = P.mean(0)
        tol = 0.02 * np.linalg.norm(np.ptp(P, 0))
        fits = []
        for sp in cands:
            a = sp.tangents[0]
            s = float(_revolution_score(P - c0, Nn, a[None] / np.linalg.norm(a), tol)[0][0])
            se = fit_section(P, sp, self.h, self.n_max)
            rf = float(np.mean(se.rho - se.rho_hat > self.margin_h * self.h))
            fits.append((sp, se, rf, s))
        self.axis_scores = [(f[0].kind, round(f[3], 3), round(f[2], 3)) for f in fits]
        best = max(fits, key=lambda f: f[3])
        if best[3] >= self.axis_tau:
            return best[0], best[1]
        best = None
        for f in fits:                              # v4's rule
            if best is None or f[2] < best[2] - 0.01:
                best = f
        return best[0], best[1]

    def _gripper_axis(self, P, N):
        """Root axis = the bounding-box axis along which the jaw gripper can wrap the object: slice
        along each axis of the minimum-volume OBB; a slice is wrappable when the narrower of its two
        cross-section widths (2-98% extent along the other two OBB axes, in metres) fits the jaw
        opening. Keep the axis with the largest fraction of points in wrappable slices (ties within
        0.05 -> the longer axis); if no slice wraps on any axis, fall back to v4's rule."""
        R = obb_axes(P)
        c = P.mean(0)
        X = (P - c) @ R                              # coordinates along the 3 OBB axes
        ext = np.ptp(X, 0)
        scores = []
        for j in range(3):
            o = [k for k in range(3) if k != j]
            t = X[:, j]
            edges = np.linspace(t.min(), t.max() + 1e-9, self.n_wrap_slabs + 1)
            sl = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, self.n_wrap_slabs - 1)
            wrap = np.zeros(len(P), bool)
            for k in range(self.n_wrap_slabs):
                m = sl == k
                if m.sum() < 5:
                    continue
                w = [np.subtract(*np.percentile(X[m, oo], [98, 2])) for oo in o]
                if min(w) * self.scale <= self.gripper_opening:
                    wrap[m] = True
            scores.append(float(wrap.mean()))
        self.axis_scores = [(round(sc, 3), round(float(e * self.scale), 3)) for sc, e in zip(scores, ext)]
        if max(scores) <= 0:
            return BITSI4._fit_primitive(self, P, N, "auto")
        best = max(range(3), key=lambda j: (round(scores[j] / 0.05), ext[j]))
        sp = straight_spine(P, R[:, best], None, self.h * self.spacing_h, kind="pca")
        return sp, fit_section(P, sp, self.h, self.n_max)

    # ------------------------------------------------------------------ child test
    def _mdl_children(self, node, P, N):
        if self.spine_coding != "cheb":
            return super()._mdl_children(node, P, N)
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
            h0 = section_code(se.rho[in_sl], se.phi[in_sl], sp.slab[in_sl], slabs, sigma, self.n_max)
            h1 = section_code(se.rho[body], se.phi[body], sp.slab[body], slabs, sigma, self.n_max)
            try:
                csp, cse = self._fit_primitive(P[m], N[m], self.child_spine)
            except Exception:
                continue
            h1 += section_code(cse.rho, cse.phi, csp.slab, np.unique(csp.slab), sigma, self.n_max)
            h1 += spine_code(csp.centres, sigma)
            self.child_log.append(dict(n=len(m), h0=h0, h1=h1, depth=node.depth))
            if h1 < h0:
                kids.append(m)
        return kids
