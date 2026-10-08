"""BITSI v2: a shape is a tree of swept primitives.

Each node is a primitive = spine + cross-section field r(s, phi) (Fourier in phi per slab).
  1. fit the node's primitive (spine.py, section.py)
  2. off-axis children: points sticking out of the fitted envelope by > margin, grouped into
     connected components; a component is a child when it is large enough and breaks the
     symmetry (covers <= max_cover of the angle about the spine; a full ring is coaxial)
  3. coaxial parts: the node's remaining points are cut along the spine at significant
     section / histogram events (events.py)
  4. children are fit recursively with curved (geodesic) spines, up to max_depth
  5. topology from attachments: the number of separate contact patches between a child and its
     parent; two or more = the child closes a loop with the parent (a handle you can hook)
Output: per-point labels (one per leaf segment) and the tree.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..profiles import median_nn_spacing
from .events import spine_events
from .section import fit_section
from .spine import Spine, geodesic_spine, pca_spine, revolution_spine


@dataclass
class Node:
    idx: np.ndarray
    depth: int
    spine: Spine = None
    section: object = None
    cuts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    children: list = field(default_factory=list)
    n_attach: int = 0
    residual_frac: float = 0.0
    seg_labels: np.ndarray = None   # labels of this node's own (body) points, global ids
    body: np.ndarray = None         # global indices of this node's own points

    @property
    def loop(self):
        return self.n_attach >= 2

    def describe(self, indent=0):
        sp = self.spine
        out = [" " * indent + f"{sp.kind} primitive, {len(self.idx)} pts, {len(self.cuts) + 1} segments, "
               f"residual {self.residual_frac:.2f}"
               + (f", attached at {self.n_attach} patch(es){' -> LOOP' if self.loop else ''}" if self.depth else "")]
        for c in self.children:
            out += c.describe(indent + 2)
        return out


def _components(P, radius):
    if len(P) == 0:
        return 0, np.zeros(0, int)
    pairs = cKDTree(P).query_pairs(radius, output_type="ndarray")
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    return connected_components(G, directed=False)


def _sub_spine(sp, m):
    return Spine(sp.kind, sp.s[m], sp.slab[m], sp.centres, sp.tangents, sp.edges, sp.score, sp.info)


class BITSI2:
    # Defaults = the configuration frozen after the development ablations on 9 rotationally
    # symmetric PartNet-E categories (RESULTS.md section 16): root spine chosen by least residual,
    # histogram events for coaxial cuts, children kept whole.
    def __init__(self, root_spine="auto", child_spine="geodesic", max_depth=2, n_max=4,
                 margin_h=3.0, min_part_frac=0.01, max_cover=0.5, graph_h=3.0, spacing_h=1.5,
                 cut_channels=("hist",), cut_axis="primitive", z_min=3.0, child_cuts=False,
                 use_color=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ------------------------------------------------------------------ fitting one node
    def _spine(self, P, N, kind):
        sp = self.h * self.spacing_h
        if kind == "revolution":
            return revolution_spine(P, N, sp)
        if kind == "pca":
            return pca_spine(P, sp)
        return geodesic_spine(P, sp)

    def _fit_primitive(self, P, N, kind):
        if kind == "auto":  # the straight spine whose primitive leaves the least residual
            best = None
            for k in ("revolution", "pca"):
                sp = self._spine(P, N, k)
                se = fit_section(P, sp, self.h, self.n_max)
                rf = float(np.mean(se.rho - se.rho_hat > self.margin_h * self.h))
                if best is None or rf < best[2] - 0.01:
                    best = (sp, se, rf)
            return best[0], best[1]
        sp = self._spine(P, N, kind)
        return sp, fit_section(P, sp, self.h, self.n_max)

    def _fit(self, idx, depth, kind, parent_body=None):
        P, N = self.P[idx], self.N[idx]
        node = Node(idx=idx, depth=depth)
        node.spine, node.section = self._fit_primitive(P, N, kind)
        out = node.section.rho - node.section.rho_hat > self.margin_h * self.h
        node.residual_frac = float(out.mean())
        # off-axis children
        child_local = []
        if depth < self.max_depth and out.any():
            oi = np.flatnonzero(out)
            ncomp, comp = _components(P[oi], self.graph_h * self.h)
            sizes = np.bincount(comp, minlength=ncomp)
            for c in np.argsort(-sizes):
                if sizes[c] < self.min_part_frac * self.n_total:
                    break
                m = oi[comp == c]
                bins = ((node.section.phi[m] + np.pi) / (2 * np.pi) * 36).astype(int) % 36
                if np.count_nonzero(np.bincount(bins, minlength=36)) / 36 > self.max_cover:
                    continue
                child_local.append(m)
        own = np.ones(len(idx), bool)
        for m in child_local:
            own[m] = False
        node.body = idx[own]
        # coaxial cuts of the node's own points
        if depth == 0 or self.child_cuts:
            if self.cut_axis == "pca" and own.sum() > 50:
                sp = self._spine(P[own], N[own], "pca")
                se = fit_section(P[own], sp, self.h, self.n_max) if "section" in self.cut_channels else None
            else:
                sp, se = _sub_spine(node.spine, own), node.section
            cuts, ev = self._cuts(sp, se, P[own], N[own], self.C[idx][own] if self.C is not None else None, own)
            node.cuts, node.events = cuts, ev
            seg = np.searchsorted(np.asarray(cuts, int), sp.slab, side="left")
        else:
            seg = np.zeros(own.sum(), int)
        node.seg_labels = self.next_label + seg
        self.next_label += int(seg.max()) + 1 if len(seg) else 0
        self.labels[node.body] = node.seg_labels
        # attachments and recursion
        tb = cKDTree(P[own]) if own.any() else None
        for m in child_local:
            gi = idx[m]
            if tb is not None:
                d, _ = tb.query(self.P[gi])
                contact = gi[d < 2.0 * self.h]
            else:
                contact = gi[:0]
            # contact patches are separate if they are apart by a fifth of the child's size
            size = np.linalg.norm(np.ptp(self.P[gi], 0))
            nc, cc = _components(self.P[contact], max(self.graph_h * self.h, 0.2 * size))
            n_att = int((np.bincount(cc, minlength=nc) >= 3).sum()) if nc else 0
            child = self._fit(gi, depth + 1, self.child_spine)
            child.n_attach = n_att
            node.children.append(child)
        return node

    def _cuts(self, sp, se, P, N, C, own=None):
        """Coaxial cuts of a node's own points along spine `sp` (slab boundary indices)."""
        return spine_events(sp, se, N, C, z_min=self.z_min, channels=self.cut_channels)

    # ------------------------------------------------------------------ public
    def fit(self, points, normals, colors=None):
        self.P = np.asarray(points, float)
        self.N = np.asarray(normals, float)
        self.C = None if (colors is None or not self.use_color) else np.asarray(colors, float)
        self.h = median_nn_spacing(self.P)
        self.n_total = len(self.P)
        self.labels = np.full(len(self.P), -1)
        self.next_label = 0
        self.root = self._fit(np.arange(len(self.P)), 0, self.root_spine)
        _, self.labels = np.unique(self.labels, return_inverse=True)
        return self.labels, self.root
