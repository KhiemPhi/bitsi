"""FitPart end to end: geometry proposes, the gripper constrains, language selects."""
from dataclasses import dataclass, field

import numpy as np

from . import lifting
from .affinity import detect_all_cuts, geometric_affinity, normal_smoothness
from .divisive import divisive_tree
from .feasibility import feasible
from .hierarchy import agglomerate
from .render import render_views
from .selection import (CANONICAL, label_segments, node_embeddings, node_embeddings_from_masks,
                        relevance, select)
from .superpoints import superpoint_adjacency, superpoints


@dataclass
class Config:
    # geometry
    geometry: str = "divisive"      # "divisive" (recursive cuts), "global" (draft's A_geo), "none"
    detector: str = "cheb"          # cheb | single | extrema | jump
    axis_only: bool = False
    # semantics
    n_views: int = 10
    image_size: int = 512
    up: tuple = (0.0, 1.0, 0.0)     # ShapeNetPart / PartNet are y-up
    sam_points_per_side: int = 16
    mask_mode: str = "jaccard"      # jaccard | finest
    vl_model: str = "siglip"        # siglip | clip
    beta: float = 0.5               # A_sem = beta A_mask + (1 - beta) A_feat
    lam: float = 0.5                # w = lam A_sem + (1 - lam) A_geo
    combine: str = "linear"         # linear (draft) | product | min
    long_range_tau: float = 0.85    # DINOv2 cosine for long-range edges (None = off)
    # gripper
    gripper_width: float = 0.08     # metres
    scale: float = 1.0              # metres per point-cloud unit
    # selection
    eps_r: float = 0.1             # log-odds units (temperature 100 => 0.001 in cosine)
    r_min: float = None
    min_select_frac: float = 0.01    # never select nodes below this fraction of points
    node_embedding: str = "masks"    # "masks" (IoU-matched crops) | "mean" (superpoint average)


@dataclass
class FitPartResult:
    sp: np.ndarray
    pairs: np.ndarray
    w: np.ndarray
    hier: object
    A_geo: np.ndarray = None
    A_sem: np.ndarray = None
    z_sp: np.ndarray = None
    w_sp: np.ndarray = None
    z_nodes: np.ndarray = None
    feasible: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    render: dict = None
    masks: list = None


def _up_rotation(up):
    """Rotation taking `up` to +z (renderer convention)."""
    u = np.asarray(up, float) / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), u @ z
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def divisive_pair_affinity(points, graph, sp, pairs, detector_kwargs, colors=None, normals=None):
    """A_geo from the recursive tree: 1 - (strength of the split that first separates
    the two superpoints), normalized by the strongest split."""
    detector_kwargs = dict(detector_kwargs)
    persist = detector_kwargs.pop("persist_levels", None)
    nodes, order = divisive_tree(points, graph, detector_kwargs=detector_kwargs, persist_levels=persist,
                                 colors=colors, normals=normals)
    M = sp.max() + 1
    # assign every point to its leaf, then each superpoint to its majority leaf
    leaf_of = np.zeros(len(points), dtype=int)
    stack = [0]
    while stack:
        i = stack.pop()
        if nodes[i].children:
            stack.extend(nodes[i].children)
        else:
            leaf_of[nodes[i].idx] = i
    sp_leaf = np.array([np.bincount(leaf_of[sp == s]).argmax() for s in range(M)])

    def path(i):
        p = [i]
        while nodes[p[-1]].parent >= 0:
            p.append(nodes[p[-1]].parent)
        return p[::-1]

    leaf_paths = {l: path(l) for l in np.unique(sp_leaf)}
    smax = max((nodes[i].split_strength for i in order), default=1.0) or 1.0
    A = np.ones(len(pairs))
    for k, (a, b) in enumerate(pairs):
        la, lb = sp_leaf[a], sp_leaf[b]
        if la == lb:
            continue
        pa, pb = leaf_paths[la], leaf_paths[lb]
        j = 0
        while j < min(len(pa), len(pb)) and pa[j] == pb[j]:
            j += 1
        lca = pa[j - 1]
        A[k] = 1 - nodes[lca].split_strength / smax
    return A, nodes


class FitPart:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()

    # ------------------------------------------------------------ build
    GEOMETRY_VARIANTS = {
        "divisive": ("divisive", dict(detector="cheb")),
        "divisive_jump": ("divisive", dict(detector="jump", delta=0.2)),
        "divisive_single": ("divisive", dict(detector="single")),
        "divisive_extrema": ("divisive", dict(detector="extrema")),
        "divisive_axis": ("divisive", dict(detector="cheb", axis_only=True)),
        "global": ("global", dict(detector="cheb")),
        "divisive_persist": ("divisive", dict(detector="cheb", persist_levels=2)),
        "divisive_mc": ("divisive", dict(channels=("geo", "topo", "color"))),
        "divisive_mc_geotopo": ("divisive", dict(channels=("geo", "topo"))),
        "divisive_mc_geocolor": ("divisive", dict(channels=("geo", "color"))),
        "divisive_hog": ("divisive", dict(events=("hist",))),
        "divisive_events": ("divisive", dict(events=("geo", "topo", "color", "hist"))),
    }

    def lift(self, points, normals, colors=None, semantic=True, geometry_variants=None):
        """Everything expensive, cacheable per object: superpoints, A_geo for each
        geometry variant, renders, SAM masks, A_mask (both modes), DINOv2 features,
        vision-language superpoint embeddings."""
        import time
        cfg = self.cfg
        T = {}
        t0 = time.time()
        sp, graph = superpoints(points, normals)
        pairs = superpoint_adjacency(sp, graph)
        M = sp.max() + 1
        T["superpoints"] = time.time() - t0
        L = dict(sp=sp, pairs=pairs, smooth=normal_smoothness(normals, sp, pairs), A_geo={})
        for name in geometry_variants or [cfg.geometry]:
            t0 = time.time()
            if name == "none":
                L["A_geo"][name] = np.ones(len(pairs))
                continue
            kind, det = self.GEOMETRY_VARIANTS[name]
            if kind == "divisive":
                L["A_geo"][name], _ = divisive_pair_affinity(points, graph, sp, pairs, det)
            else:
                L["A_geo"][name] = geometric_affinity(points, sp, pairs, detect_all_cuts(points, **det))
            T["geometry_" + name] = time.time() - t0
        L["timings"] = T
        if not semantic:
            return L

        t0 = time.time()
        Rup = _up_rotation(cfg.up)
        rend = render_views(points @ Rup.T, colors, normals @ Rup.T, k=cfg.n_views, size=cfg.image_size)
        T["render"] = time.time() - t0
        t0 = time.time()
        masks = lifting.run_sam(rend["images"], cfg.sam_points_per_side)
        T["sam"] = time.time() - t0
        t0 = time.time()
        sp_pix = np.stack([lifting.sp_pixel_index(p2p, sp) for p2p in rend["pix2pt"]])
        fr, cn = zip(*[lifting.mask_membership(m, s, M) for m, s in zip(masks, sp_pix)])
        L["A_mask"] = {mode: lifting.mask_affinity(fr, cn, mode=mode)[0] for mode in ("jaccard", "finest")}
        feats, _ = lifting.dino_superpoint_features(rend["images"], sp_pix, M)
        L["A_feat"] = np.clip(lifting_cos(feats), 0, 1)
        T["lift"] = time.time() - t0
        t0 = time.time()
        L["z_sp"], L["w_sp"], emb, owners = lifting.vl_superpoint_embeddings(rend["images"], masks, fr, M,
                                                                            cfg.vl_model)
        # per-crop embeddings and their 3D footprint, for mask-matched node embeddings
        L["mask_emb"] = emb.astype(np.float16)
        L["mask_frac"] = np.stack([fr[v][k] for v, k in owners]).astype(np.float16) if owners else np.zeros((0, M))
        L["mask_vis"] = np.stack([cn[v] > 0 for v, k in owners]) if owners else np.zeros((0, M), bool)
        T["vl"] = time.time() - t0
        L["images"] = rend["images"]
        L["n_masks"] = [len(m) for m in masks]
        return L

    def assemble(self, L, geometry=None, lam=None, beta=None, mask_mode=None, long_range_tau="cfg",
                 combine=None, sem_source="lifted"):
        """Affinities -> hierarchy -> node embeddings (cheap; reruns for every ablation)."""
        cfg = self.cfg
        geometry = geometry or cfg.geometry
        lam = cfg.lam if lam is None else lam
        beta = cfg.beta if beta is None else beta
        mask_mode = mask_mode or cfg.mask_mode
        tau = cfg.long_range_tau if long_range_tau == "cfg" else long_range_tau
        sp, pairs = L["sp"], L["pairs"]
        M = sp.max() + 1
        A_geo_pairs = L["A_geo"][geometry]
        res = FitPartResult(sp=sp, pairs=pairs, w=None, hier=None, timings=L["timings"])
        if "A_feat" not in L:
            w = A_geo_pairs * L["smooth"]
            res.w, res.A_geo, res.hier = w, A_geo_pairs, agglomerate(M, pairs, w)
            return res
        A_feat, A_mask = L["A_feat"], L["A_mask"][mask_mode]
        A_sem = np.where(np.isnan(A_mask), A_feat, beta * np.nan_to_num(A_mask) + (1 - beta) * A_feat)
        if sem_source != "lifted" and "pf_feat" in L:
            # PartField features as (part of) the semantic affinity: cosine of superpoint means
            F = L["pf_feat"].astype(np.float32)
            F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
            Fs = np.zeros((M, F.shape[1]))
            np.add.at(Fs, sp, F)
            Fs /= np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-8
            A_pf = np.clip(Fs @ Fs.T, 0, 1)
            A_sem = A_pf if sem_source == "partfield" else 0.5 * (A_sem + A_pf)
        all_pairs, A_geo = pairs, A_geo_pairs
        if tau is not None:
            iu = np.triu_indices(M, 1)
            far = np.stack(iu, 1)[A_feat[iu] >= tau]
            have = {tuple(p) for p in pairs}
            far = np.array([p for p in far if tuple(p) not in have]).reshape(-1, 2)
            if len(far):
                # geometry can't vouch for non-adjacent pairs; give them the median A_geo
                all_pairs = np.vstack([pairs, far])
                A_geo = np.concatenate([A_geo_pairs, np.full(len(far), np.median(A_geo_pairs))])
        sem = A_sem[all_pairs[:, 0], all_pairs[:, 1]]
        combine = combine or cfg.combine
        if combine == "product":    # a strong cut vetoes a merge however similar the semantics
            w = sem * A_geo
        elif combine == "min":
            w = np.minimum(sem, A_geo)
        else:                       # the draft's w = lam A_sem + (1 - lam) A_geo
            w = lam * sem + (1 - lam) * A_geo
        hier = agglomerate(M, all_pairs, w)
        if self.cfg.node_embedding == "masks" and "mask_emb" in L:
            z_nodes = node_embeddings_from_masks(hier, sp, L["mask_emb"], L["mask_frac"], L["mask_vis"])
        else:
            z_nodes, _ = node_embeddings(hier, L["z_sp"], L["w_sp"])
        res.pairs, res.w, res.hier = all_pairs, w, hier
        res.A_geo, res.A_sem = A_geo, A_sem
        res.z_sp, res.w_sp, res.z_nodes = L["z_sp"], L["w_sp"], z_nodes
        return res

    def build(self, points, normals, colors=None, semantic=True):
        return self.assemble(self.lift(points, normals, colors, semantic))

    # ------------------------------------------------------------ gripper
    def tag_feasible(self, res, points, normals, nodes=None):
        cfg = self.cfg
        nodes = range(res.hier.n_nodes) if nodes is None else nodes
        for nid in nodes:
            if nid in res.feasible:
                continue
            idx = np.isin(res.sp, res.hier.members[nid])
            res.feasible[nid] = feasible(points[idx], normals[idx], cfg.gripper_width, cfg.scale)
        return res.feasible

    # ------------------------------------------------------------ language
    def select(self, res, query, points, normals, use_feasibility=True):
        t = lifting.encode_text([query] + list(CANONICAL), self.cfg.vl_model)
        r = relevance(res.z_nodes, t[0], t[1:])
        sizes = np.array([np.isin(res.sp, res.hier.members[i]).sum() for i in range(res.hier.n_nodes)])
        fmask = None
        if use_feasibility:
            # only test the most relevant nodes (the feasibility test is the slow part)
            top = np.argsort(-r)[:40]
            self.tag_feasible(res, points, normals, top)
            fmask = np.zeros(len(r), bool)
            fmask[[i for i in top if res.feasible[i]]] = True
        h, status = select(r, sizes, fmask, self.cfg.eps_r, self.cfg.r_min,
                           min_size=self.cfg.min_select_frac * len(res.sp))
        return h, status, r


def lifting_cos(feats):
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return f @ f.T
