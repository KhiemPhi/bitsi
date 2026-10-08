"""Incomplete point clouds from an eye-in-hand camera, and where to look next.

Scene. The object stands upright on the table (PartNet-E is +y up) at the category's metric size
(scales.py), under a random yaw, centred at TABLE_XY in front of the robot. The robot carries a
RealSense-D405-like camera on its hand (424 x 240 depth, 87 x 58 deg FOV).

Observation (simulated depth on the object's dense surface samples). A point is seen from a camera
pose when it is in the frustum and range, survives a z-buffer of the cloud's own splats (self-
occlusion; splat side = the cloud's sampling spacing) and the surface is not seen edge-on
(|n . v| >= 0.1). Each seen point gets depth noise along its ray, sigma(z) = 0.5 mm + 2 mm * z^2
(z in m), and each view a registration error (hand-eye + kinematics: 1 mm, 0.3 deg). Views are
fused the way a TSDF does it: observations are averaged per 3 mm voxel (the fused error is then
below the sampling spacing; plain concatenation leaves a doubled ~2 mm-thick surface). The GT label
of a fused point is that of the surface sample observed closest to the voxel mean. Not simulated: specular/transparent failures,
the table plane (removed, as a real pipeline would by plane fitting), other objects.

Viewpoints. Camera on a sphere of radius view_distance(...) around the object centre, looking at
it, on the grid (az every 15 deg, el in {15, 35, 55, 75} deg); only those the robot can reach
(sim/view_reach.py) are allowed. Every strategy starts at the same HOME view, from the robot's side.

Strategies
  orbit      fixed coarse-to-fine azimuth order at el 35 (nearest reachable elevation)
  nbv_cov    next-best-view by unobserved voxels (occupancy grid; a view's gain = unknown voxels its
             rays cross before an occupied voxel) -- the standard volumetric NBV baseline
  nbv_bitsi  ours: fit BITSI v4 to what has been seen; its primitives predict the unseen surface
             (the fitted cross-section r(s, phi) on every slab of every primitive). A view's gain =
             predicted surface cells with no observation yet that face the view and are not
             occluded. Stops when < 5% of the predicted cells are unobserved.
"""
import numpy as np
from scipy.spatial import cKDTree

from .scales import CATEGORY_SIZE_M

TABLE_XY = np.array([0.55, 0.0])
IMG_W, IMG_H, HFOV, VFOV = 424, 240, 87.0, 58.0
FX = IMG_W / 2 / np.tan(np.deg2rad(HFOV / 2))
FY = IMG_H / 2 / np.tan(np.deg2rad(VFOV / 2))
MIN_RANGE, MAX_RANGE = 0.07, 1.2
CAM_BEHIND_TCP = 0.10
AZ = np.arange(0, 360, 15.0)
EL = np.array([15.0, 35.0, 55.0, 75.0])
HOME = (180.0, 35.0)  # camera between the robot (at the origin) and the object


def view_dir(az, el):
    a, e = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


# ------------------------------------------------------------------ scene
def place(P, N, category, seed=0):
    """Metric, z-up, on the table at TABLE_XY with a random yaw. Returns W, Nw, scale."""
    s = CATEGORY_SIZE_M[category] / float(np.ptp(P, 0).max())
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])        # +y up -> +z up
    yaw = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    c, si = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1.0]])
    R = Rz @ Rx
    W = (P * s) @ R.T
    Nw = N @ R.T
    lo, hi = W.min(0), W.max(0)
    W = W - np.array([(lo[0] + hi[0]) / 2 - TABLE_XY[0], (lo[1] + hi[1]) / 2 - TABLE_XY[1], lo[2]])
    return W, Nw, s


def object_frame(W):
    lo, hi = W.min(0), W.max(0)
    centre = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))
    return centre, diag


def view_distance(diag):
    """Fit the object in the vertical FOV with margin; keep the fingertips (CAM_BEHIND_TCP ahead of
    the camera) >= 3 cm outside the bounding sphere."""
    d = max(0.5 * diag / np.tan(np.deg2rad(VFOV / 2 - 4)), 0.5 * diag + CAM_BEHIND_TCP + 0.03, 0.15)
    return round(float(d), 2)


def camera_pose(centre, dist, az, el):
    d = view_dir(az, el)
    pos = centre + dist * d
    f = -d
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    return pos, np.stack([r, dn, f])  # rows: right, down, forward


# ------------------------------------------------------------------ sensing
def spacing(W):
    return float(np.median(cKDTree(W).query(W, k=2)[0][:, 1]))


def visible(W, Nw, cam, Rc, h, extra_occluders=None):
    """Indices of points of W seen from (cam, Rc) under self-occlusion; `extra_occluders` (M,3)
    only occlude. Splat z-buffer at IMG_W x IMG_H."""
    X = (W - cam) @ Rc.T
    allp = X if extra_occluders is None else np.vstack([X, (extra_occluders - cam) @ Rc.T])
    z = allp[:, 2]
    ok = (z > MIN_RANGE) & (z < MAX_RANGE)
    u = FX * allp[:, 0] / np.where(ok, z, 1) + IMG_W / 2
    v = FY * allp[:, 1] / np.where(ok, z, 1) + IMG_H / 2
    ok &= (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    zb = np.full(IMG_W * IMG_H, np.inf)
    idx = np.flatnonzero(ok)
    rad = np.clip(np.ceil(0.5 * FX * h / z[idx]).astype(int), 0, 6)
    for r in np.unique(rad):
        sel = idx[rad == r]
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                uu = np.clip(u[sel].astype(int) + du, 0, IMG_W - 1)
                vv = np.clip(v[sel].astype(int) + dv, 0, IMG_H - 1)
                np.minimum.at(zb, vv * IMG_W + uu, z[sel])
    n = len(W)
    own = idx[idx < n]
    pix = v[own].astype(int) * IMG_W + u[own].astype(int)
    seen = z[own] <= zb[pix] + max(1.5 * h, 0.003)
    own = own[seen]
    if Nw is not None:
        ray = W[own] - cam
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        own = own[np.abs(np.sum(Nw[own] * ray, 1)) >= 0.1]
    return own


def observe(W, Nw, cam, Rc, h, rng):
    """One noisy depth view: (indices of seen surface samples, their measured world positions)."""
    idx = visible(W, Nw, cam, Rc, h)
    ray = W[idx] - cam
    z = ray @ Rc[2]
    dist = np.linalg.norm(ray, axis=1)
    sig = 0.0005 + 0.002 * z ** 2
    meas = cam + ray * ((dist + rng.normal(0, sig)) / dist)[:, None]
    ang = rng.normal(0, np.deg2rad(0.3), 3)              # per-view registration error
    K = np.array([[0, -ang[2], ang[1]], [ang[2], 0, -ang[0]], [-ang[1], ang[0], 0]])
    c0 = W.mean(0)
    meas = (meas - c0) @ (np.eye(3) + K).T + c0 + rng.normal(0, 0.001, 3)
    return idx, meas


class Fused:
    """Running fusion of views: observations averaged per voxel (TSDF-like), at most max_points."""

    def __init__(self, h, voxel=0.003, max_points=20000, seed=0):
        self.h, self.voxel, self.max_points = h, max(voxel, h), max_points
        self.idx, self.pts, self.cam = np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3))
        self.rng = np.random.default_rng(seed)

    def add(self, idx, pts, cam):
        self.idx = np.concatenate([self.idx, idx])
        self.pts = np.vstack([self.pts, pts])
        self.cam = np.vstack([self.cam, np.repeat(cam[None], len(idx), 0)])

    def cloud(self):
        """(sample indices, points, normals oriented towards their cameras), one point per voxel."""
        import open3d as o3d
        key = np.floor(self.pts / self.voxel).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        nv = len(cnt)
        mean = np.zeros((nv, 3))
        np.add.at(mean, inv, self.pts)
        mean /= cnt[:, None]
        cam = np.zeros((nv, 3))
        np.add.at(cam, inv, self.cam)
        cam /= cnt[:, None]
        # representative surface sample: the observation closest to its voxel mean
        d = np.linalg.norm(self.pts - mean[inv], axis=1)
        order = np.lexsort((d, inv))
        first = order[np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]]
        idx = np.empty(nv, int)
        idx[inv[first]] = self.idx[first]
        keep = np.arange(nv)
        if nv > self.max_points:
            keep = np.sort(self.rng.choice(nv, self.max_points, replace=False))
        pts, idx, cam = mean[keep], idx[keep], cam[keep]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        n = np.asarray(pc.normals)
        n *= np.sign(np.sum(n * (cam - pts), 1))[:, None]
        return idx, pts, n


# ------------------------------------------------------------------ strategies
def nearest_reachable(az, el, reach):
    best = min(reach, key=lambda r: (abs(((r[0] - az + 180) % 360) - 180) / 15) ** 2 + ((r[1] - el) / 20) ** 2)
    return tuple(best)


def orbit_sequence(reach, k):
    order = [180, 0, 90, 270, 135, 315, 45, 225, 157.5, 337.5, 67.5, 247.5]
    seq = []
    for az in order:
        v = nearest_reachable(HOME[0] + az - 180, 35.0, reach)
        if v not in seq:
            seq.append(v)
        if len(seq) == k:
            break
    return seq


class VoxelNBV:
    """Occupancy grid (unknown / free / occupied) over the object's bounding sphere."""

    def __init__(self, centre, diag, n=40):
        self.lo = centre - 0.6 * diag
        self.vs = 1.2 * diag / n
        self.n = n
        self.state = np.zeros((n, n, n), np.int8)   # 0 unknown, 1 free, 2 occupied

    def _cells(self, X):
        g = np.floor((X - self.lo) / self.vs).astype(int)
        ok = np.all((g >= 0) & (g < self.n), 1)
        return g, ok

    def _rays(self, cam, Rc, step_px=6):
        uu, vv = np.meshgrid(np.arange(0, IMG_W, step_px), np.arange(0, IMG_H, step_px))
        d = np.stack([(uu.ravel() - IMG_W / 2) / FX, (vv.ravel() - IMG_H / 2) / FY, np.ones(uu.size)], 1)
        d = d @ Rc
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def integrate(self, cam, Rc, pts):
        g, ok = self._cells(pts)
        self.state[tuple(g[ok].T)] = 2
        ts = np.arange(0, MAX_RANGE, self.vs / 2)
        # free space (1) to just before each observed point
        L = np.linalg.norm(pts - cam, axis=1)
        dirs = (pts - cam) / L[:, None]
        self._carve(cam, dirs, ts, L - 1.5 * self.vs)
        # (2) along every pixel ray only up to the first OCCUPIED voxel: a ray that meets the object stops
        # there (the earlier version carved every pixel ray to MAX_RANGE, i.e. through the object, so the
        # hidden side was wrongly marked free -- RESULTS.md section 33)
        pdirs = self._rays(cam, Rc)
        for a in range(0, len(pdirs), 512):
            d = pdirs[a:a + 512]
            X = cam + d[:, None, :] * ts[None, :, None]
            gg, okc = self._cells(X.reshape(-1, 3))
            st = np.full(len(gg), -1)
            st[okc] = self.state[tuple(gg[okc].T)]
            st = st.reshape(len(d), len(ts))
            hit = np.where((st == 2).any(1), np.argmax(st == 2, 1), len(ts))
            self._carve(cam, d, ts, ts[np.minimum(hit, len(ts) - 1)] - 1.5 * self.vs + (hit == len(ts)) * 1e9)

    def _carve(self, cam, dirs, ts, limit):
        for t0 in range(0, len(ts), 64):
            t = ts[t0:t0 + 64]
            X = cam + dirs[:, None, :] * t[None, :, None]
            m = t[None, :] < limit[:, None]
            g, okc = self._cells(X.reshape(-1, 3))
            okc &= m.ravel()
            gg = g[okc]
            cur = self.state[tuple(gg.T)]
            gg = gg[cur == 0]
            self.state[tuple(gg.T)] = 1

    def gain(self, cam, Rc):
        dirs = self._rays(cam, Rc, step_px=8)
        ts = np.arange(0, MAX_RANGE, self.vs / 2)
        X = cam + dirs[:, None, :] * ts[None, :, None]
        g, ok = self._cells(X.reshape(-1, 3))
        st = np.full(len(g), -1)
        st[ok] = self.state[tuple(g[ok].T)]
        st = st.reshape(len(dirs), len(ts))
        hit = np.where((st == 2).any(1), np.argmax(st == 2, 1), len(ts))
        before = np.arange(len(ts))[None, :] < hit[:, None]
        # unknown voxels crossed before the first occupied one (cells counted once)
        cells = g.reshape(len(dirs), len(ts), 3)[(st == 0) & before]
        return len(np.unique(cells, axis=0)) if len(cells) else 0


def predicted_surface(root, h):
    """Points + outward normals of every primitive's fitted cross-section (24 angle bins per
    slab), and for each the number of observed points within 2h of it, per (primitive, slab, bin)."""
    from .bitsi2.section import _basis, _frames
    out_p, out_n, out_obs = [], [], []
    stack = [root]
    while stack:
        nd = stack.pop()
        stack += nd.children
        sp, se = nd.spine, nd.section
        if sp is None or se is None:
            continue
        S = sp.n_slabs
        U, V = _frames(sp.tangents)
        phis = -np.pi + (np.arange(24) + 0.5) * 2 * np.pi / 24
        B = _basis(phis, se.n_max)                          # (24, D)
        r = se.coef @ B.T                                   # (S, 24)
        dirs = np.cos(phis)[None, :, None] * U[:, None, :] + np.sin(phis)[None, :, None] * V[:, None, :]
        Pp = sp.centres[:, None, :] + np.maximum(r, 0)[:, :, None] * dirs
        pb = np.clip(((se.phi + np.pi) / (2 * np.pi) * 24).astype(int), 0, 23)
        near = np.abs(se.rho - se.rho_hat) < 3 * h
        obs = np.zeros((S, 24))
        np.add.at(obs, (sp.slab[near], pb[near]), 1)
        out_p.append(Pp.reshape(-1, 3))
        out_n.append(dirs.reshape(-1, 3))
        out_obs.append(obs.ravel())
    return np.vstack(out_p), np.vstack(out_n), np.concatenate(out_obs)


def bitsi_gain(pred_p, pred_n, unobs, cam, Rc, observed_pts, h):
    """Unobserved predicted cells that face the camera and are not hidden by what has been seen or
    by the predicted surface itself."""
    cand = np.flatnonzero(unobs)
    if len(cand) == 0:
        return 0
    face = np.sum(pred_n[cand] * (cam - pred_p[cand]), 1) > 0
    cand = cand[face]
    if len(cand) == 0:
        return 0
    occ = np.vstack([observed_pts, pred_p[~unobs]])
    vis = visible(pred_p[cand], None, cam, Rc, max(h, 0.004), extra_occluders=occ)
    return len(vis)


class BoxNBV(VoxelNBV):
    """Bounding-box completion (user's proposal): fit a gravity-aligned box to what has been seen
    (z = up; x, y = principal axes of the footprint; 2-98% extents) and inflate it by `margin` of its
    extent on every side but the bottom (a partial view underestimates the object: when the points
    cling to one face, the rest of the box is still unknown). The occupancy grid (unknown / free /
    occupied, by ray casting as in VoxelNBV) covers the object's bounding sphere; the gain of a view is
    the number of UNKNOWN voxels INSIDE THE CURRENT BOX its rays reach before an occupied voxel, so the
    robot goes to the faces that are still empty. The box is refitted after every view; looking stops
    when < stop of the box's voxels are unknown."""

    def __init__(self, centre, diag, n=40, margin=0.2, stop=0.05):
        super().__init__(centre, diag, n)
        self.margin, self.stop = margin, stop
        self.box = None

    def fit_box(self, pts, cams=None):
        """cams: camera positions so far. A partial view sees the faces towards the cameras, so the
        footprint is squared (as deep as it is wide: objects are rarely much thinner than wide) by
        extending it AWAY from the cameras, where the unseen faces are; then the margin."""
        c = pts.mean(0)
        xy = pts[:, :2] - c[:2]
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        R = np.eye(3)
        R[:2, 0], R[:2, 1] = vt[0], vt[1]
        X = (pts - c) @ R
        lo, hi = np.percentile(X, 2, axis=0), np.percentile(X, 98, axis=0)
        ext = hi - lo
        if cams is not None and len(cams):
            side = (np.asarray(cams).mean(0) - c) @ R          # where the cameras are, in box axes
            w = ext[:2].max()
            for j in range(2):
                miss = w - ext[j]
                if miss > 0:
                    if side[j] > 0:
                        lo[j] -= miss                          # cameras on +j: the unseen side is -j
                    else:
                        hi[j] += miss
            ext = hi - lo
        lo, hi = lo - self.margin * ext, hi + self.margin * ext
        lo[2] = max(lo[2] + self.margin * ext[2], TABLE_FLOOR - c[2])    # no margin into the table
        self.box = (c, R, lo, hi)

    def _in_box(self, X):
        c, R, lo, hi = self.box
        Y = (X - c) @ R
        return np.all((Y >= lo) & (Y <= hi), axis=-1)

    def unknown_fraction(self):
        g = np.stack(np.meshgrid(*[np.arange(self.n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        centres = self.lo + (g + 0.5) * self.vs
        inb = self._in_box(centres)
        st = self.state.reshape(-1)
        return float(np.mean(st[inb] == 0)) if inb.any() else 0.0

    def gain(self, cam, Rc):
        dirs = self._rays(cam, Rc, step_px=8)
        ts = np.arange(0, MAX_RANGE, self.vs / 2)
        X = cam + dirs[:, None, :] * ts[None, :, None]
        g, ok = self._cells(X.reshape(-1, 3))
        st = np.full(len(g), -1)
        st[ok] = self.state[tuple(g[ok].T)]
        st = st.reshape(len(dirs), len(ts))
        hit = np.where((st == 2).any(1), np.argmax(st == 2, 1), len(ts))
        before = np.arange(len(ts))[None, :] < hit[:, None]
        inb = self._in_box(X).reshape(len(dirs), len(ts))
        cells = g.reshape(len(dirs), len(ts), 3)[(st == 0) & before & inb]
        return len(np.unique(cells, axis=0)) if len(cells) else 0


TABLE_FLOOR = 0.0
