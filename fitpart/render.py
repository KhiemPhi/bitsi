"""Point-splat renderer with a z-buffer (Sec. 4.2, "Rendering").

Renders a point cloud from K viewpoints on a sphere. Each view returns an RGB
image, the index of the point seen at each pixel (-1 = background), and per-point
visibility, which is all the lifting step needs. Splats are square, sized from the
median point spacing so surfaces render closed.
"""
import numpy as np
import torch


def sphere_viewpoints(k, elevations=(30.0, -20.0)):
    """k camera directions alternating between the given elevations, evenly in azimuth."""
    dirs = []
    for i in range(k):
        el = np.deg2rad(elevations[i % len(elevations)])
        az = 2 * np.pi * i / k
        dirs.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(dirs)


def look_at(eye_dir, up=(0, 0, 1)):
    """Rotation whose rows are camera right, down, forward (forward = towards the origin)."""
    f = -np.asarray(eye_dir, float)
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    if abs(f @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


@torch.no_grad()
def render_views(points, colors=None, normals=None, k=10, size=512, splat=None, fov_margin=1.1,
                 device="cuda", light=True):
    """Render k views.

    points (N,3), colors (N,3) in [0,1] or None (grey), normals (N,3) or None.
    Returns dict with images (k,H,W,3) uint8, pix2pt (k,H,W) int64, visible (k,N) bool,
    and the camera rotations.
    """
    P = torch.as_tensor(points, dtype=torch.float32, device=device)
    c = P.mean(0)
    P = P - c
    radius = P.norm(dim=1).max().item()
    N = len(P)
    if colors is None:
        col = torch.full((N, 3), 0.7, device=device)
    else:
        col = torch.as_tensor(colors, dtype=torch.float32, device=device)
        if col.max() > 1.5:
            col = col / 255.0
    nrm = None if normals is None else torch.as_tensor(normals, dtype=torch.float32, device=device)

    # splat half-width in pixels from median spacing (orthographic projection)
    scale = size / (2 * radius * fov_margin)
    if splat is None:
        from scipy.spatial import cKDTree
        sub = points[np.random.default_rng(0).choice(N, min(N, 4000), replace=False)]
        dnn, _ = cKDTree(points).query(sub, k=2)
        splat = int(np.clip(np.ceil(0.75 * np.median(dnn[:, 1]) * scale), 1, 8))
    offs = torch.stack(torch.meshgrid(torch.arange(-splat, splat + 1, device=device),
                                      torch.arange(-splat, splat + 1, device=device), indexing="ij"), -1).reshape(-1, 2)

    views = sphere_viewpoints(k)
    images, pix2pts, visible, rots = [], [], [], []
    for v in views:
        R = torch.as_tensor(look_at(v), dtype=torch.float32, device=device)
        q = P @ R.T  # x right, y down, z forward (depth)
        u = (q[:, 0] * scale + size / 2).round().long()
        w = (q[:, 1] * scale + size / 2).round().long()
        z = q[:, 2]
        # splat every point to its footprint
        uu = (u[:, None] + offs[None, :, 0]).reshape(-1)
        ww = (w[:, None] + offs[None, :, 1]).reshape(-1)
        zz = z[:, None].expand(-1, len(offs)).reshape(-1)
        pid = torch.arange(N, device=device)[:, None].expand(-1, len(offs)).reshape(-1)
        ok = (uu >= 0) & (uu < size) & (ww >= 0) & (ww < size)
        uu, ww, zz, pid = uu[ok], ww[ok], zz[ok], pid[ok]
        lin = ww * size + uu
        zbuf = torch.full((size * size,), float("inf"), device=device)
        zbuf.scatter_reduce_(0, lin, zz, reduce="amin")
        win = zz <= zbuf[lin] + 1e-6
        p2p = torch.full((size * size,), -1, dtype=torch.long, device=device)
        p2p[lin[win]] = pid[win]
        # a point is visible if its own centre pixel depth is within tolerance of the z-buffer
        tol = 2.0 * splat / scale
        cen_ok = (u >= 0) & (u < size) & (w >= 0) & (w < size)
        cen = (w.clamp(0, size - 1) * size + u.clamp(0, size - 1))
        vis = cen_ok & (z <= zbuf[cen] + tol)
        # shade
        img = torch.ones((size * size, 3), device=device)
        hit = p2p >= 0
        shade = col[p2p[hit]]
        if light and nrm is not None:
            nv = nrm[p2p[hit]] @ R.T
            lam = nv[:, 2].abs()  # headlight, two-sided
            shade = shade * (0.35 + 0.65 * lam[:, None])
        img[hit] = shade
        images.append((img.reshape(size, size, 3).clamp(0, 1) * 255).byte().cpu().numpy())
        pix2pts.append(p2p.reshape(size, size).cpu().numpy())
        visible.append(vis.cpu().numpy())
        rots.append(R.cpu().numpy())
    return dict(images=np.stack(images), pix2pt=np.stack(pix2pts), visible=np.stack(visible),
                rotations=np.stack(rots), scale=scale, center=c.cpu().numpy(), splat=splat)
