"""Mesh helpers: convex decomposition (VHACD) + caching for PyBullet collision.

ShapeNetCore / GSO meshes are non-convex; PyBullet needs a convex-decomposed
collision mesh for stable grasping. VHACD is run once per mesh and cached.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

try:
    import pybullet as p
except ImportError:  # allow import without pybullet (syntax/tooling)
    p = None


def _hash_path(path: str) -> str:
    return hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()[:12]


def ensure_collision_mesh(obj_path: str, cache_dir: str, resolution: int = 200_000,
                          force: bool = False) -> str:
    """Return a VHACD convex-decomposed .obj for ``obj_path``, running VHACD if needed.

    Cached under ``cache_dir`` keyed by the source path. Requires pybullet.
    """
    if p is None:
        raise ImportError("pybullet is required for VHACD (pip install pybullet)")
    obj_path = str(obj_path)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(cache_dir, f"vhacd_{_hash_path(obj_path)}.obj")
    log_path = os.path.join(cache_dir, f"vhacd_{_hash_path(obj_path)}.log")
    if force or not os.path.exists(out_path):
        # p.vhacd writes an .obj of convex parts; params tuned for tabletop objects.
        p.vhacd(obj_path, out_path, log_path, resolution=resolution)
    return out_path


def load_mesh_object(world, obj_path: str, scale: float = 1.0,
                     position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0),
                     mass: float = 0.2, use_vhacd: bool = True,
                     lateral_friction: float = 1.0) -> int:
    """Load a mesh as a rigid body (VHACD collision + raw visual). Returns body id."""
    if p is None:
        raise ImportError("pybullet is required to load meshes")
    col_path = (ensure_collision_mesh(obj_path, world.cfg.cache_dir)
                if use_vhacd else obj_path)
    col = p.createCollisionShape(p.GEOM_MESH, fileName=col_path,
                                 meshScale=[scale] * 3,
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH if not use_vhacd else 0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=[scale] * 3)
    body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=list(position),
                             baseOrientation=list(orientation))
    p.changeDynamics(body, -1, lateralFriction=lateral_friction)
    return body
