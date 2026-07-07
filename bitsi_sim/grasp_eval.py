"""Grasp-success validation under physics (headless).

Two entry points:
  evaluate_grasp            -> plain lift/shake test (did the object stay held?)
  evaluate_target_part_grasp-> the P5 metric: the grasp must hold the object AND
                               its contacts must land on the *target part* without
                               colliding with the rest of the object.

The target-part variant is what couples decomposition quality to grasp success:
a decomposition that merges the target into a neighbour makes a clean, collision-
free grasp of "the target part" impossible.
"""
from __future__ import annotations

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None

from .config import SimConfig
from .geometry import Grasp


def _contact_world_points(contacts) -> np.ndarray:
    """World-frame contact positions on the object (positionOnB = index 6)."""
    if not contacts:
        return np.zeros((0, 3))
    return np.array([c[6] for c in contacts], dtype=np.float64)


def evaluate_grasp(world, gripper, object_id: int, grasp: Grasp,
                   cfg: SimConfig | None = None) -> dict:
    """Approach -> close -> lift. Returns success + diagnostics."""
    if p is None:
        raise ImportError("pybullet is required for grasp evaluation")
    cfg = cfg or world.cfg

    z0 = world.object_z(object_id)
    gripper.place_at(grasp)          # opens + positions
    world.step(cfg.open_steps)
    gripper.close()
    world.step(cfg.close_steps)
    world.step(cfg.settle_steps)     # let the grip stabilise before measuring/lifting

    contacts = gripper.contacts_with(object_id)
    contact_pts = _contact_world_points(contacts)
    n_contacts = len(contact_pts)

    gripper.lift(cfg.lift_height, world)
    z1 = world.object_z(object_id)
    dz = z1 - z0
    lifted = dz >= cfg.lift_success_dz

    return {
        "success": bool(lifted and n_contacts >= 2),
        "lifted": bool(lifted),
        "dz": float(dz),
        "n_contacts": int(n_contacts),
        "contact_points": contact_pts,
        "final_width": float(gripper.width()),
    }


def evaluate_target_part_grasp(world, gripper, object_id: int, grasp: Grasp,
                               target_points: np.ndarray,
                               nontarget_points: np.ndarray | None = None,
                               cfg: SimConfig | None = None) -> dict:
    """P5 target-part grasp metric.

    Args:
        target_points:    (M,3) points of the part we intend to grasp (world frame).
        nontarget_points: (K,3) points of the rest of the object (obstacles).
    Success requires: object lifted, >=50% of contacts on the target part, and
    (if nontarget given) no contact that is nearer a nontarget point.
    """
    from scipy.spatial import cKDTree

    cfg = cfg or world.cfg
    res = evaluate_grasp(world, gripper, object_id, grasp, cfg)
    contact_pts = res["contact_points"]

    on_target_frac = 0.0
    collided_nontarget = False
    if len(contact_pts) > 0:
        tgt_tree = cKDTree(np.asarray(target_points).reshape(-1, 3))
        d_tgt, _ = tgt_tree.query(contact_pts)
        if nontarget_points is not None and len(nontarget_points) > 0:
            nt_tree = cKDTree(np.asarray(nontarget_points).reshape(-1, 3))
            d_nt, _ = nt_tree.query(contact_pts)
            # assign each contact to the nearest part (density/scale-robust)
            on_target = d_tgt <= d_nt
            # collision = a contact on the nontarget side AND physically touching it
            collided_nontarget = bool(np.any((d_nt < d_tgt)
                                             & (d_nt <= cfg.target_contact_radius)))
        else:
            on_target = d_tgt <= cfg.target_contact_radius
        on_target_frac = float(on_target.mean())

    success = bool(res["lifted"] and on_target_frac >= 0.5 and not collided_nontarget)
    res.update({
        "success": success,
        "on_target_frac": on_target_frac,
        "collided_nontarget": collided_nontarget,
    })
    return res
