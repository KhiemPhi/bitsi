"""Headless smoke test for bitsi_sim.

Runs entirely with p.DIRECT (no display). Drops a box on the table, then:
  1. plain grasp: top-down grasp at the box centre -> should lift;
  2. target-part grasp: treat the box's upper band as the "target part" and the
     lower band as non-target obstacle -> checks contact classification.

Usage (on the server, inside the sim env):
    python -m bitsi_sim.demo_grasp            # headless
    python -m bitsi_sim.demo_grasp --egl      # headless + GPU offscreen render
"""
from __future__ import annotations

import argparse

import numpy as np

from .config import GripperSpec, SimConfig
from .geometry import Grasp
from .gripper import FloatingGripper
from .grasp_eval import evaluate_grasp, evaluate_target_part_grasp
from .world import BulletWorld


def _box_surface_points(center, he, n=2000, seed=0):
    """Sample points on the surface of an axis-aligned box (for a synthetic part)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, dtype=np.float64)
    he = np.asarray(he, dtype=np.float64)
    pts = (rng.uniform(-1, 1, size=(n, 3)) * he) + c
    # snap each point to the nearest face -> lies on the surface
    axis = rng.integers(0, 3, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    pts[np.arange(n), axis] = c[axis] + sign * he[axis]
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gui", action="store_true", help="show GUI (needs X); default headless")
    ap.add_argument("--egl", action="store_true", help="load eglRenderer (GPU offscreen)")
    args = ap.parse_args()

    cfg = SimConfig(gui=args.gui, use_egl=args.egl)
    spec = GripperSpec()

    with BulletWorld(cfg) as world:
        he = (0.02, 0.02, 0.03)
        obj = world.add_box(half_extents=he, mass=0.3, lateral_friction=1.5)
        world.step(120)  # let it settle
        pos = np.array([0.0, 0.0, world.object_z(obj)])

        # --- 1. plain top-down grasp across the short (x) axis ---
        grasp = Grasp(center=pos.copy(),
                      approach=np.array([0.0, 0.0, -1.0]),
                      jaw_axis=np.array([1.0, 0.0, 0.0]),
                      width=2 * he[0] + 0.015)
        gr = FloatingGripper(world, spec)
        res = evaluate_grasp(world, gr, obj, grasp, cfg)
        print(f"[plain]  success={res['success']}  dz={res['dz']:.3f}  "
              f"contacts={res['n_contacts']}  final_width={res['final_width']:.3f}")
        gr.remove()

        # --- 2. target-part grasp: object = target part, with a neighbour part
        #        8 cm away in +Y as the non-target obstacle (spatially separated,
        #        so a clean side grasp should NOT collide with it). ---
        world.reset()
        obj = world.add_box(half_extents=he, mass=0.3, lateral_friction=1.5)
        world.step(120)
        pos = np.array([0.0, 0.0, world.object_z(obj)])
        target = _box_surface_points(pos, he, n=3000)          # the part we grasp
        nontarget = target + np.array([0.0, 0.08, 0.0])         # neighbour part, +8 cm in Y
        grasp2 = Grasp(center=pos.copy(),
                       approach=np.array([0.0, 0.0, -1.0]),
                       jaw_axis=np.array([1.0, 0.0, 0.0]),       # jaws across X, away from neighbour
                       width=2 * he[0] + 0.015)
        gr = FloatingGripper(world, spec)
        res2 = evaluate_target_part_grasp(world, gr, obj, grasp2, target, nontarget, cfg)
        print(f"[target] success={res2['success']}  dz={res2['dz']:.3f}  "
              f"on_target_frac={res2['on_target_frac']:.2f}  "
              f"collided_nontarget={res2['collided_nontarget']}")
        gr.remove()

    print("OK: headless sim ran end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
