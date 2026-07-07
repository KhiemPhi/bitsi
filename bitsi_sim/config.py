"""Configuration for the bitsi headless grasp-validation sim.

Gripper defaults match the Robotiq 2F-85 that ships on the Kinova Gen3, so that
sim grasp-success is meaningful for the real arm. Values marked ``VERIFY`` should
be confirmed against the official 2F-85 datasheet / the ros_kortex meshes
(datasets/P5/ros_kortex) before the hardware experiments.

The bitsi heuristic (bitsi_slicer) was tuned with gripper_width=0.13,
gripper_height=0.07, gripper_length=0.11 -- those are ~50% wider than the real
2F-85 (85 mm stroke) and are kept here only as ``legacy_*`` for migration/audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GripperSpec:
    """Parallel-jaw gripper geometry (Robotiq 2F-85)."""

    # --- Robotiq 2F-85 (real Kinova gripper) ---
    max_width: float = 0.085          # jaw stroke, fully open (m)            [datasheet: 85 mm]
    min_width: float = 0.0            # fully closed (m)
    finger_pad_length: float = 0.0375  # contact-pad length along approach (m)  VERIFY
    finger_pad_width: float = 0.022    # contact-pad width, perp to jaw+approach (m) VERIFY
    finger_thickness: float = 0.012    # finger stock thickness (m)             VERIFY
    finger_length: float = 0.045       # finger reach below the coupler (m)     VERIFY
    max_force: float = 100.0           # jaw grip/effort force (N); 2F-85 range 20-235
    base_half_extents: tuple = (0.03, 0.045, 0.02)  # coupler/body hull (m)     VERIFY
    lateral_friction: float = 1.0      # pad friction coefficient

    # --- bitsi heuristic legacy (audit only; do NOT use for real grasps) ---
    legacy_width: float = 0.13
    legacy_height: float = 0.07
    legacy_length: float = 0.11


@dataclass
class SimConfig:
    """Physics + grasp-test settings. Headless (p.DIRECT) by default."""

    gui: bool = False                  # server default: headless. True -> p.GUI (needs X)
    use_egl: bool = False              # GPU offscreen rendering via eglRenderer plugin
    timestep: float = 1.0 / 240.0
    gravity: float = -9.81
    solver_iters: int = 150

    # table
    table_height: float = 0.40         # top surface z (m)
    table_half_extents: tuple = (0.35, 0.35, 0.02)

    # grasp-success (lift/shake) test
    lift_height: float = 0.12          # how far to raise the gripper (m)
    lift_success_dz: float = 0.05      # object must rise >= this to count as held (m)
    open_steps: int = 60
    close_steps: int = 120
    settle_steps: int = 60
    lift_steps: int = 240

    # target-part grasp: a contact counts as "on target" within this radius (m)
    target_contact_radius: float = 0.012

    data_root: str = "datasets"        # matches download_datasets.py (BITSI_DATA_ROOT)
    cache_dir: str = "bitsi_sim/_cache"  # VHACD collision-mesh cache
