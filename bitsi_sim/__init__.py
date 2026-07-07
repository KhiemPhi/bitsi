"""bitsi_sim — headless PyBullet grasp-validation environment for the P5
learned-cutting-policy paper.

Validates *target-part* grasps produced by a point-cloud decomposition, using a
floating Robotiq-2F-85-dimensioned parallel-jaw gripper. Import surface:

    from bitsi_sim import (
        BulletWorld, SimConfig, GripperSpec, FloatingGripper, Grasp,
        sample_antipodal_grasps, evaluate_grasp, evaluate_target_part_grasp,
    )
"""
from .config import GripperSpec, SimConfig
from .geometry import Grasp
from .grasp_eval import evaluate_grasp, evaluate_target_part_grasp
from .grasp_sampler import grasp_success_rate, sample_antipodal_grasps
from .gripper import FloatingGripper
from .recorder import FrameRecorder
from .world import BulletWorld

__all__ = [
    "BulletWorld", "SimConfig", "GripperSpec", "FloatingGripper", "Grasp",
    "sample_antipodal_grasps", "grasp_success_rate",
    "evaluate_grasp", "evaluate_target_part_grasp", "FrameRecorder",
]
