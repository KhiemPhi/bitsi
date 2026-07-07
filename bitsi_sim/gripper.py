"""Floating parallel-jaw gripper for grasp-success validation.

We model the Robotiq 2F-85 as a free-floating base + two prismatic fingers
(generated URDF, sized from GripperSpec). This is the standard "floating gripper"
evaluation used by grasp benchmarks (ACRONYM / Contact-GraspNet): it isolates
grasp quality from arm reachability/IK. To later test reachability on the real
Kinova, load the full Gen3+2F-85 URDF from datasets/P5/ros_kortex instead
(see load_kinova_gripper stub at the bottom).

Local frame convention (matches geometry.Grasp.orthonormal_frame):
    +Y = jaw open/close axis      -Z = approach (fingers point along -Z)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None

from .config import GripperSpec
from .geometry import Grasp, rotation_to_quaternion


_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="parallel_jaw">
  <link name="base">
    <inertial><mass value="0.2"/>
      <inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="{bx2} {by2} {bz2}"/></geometry></collision>
    <visual><geometry><box size="{bx2} {by2} {bz2}"/></geometry>
      <material name="grey"><color rgba="0.3 0.3 0.3 1"/></material></visual>
  </link>
  {finger_left}
  {finger_right}
  <joint name="finger_left_joint" type="prismatic">
    <parent link="base"/><child link="finger_left"/>
    <origin xyz="0 0 {finger_z}"/><axis xyz="0 1 0"/>
    <limit lower="{jmin}" upper="{jmax}" effort="{effort}" velocity="1.0"/>
  </joint>
  <joint name="finger_right_joint" type="prismatic">
    <parent link="base"/><child link="finger_right"/>
    <origin xyz="0 0 {finger_z}"/><axis xyz="0 -1 0"/>
    <limit lower="{jmin}" upper="{jmax}" effort="{effort}" velocity="1.0"/>
  </joint>
</robot>
"""

_FINGER_TEMPLATE = """  <link name="finger_{side}">
    <inertial><mass value="0.02"/>
      <inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="{fx2} {fy2} {fz2}"/></geometry></collision>
    <visual><geometry><box size="{fx2} {fy2} {fz2}"/></geometry>
      <material name="dark"><color rgba="0.1 0.1 0.1 1"/></material></visual>
  </link>"""


class FloatingGripper:
    """Kinematically-servoed floating parallel-jaw gripper."""

    def __init__(self, world, spec: GripperSpec | None = None):
        if p is None:
            raise ImportError("pybullet is required for FloatingGripper")
        self.world = world
        self.spec = spec or GripperSpec()
        self._fy = self.spec.finger_thickness / 2.0      # finger half-thickness (jaw axis)
        self._jmin = self._fy                            # closed: pads meet at centre
        self._jmax = self.spec.max_width / 2.0 + self._fy
        self.urdf_path = self._write_urdf()
        self.body = p.loadURDF(self.urdf_path, useFixedBase=False)
        self.jl, self.jr = 0, 1                          # joint indices (order as declared)
        for j in (self.jl, self.jr):
            p.changeDynamics(self.body, j, lateralFriction=self.spec.lateral_friction)
        # pin the base to a controllable world target (floating-gripper trick)
        self.constraint = p.createConstraint(
            self.body, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1.0])
        self._target_pos = np.array([0, 0, 1.0], dtype=np.float64)
        self._target_orn = np.array([0, 0, 0, 1.0], dtype=np.float64)

    # -- construction -----------------------------------------------------
    def _write_urdf(self) -> str:
        s = self.spec
        bx, by, bz = s.base_half_extents
        fx, fy, fz = s.finger_pad_width / 2, s.finger_thickness / 2, s.finger_length / 2
        finger_z = -(bz + fz)  # fingers hang below the base along -Z (approach)
        left = _FINGER_TEMPLATE.format(side="left", fx2=2 * fx, fy2=2 * fy, fz2=2 * fz)
        right = _FINGER_TEMPLATE.format(side="right", fx2=2 * fx, fy2=2 * fy, fz2=2 * fz)
        urdf = _URDF_TEMPLATE.format(
            bx2=2 * bx, by2=2 * by, bz2=2 * bz,
            finger_left=left, finger_right=right, finger_z=finger_z,
            jmin=self._jmin, jmax=self._jmax, effort=s.max_force)
        cache = Path(self.world.cfg.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        path = str(cache / "parallel_jaw.urdf")
        with open(path, "w") as f:
            f.write(urdf)
        return path

    # -- pose control -----------------------------------------------------
    def set_pose(self, position, orientation, max_force: float = 2000.0,
                 teleport: bool = False) -> None:
        """Command the pinned base to a world pose.

        teleport=True snaps the base there (resetBasePositionAndOrientation) --
        used for the initial placement so we don't wait for the soft constraint
        to traverse from the spawn pose. For dynamic motion (the lift) leave it
        False so the object's weight is felt through the constraint.
        """
        self._target_pos = np.asarray(position, dtype=np.float64)
        self._target_orn = np.asarray(orientation, dtype=np.float64)
        if teleport:
            p.resetBasePositionAndOrientation(self.body, list(self._target_pos),
                                              list(self._target_orn))
        p.changeConstraint(self.constraint, list(self._target_pos),
                           list(self._target_orn), maxForce=max_force)

    def place_at(self, grasp: Grasp) -> None:
        """Snap the gripper so its (open) pads straddle ``grasp.center``."""
        R = grasp.orthonormal_frame()
        quat = rotation_to_quaternion(R)
        bz = self.spec.base_half_extents[2]
        fz = self.spec.finger_length / 2.0
        approach = grasp.approach / (np.linalg.norm(grasp.approach) + 1e-9)
        # base sits back along the approach so the finger centres land on grasp.center
        base_pos = np.asarray(grasp.center, dtype=np.float64) - (bz + fz) * approach
        self.set_pose(base_pos, quat, teleport=True)
        self.reset_fingers_open()

    def reset_fingers_open(self) -> None:
        """Snap both fingers fully open (no servo-convergence wait)."""
        for j in (self.jl, self.jr):
            p.resetJointState(self.body, j, self._jmax)
        self.open()

    # -- jaw control ------------------------------------------------------
    def _servo(self, jv: float, force: float | None = None) -> None:
        jv = float(np.clip(jv, self._jmin, self._jmax))
        f = self.spec.max_force if force is None else force
        for j in (self.jl, self.jr):
            p.setJointMotorControl2(self.body, j, p.POSITION_CONTROL,
                                    targetPosition=jv, force=f)

    def open(self) -> None:
        self._servo(self._jmax)

    def close(self) -> None:
        self._servo(self._jmin)

    def set_width(self, width: float) -> None:
        self._servo(width / 2.0 + self._fy)

    def width(self) -> float:
        jv = 0.5 * (p.getJointState(self.body, self.jl)[0]
                    + p.getJointState(self.body, self.jr)[0])
        return max(0.0, 2.0 * (jv - self._fy))

    def lift(self, dz: float, world, steps: int | None = None,
             max_force: float = 2000.0) -> None:
        """Raise the base by ``dz`` over ``steps``, ramping so friction carries the object."""
        steps = steps or world.cfg.lift_steps
        start = self._target_pos.copy()
        for i in range(steps):
            frac = (i + 1) / steps
            self.set_pose(start + np.array([0.0, 0.0, dz * frac]),
                          self._target_orn, max_force=max_force)
            world.step(1)

    # -- sensing ----------------------------------------------------------
    def contacts_with(self, body_id: int):
        """List of contact points with ``body_id``; each has world pos (index 5/6)."""
        return p.getContactPoints(bodyA=self.body, bodyB=body_id)

    def remove(self) -> None:
        p.removeConstraint(self.constraint)
        p.removeBody(self.body)


def load_kinova_gripper(world, kortex_root: str, spec: GripperSpec | None = None):
    """Stub: load the real Gen3 + Robotiq 2F-85 URDF for reachability experiments.

    Expects datasets/P5/ros_kortex (download_datasets.py --paper P5). Left as a
    hook -- floating-gripper eval above is the primary grasp-success metric.
    """
    urdf = os.path.join(kortex_root, "kortex_description", "robots", "gen3_robotiq_2f_85.urdf")
    raise NotImplementedError(
        f"Full Kinova arm loading not yet wired. Point this at {urdf} once "
        "ros_kortex is downloaded and you need reachability (not just grasp success).")
