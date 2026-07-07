"""Small geometry helpers: grasp representation and frame math.

Pure numpy so it imports without pybullet (used by both the sampler and the sim).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grasp:
    """A parallel-jaw grasp in the world frame.

    center:   midpoint between the two finger contacts (3,)
    approach: unit vector the gripper travels along INTO the object (3,)
              (for a top-down grasp this is [0, 0, -1])
    jaw_axis: unit vector along which the jaws open/close (3,), perpendicular
              to ``approach``; the two contacts lie at center +/- (width/2)*jaw_axis
    width:    commanded opening between the pads (m)
    score:    optional quality (e.g. antipodal alignment / force-closure margin)
    """

    center: np.ndarray
    approach: np.ndarray
    jaw_axis: np.ndarray
    width: float
    score: float = 0.0

    def orthonormal_frame(self) -> np.ndarray:
        """Return a 3x3 rotation whose columns are (x, jaw_axis, approach-as-local-Z).

        Convention used by the sim gripper (see gripper.py):
          local +Y = jaw open/close axis
          local -Z = approach direction (fingers point along local -Z)
        """
        z_local = -_unit(self.approach)          # fingers point along local -Z == approach
        y_local = _unit(self.jaw_axis)
        # re-orthogonalise jaw axis against approach
        y_local = _unit(y_local - np.dot(y_local, z_local) * z_local)
        x_local = np.cross(y_local, z_local)
        R = np.column_stack([x_local, y_local, z_local])
        return R


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > eps else v


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [x, y, z, w] (PyBullet order)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (min, max) of an (N,3) point set."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)
