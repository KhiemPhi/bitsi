"""Headless PyBullet world: table-top scene for grasp validation.

Connects with p.DIRECT (no X server) by default. Optionally loads the
eglRenderer plugin for GPU-accelerated offscreen rendering (depth/segmentation
images) on a server with a GPU (e.g. the A100 on this box).
"""
from __future__ import annotations

import pkgutil

try:
    import pybullet as p
    import pybullet_data
except ImportError as e:  # allow tooling/import without the dep
    p = None
    pybullet_data = None
    _IMPORT_ERR = e

from .config import SimConfig


class BulletWorld:
    """A minimal, headless table-top physics world."""

    def __init__(self, cfg: SimConfig | None = None):
        if p is None:
            raise ImportError(
                f"pybullet not available ({_IMPORT_ERR}). "
                "pip install pybullet  (see bitsi_sim/README.md for headless setup)."
            )
        self.cfg = cfg or SimConfig()
        self._egl_plugin = None
        self.cid = self._connect()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, self.cfg.gravity)
        p.setTimeStep(self.cfg.timestep)
        p.setPhysicsEngineParameter(numSolverIterations=self.cfg.solver_iters)
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = self._make_table()
        self.recorder = None          # optional FrameRecorder (see recorder.py)
        self._step_count = 0

    # -- setup ------------------------------------------------------------
    def _connect(self) -> int:
        mode = p.GUI if self.cfg.gui else p.DIRECT
        cid = p.connect(mode)
        if self.cfg.use_egl and not self.cfg.gui:
            egl = pkgutil.get_loader("eglRenderer")
            if egl is not None:
                self._egl_plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self._egl_plugin = p.loadPlugin("eglRendererPlugin")
        return cid

    def _make_table(self) -> int:
        he = list(self.cfg.table_half_extents)
        top = self.cfg.table_height
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.7, 0.7, 0.7, 1])
        # static (mass 0); top surface sits at z = table_height
        body = p.createMultiBody(0, col, vis, basePosition=[0, 0, top - he[2]])
        p.changeDynamics(body, -1, lateralFriction=1.0)
        return body

    # -- helpers ----------------------------------------------------------
    def add_box(self, half_extents=(0.02, 0.02, 0.03), position=None,
                mass: float = 0.1, rgba=(0.2, 0.5, 0.9, 1.0),
                lateral_friction: float = 1.0) -> int:
        """Add a box object resting on the table (default: centred on the table)."""
        he = list(half_extents)
        if position is None:
            position = [0.0, 0.0, self.cfg.table_height + he[2] + 1e-3]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=list(rgba))
        body = p.createMultiBody(mass, col, vis, basePosition=list(position))
        p.changeDynamics(body, -1, lateralFriction=lateral_friction)
        return body

    def object_z(self, body_id: int) -> float:
        return p.getBasePositionAndOrientation(body_id)[0][2]

    def attach_recorder(self, recorder) -> None:
        """Register a FrameRecorder; step() then grabs a frame every recorder.stride steps."""
        self.recorder = recorder

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            p.stepSimulation()
            self._step_count += 1
            if self.recorder is not None and self._step_count % self.recorder.stride == 0:
                self.recorder.capture()

    def reset(self) -> None:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, self.cfg.gravity)
        p.setTimeStep(self.cfg.timestep)
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = self._make_table()

    def disconnect(self) -> None:
        if self.cid is not None:
            p.disconnect(self.cid)
            self.cid = None

    def __enter__(self) -> "BulletWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
