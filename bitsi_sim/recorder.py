"""Headless frame recorder -> GIF / MP4.

PyBullet's built-in STATE_LOGGING_VIDEO_MP4 needs the GUI (OpenGL window), so it
does not work under p.DIRECT. Instead we grab frames with p.getCameraImage every
`stride` simulation steps (GPU via EGL if loaded, else the CPU TinyRenderer) and
encode them with imageio.

Speed control (playback speed relative to real time):
    speed = fps * stride / (1 / sim_timestep)     # sim runs at 240 steps/s
  e.g. stride=8, fps=30  -> 1.0x ;  stride=16, fps=30 -> 2.0x ;
       stride=8,  fps=60 -> 2.0x ;  stride=32, fps=30 -> 4.0x .
Raise `stride` (skip more sim steps) or `fps` (faster playback) to speed up.
"""
from __future__ import annotations

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None


class FrameRecorder:
    def __init__(self, world, width: int = 640, height: int = 480,
                 target=None, distance: float = 0.9, yaw: float = 50.0,
                 pitch: float = -35.0, fov: float = 55.0, stride: int = 8):
        if p is None:
            raise ImportError("pybullet is required for FrameRecorder")
        self.world = world
        self.w, self.h = int(width), int(height)
        self.distance, self.yaw, self.pitch = distance, yaw, pitch
        self.stride = max(1, int(stride))
        self.target = list(target) if target is not None else [0.0, 0.0, world.cfg.table_height]
        self.proj = p.computeProjectionMatrixFOV(fov, self.w / self.h, 0.01, 3.0)
        # EGL (GPU) if the plugin was loaded on this world, else CPU TinyRenderer
        self.renderer = (p.ER_BULLET_HARDWARE_OPENGL
                         if getattr(world, "_egl_plugin", None) is not None
                         else p.ER_TINY_RENDERER)
        self.frames: list[np.ndarray] = []

    def capture(self) -> None:
        view = p.computeViewMatrixFromYawPitchRoll(
            self.target, self.distance, self.yaw, self.pitch, 0.0, 2)
        img = p.getCameraImage(self.w, self.h, view, self.proj, renderer=self.renderer)
        rgb = np.reshape(img[2], (self.h, self.w, 4))[:, :, :3].astype(np.uint8)
        self.frames.append(rgb)

    def clear(self) -> None:
        self.frames.clear()

    def save(self, path: str, fps: int = 30) -> str:
        if not self.frames:
            raise RuntimeError("no frames captured (attach the recorder before stepping)")
        import imageio.v2 as imageio
        if path.lower().endswith(".gif"):
            imageio.mimsave(path, self.frames, fps=fps, loop=0)
        else:  # mp4 / webm via imageio-ffmpeg
            imageio.mimsave(path, self.frames, fps=fps, quality=8, macro_block_size=None)
        return path

    @property
    def speed(self) -> float:
        """Playback speed vs real time at a given fps is set at save(); this is the
        capture ratio -- multiply by (fps/30) mentally, or just read the demo print."""
        return self.stride / (1.0 / self.world.cfg.timestep)
