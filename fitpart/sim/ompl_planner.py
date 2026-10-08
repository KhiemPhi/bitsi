"""OMPL motion planning for PandaScene / KinovaScene (free-space joint motions).

  * RRT-Connect from OMPL (the library MoveIt uses) in the arm's joint space (bounds = joint limits;
    the Gen3's continuous joints at +-2 pi), state validity = the scene's collision check, motion
    validation at `resolution` (fraction of the space extent), path simplification (shortcutting +
    smoothing) and densification for execution.
  * Self-collision with an allowed-collision matrix built like MoveIt's setup assistant: robot body pairs
    in contact in >= 95% of random configurations ("always in collision") or at the home pose are
    allowed; any other robot-robot contact is a collision.
Cartesian segments (approach, lift, task screw) stay sequential-IK chains, as in MoveIt's
computeCartesianPath; the callers now also collision-check them.
OMPL 1.7 Python bindings (pip "ompl<2"); 2.0.1's bindings segfault on state allocation here.
"""
import numpy as np


def allowed_pairs(sc, n=200, frac=0.95, seed=0):
    mj, m, d = sc.mj, sc.model, sc.chk
    rng = np.random.default_rng(seed)
    count = {}
    confs = [sc.home if hasattr(sc, "home") else sc.data.qpos[:7].copy()]
    confs += [rng.uniform(sc.lo, sc.hi) for _ in range(n)]
    home_pairs = set()
    for k, q in enumerate(confs):
        d.qpos[:] = sc.data.qpos
        d.qpos[:7] = q
        mj.mj_forward(m, d)
        seen = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if b1 in sc.robot_bodies and b2 in sc.robot_bodies and b1 != b2:
                seen.add((min(b1, b2), max(b1, b2)))
        if k == 0:
            home_pairs = seen
        else:
            for p in seen:
                count[p] = count.get(p, 0) + 1
    always = {p for p, c in count.items() if c >= frac * n}
    return always | home_pairs


class OMPLPlanner:
    def __init__(self, sc, planner="RRTConnect", time=5.0, resolution=0.005, self_collision=True):
        from ompl import base as ob, geometric as og
        self.ob, self.og, self.sc = ob, og, sc
        self.time, self.planner = time, planner
        if self_collision:
            sc.allowed_self = allowed_pairs(sc)
        self.space = ob.RealVectorStateSpace(7)
        b = ob.RealVectorBounds(7)
        for i in range(7):
            b.setLow(i, float(sc.lo[i]))
            b.setHigh(i, float(sc.hi[i]))
        self.space.setBounds(b)
        self.ss = og.SimpleSetup(self.space)
        self.kw = {}
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._valid))
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(resolution)
        self._keep = []          # keep OMPL states alive (the 1.x bindings double-free otherwise)

    def _valid(self, s):
        q = np.array([s[i] for i in range(7)])
        return not self.sc.in_collision(q, **self.kw)

    def plan(self, qa, qb, seed=0, n_out=None, **kw):
        ob, og = self.ob, self.og
        from ompl import util as ou
        ou.RNG.setSeed(int(seed) + 1)
        self.kw = kw
        self.ss.clear()
        st, gl = ob.State(self.space), ob.State(self.space)
        for i in range(7):
            st[i], gl[i] = float(qa[i]), float(qb[i])
        self._keep += [st, gl]
        self.ss.setStartAndGoalStates(st, gl)
        si = self.ss.getSpaceInformation()
        self.ss.setPlanner(getattr(og, self.planner)(si))
        if not self.ss.solve(self.time) or not self.ss.haveExactSolutionPath():
            return None
        self.ss.simplifySolution(1.0)
        p = self.ss.getSolutionPath()
        n = n_out or max(2, int(np.ceil(np.abs(np.asarray(qb) - np.asarray(qa)).max() / 0.1)) + 1)
        p.interpolate(max(n, p.getStateCount()))
        return [np.array([p.getState(k)[i] for i in range(7)]) for k in range(p.getStateCount())]
