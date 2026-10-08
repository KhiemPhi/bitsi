"""Task-dependent grasp metric for a screw motion: SOCP grasp-force optimization, a direct port of
ITSI's socp_module/two_finger_gfop_general_screw_motion.py (Fakhari et al. formulation).

Soft-finger contacts (elliptic approximation) at two contacts; equilibrium
    G f_C + F_ext = tau * Ad_{g_OE} w_E,     w_E = (0, 0, 0, screw_axis)   (pure moment about the axis)
with friction cones, a torsion limit and a normal-force limit F; maximize tau, the largest task
moment about the screw axis the grasp can exert while balancing the object's weight.

Parameter values (mu, e_r, sigma, F) are not set in the ITSI repo; the defaults below are
assumptions to be confirmed.
"""
import cvxpy as cp
import numpy as np

DEFAULTS = dict(mu=0.5, e1=1.0, e2=1.0, er=0.02, sigma=0.02, F=70.0)


def _hat(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def _grasp_map_sf(R, p):
    """6x6 soft-finger grasp map (ITSI get_grasp_map, contact_model='sf')."""
    B = np.zeros((6, 6))
    B[0:3, 0:3] = np.eye(3)
    B[3:6, 5] = [0, 0, 1]
    Ad = np.zeros((6, 6))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = _hat(p) @ R
    Ad[3:6, 3:6] = R
    return Ad @ B


def _contact_frame(n):
    """Rotation whose z column is the contact normal (direction of the finger force)."""
    n = n / np.linalg.norm(n)
    t = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    t /= np.linalg.norm(t)
    return np.stack([t, np.cross(n, t), n], 1)


def screw_tau(c1, c2, com, mass, screw_axis, screw_point, translation=False, **params):
    """tau for a parallel-jaw grasp (contacts c1, c2, world frame) and a screw (axis, point).
    translation=True (pitch = inf): the task wrench is a force along the axis instead of a moment
    about it, so tau is the largest such force the grasp can exert while balancing the weight."""
    P = dict(DEFAULTS, **params)
    x = (c2 - c1) / np.linalg.norm(c2 - c1)
    G = np.hstack([_grasp_map_sf(_contact_frame(x), c1 - com),     # finger at c1 pushes along +x
                   _grasp_map_sf(_contact_frame(-x), c2 - com)])   # finger at c2 pushes along -x
    F_ext = np.array([0, 0, -9.81 * mass, 0, 0, 0])
    AdE = np.zeros((6, 6))
    AdE[0:3, 0:3] = np.eye(3)
    AdE[3:6, 0:3] = _hat(np.asarray(screw_point) - com)
    AdE[3:6, 3:6] = np.eye(3)
    a = np.asarray(screw_axis, float) / np.linalg.norm(screw_axis)
    wE = np.r_[a, np.zeros(3)] if translation else np.r_[np.zeros(3), a]
    f = cp.Variable(12)
    tau = cp.Variable()
    mu, e1, e2, er, sg, F = P["mu"], P["e1"], P["e2"], P["er"], P["sigma"], P["F"]
    cons = [G @ f + F_ext == tau * (AdE @ wE),
            cp.SOC(mu * f[2], cp.hstack([f[0] / e1, f[1] / e2, f[5] / er])),
            cp.SOC(mu * f[8], cp.hstack([f[6] / e1, f[7] / e2, f[11] / er])),
            cp.SOC(sg * f[2], f[5:6]), cp.SOC(sg * f[8], f[11:12]),
            f[2] >= 0, f[8] >= 0, f[3] == 0, f[9] == 0, f[4] == 0, f[10] == 0, f[2] <= F, f[8] <= F]
    prob = cp.Problem(cp.Maximize(tau), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return float("nan")
    return float(tau.value) if prob.status in ("optimal", "optimal_inaccurate") else float("nan")
