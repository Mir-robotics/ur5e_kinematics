"""
Rigid-body dynamics: mass matrix M(q), Coriolis/centrifugal term C(q,qdot)@qdot,
and gravity vector G(q).

This file was referenced by controller.py / main.py but wasn't part of the
uploaded set, so it's implemented fresh here.

Method: per-link center-of-mass Jacobians (linear + angular), built the same
way as jacobian.geometric_jacobian but truncated to the joints that actually
affect each link and evaluated at the link's COM instead of the end effector.
From those:

    M(q)    = sum_i [ Jv_i^T m_i Jv_i + Jw_i^T (R_i I_i R_i^T) Jw_i ]
    G(q)    = sum_i  Jv_i^T (m_i * g0 * z_hat)          (z_hat = [0,0,1])
    C(q,qd) = Christoffel symbols of M(q), via central finite differences
              of M w.r.t. q; C(q,qd)@qd is what's returned (that's what
              controller.py actually needs).

This is the standard "energy/Jacobian" formulation (Siciliano et al.,
"Robotics: Modelling, Planning and Control", ch. 7) rather than recursive
Newton-Euler — easier to keep correct and to unit test, at the cost of being
a bit slower per call (fine for control-loop-in-the-loop simulation, not
meant for a real-time 1kHz controller without further optimization).

Note: robot.inertia is a placeholder (zeros) per robot.py's TODO — until
that's filled from a real URDF/datasheet, M/C only capture the *point-mass*
contribution of each link, not its rotational inertia about its own COM.
G(q) is unaffected by this (gravity torque only depends on mass + COM
location), so it's already exact for the given mass/COM parameters.
"""
import numpy as np
try:
    from .fk import forward_kinematics
except ImportError:
    from fk import forward_kinematics

GRAVITY = 9.80665


def _com_jacobians(robot, q, link_idx, transforms=None):
    """
    Linear (3xn) and angular (3xn) Jacobians of link `link_idx`'s COM,
    w.r.t. joints 0..link_idx (later joints don't affect this link).
    Pass a precomputed `transforms` (from forward_kinematics) to avoid
    re-running FK once per link when computing all of M(q).
    """
    if transforms is None:
        transforms, _ = forward_kinematics(robot, q)
    T_link = transforms[link_idx]
    R_link = T_link[:3, :3]
    p_com = T_link[:3, 3] + R_link @ robot.com[link_idx]

    n = robot.n
    Jv = np.zeros((3, n))
    Jw = np.zeros((3, n))
    for i in range(link_idx + 1):
        if i == 0:
            z_im1 = np.array([0.0, 0.0, 1.0])
            o_im1 = np.zeros(3)
        else:
            z_im1 = transforms[i - 1][:3, 2]
            o_im1 = transforms[i - 1][:3, 3]
        Jv[:, i] = np.cross(z_im1, p_com - o_im1)
        Jw[:, i] = z_im1
    return Jv, Jw, R_link


def compute_M(robot, q):
    """Joint-space mass/inertia matrix M(q), shape (n, n).

    Includes reflected motor/gearbox inertia (robot.armature) added directly
    to the diagonal, matching how MuJoCo (and real actuated joints) do it:
    M_total = M_rigid_body + diag(armature). Without this term the wrist
    joints in particular are severely underestimated, since their link-only
    inertia is tiny (see tests/test_mujoco_crossvalidation.py)."""
    n = robot.n
    transforms, _ = forward_kinematics(robot, q)  # computed once, reused per link
    M = np.zeros((n, n))
    for i in range(n):
        Jv, Jw, R_link = _com_jacobians(robot, q, i, transforms=transforms)
        I_world = R_link @ robot.inertia_matrix(i) @ R_link.T
        M += robot.link_mass[i] * (Jv.T @ Jv) + Jw.T @ I_world @ Jw
    if hasattr(robot, "armature"):
        M += np.diag(robot.armature)
    return M


def compute_G(robot, q):
    """Gravity torque vector G(q), shape (n,)."""
    n = robot.n
    transforms, _ = forward_kinematics(robot, q)
    G = np.zeros(n)
    z_hat = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        Jv, _, _ = _com_jacobians(robot, q, i, transforms=transforms)
        G += robot.link_mass[i] * GRAVITY * (Jv.T @ z_hat)
    return G


def compute_C_qdot(robot, q, qdot, eps=1e-6):
    """
    Returns C(q, qdot) @ qdot (shape (n,)), via the Christoffel-symbol
    definition, using central finite differences of M(q) for dM/dq_i.
    """
    n = robot.n
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)

    dM = np.zeros((n, n, n))  # dM[:,:,k] = dM/dq_k
    for k in range(n):
        dq = np.zeros(n)
        dq[k] = eps
        M_plus = compute_M(robot, q + dq)
        M_minus = compute_M(robot, q - dq)
        dM[:, :, k] = (M_plus - M_minus) / (2 * eps)

    C_qdot = np.zeros(n)
    for k in range(n):
        christoffel = 0.5 * (dM[k, :, :] + dM[k, :, :].T - dM[:, :, k])
        C_qdot[k] = qdot @ christoffel @ qdot
    return C_qdot


def compute_dynamics(robot, q, qdot):
    """Convenience: returns (M, C_qdot, G) in one call, reusing dM work."""
    return compute_M(robot, q), compute_C_qdot(robot, q, qdot), compute_G(robot, q)


def forward_dynamics(robot, q, qdot, tau):
    """
    qddot = M(q)^-1 [ tau - C(q,qdot)@qdot - G(q) ]

    Needed to actually simulate the robot (integrate q,qdot forward given
    applied joint torques) rather than only doing inverse dynamics.
    """
    M = compute_M(robot, q)
    C_qdot = compute_C_qdot(robot, q, qdot)
    G = compute_G(robot, q)
    rhs = np.asarray(tau, dtype=float) - C_qdot - G
    return np.linalg.solve(M, rhs)


def simulate(robot, q0, qdot0, tau_fn, dt, n_steps):
    """
    RK4-integrate the robot's own forward_dynamics under a torque policy
    tau_fn(t, q, qdot) -> tau (n,).

    Returns (q_hist, qdot_hist, t_hist), each length n_steps+1 (includes the
    initial state at t=0).
    """
    n = robot.n
    q = np.asarray(q0, dtype=float).copy()
    qdot = np.asarray(qdot0, dtype=float).copy()
    t = 0.0

    q_hist = [q.copy()]
    qdot_hist = [qdot.copy()]
    t_hist = [t]

    for _ in range(n_steps):
        def deriv(x, t_local):
            qq, qqd = x[:n], x[n:]
            tau = tau_fn(t_local, qq, qqd)
            qqdd = forward_dynamics(robot, qq, qqd, tau)
            return np.hstack([qqd, qqdd])

        x = np.hstack([q, qdot])
        k1 = deriv(x, t)
        k2 = deriv(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = deriv(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = deriv(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        q, qdot = x[:n], x[n:]
        t += dt
        q_hist.append(q.copy())
        qdot_hist.append(qdot.copy())
        t_hist.append(t)

    return np.array(q_hist), np.array(qdot_hist), np.array(t_hist)


if __name__ == "__main__":
    from robot import UR5e
    r = UR5e()
    q = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])
    qdot = np.array([0.1, -0.2, 0.05, 0.0, 0.1, -0.1])

    M = compute_M(r, q)
    print("M(q) eigenvalues (should all be > 0):", np.round(np.linalg.eigvalsh(M), 4))
    print("M symmetric?", np.allclose(M, M.T))

    G = compute_G(r, q)
    print("G(q):", np.round(G, 4))

    C_qdot = compute_C_qdot(r, q, qdot)
    print("C(q,qdot)@qdot:", np.round(C_qdot, 4))

    # sanity check: skew-symmetry of (Mdot - 2C) is the standard passivity
    # test; check it via finite-difference Mdot along the qdot direction.
    dt = 1e-6
    M0 = compute_M(r, q)
    M1 = compute_M(r, q + qdot * dt)
    Mdot = (M1 - M0) / dt
    # Build full C via Christoffel symbols to test skew-symmetry of Mdot-2C
    n = r.n
    dM = np.zeros((n, n, n))
    for k in range(n):
        dq = np.zeros(n)
        dq[k] = 1e-6
        dM[:, :, k] = (compute_M(r, q + dq) - compute_M(r, q - dq)) / (2e-6)
    C = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            C[k, j] = 0.5 * sum(
                (dM[k, j, i] + dM[k, i, j] - dM[i, j, k]) * qdot[i] for i in range(n)
            )
    skew = Mdot - 2 * C
    print("max|skew + skew^T| (should be ~0):", np.max(np.abs(skew + skew.T)))
