"""
Deeper dynamics tests: forward-dynamics self-consistency and a real
closed-loop simulation (not just single-point M/C/G evaluation).
"""
import numpy as np
import pytest

from ur5e_kinematics import UR5e, compute_M, compute_C_qdot, compute_G
from ur5e_kinematics.dynamics import forward_dynamics, simulate
from ur5e_kinematics.controller import computed_torque_control, pd_control
from ur5e_kinematics.trajectory import quintic_trajectory


@pytest.fixture
def robot():
    return UR5e()


def test_forward_dynamics_self_consistent(robot):
    """M(q) @ forward_dynamics(...) should reproduce (tau - C@qdot - G)."""
    rng = np.random.default_rng(3)
    q = rng.uniform(-np.pi, np.pi, 6)
    qdot = rng.uniform(-1, 1, 6)
    tau = rng.uniform(-50, 50, 6)

    qddot = forward_dynamics(robot, q, qdot, tau)
    M = compute_M(robot, q)
    C_qdot = compute_C_qdot(robot, q, qdot)
    G = compute_G(robot, q)

    lhs = M @ qddot
    rhs = tau - C_qdot - G
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_free_fall_conserves_energy_short_horizon(robot):
    """No control torque, no friction in the model: total mechanical energy
    (an M-weighted kinetic term + a Jv-based potential term) should be
    conserved to integrator error over a short horizon."""
    q0 = np.array([0.1, -0.8, 0.9, 0.0, 0.3, 0.0])
    qdot0 = np.zeros(6)

    def zero_tau(t, q, qdot):
        return np.zeros(6)

    q_hist, qdot_hist, t_hist = simulate(robot, q0, qdot0, zero_tau, dt=0.005, n_steps=40)

    def kinetic(q, qdot):
        M = compute_M(robot, q)
        return 0.5 * qdot @ M @ qdot

    def potential(q):
        # numerically integrate G(q) isn't needed; use the same Jv-based
        # potential energy compute_G is the gradient of, evaluated directly.
        from ur5e_kinematics.dynamics import _com_jacobians, GRAVITY
        from ur5e_kinematics.fk import forward_kinematics
        transforms, _ = forward_kinematics(robot, q)
        U = 0.0
        for i in range(robot.n):
            T_link = transforms[i]
            p_com = T_link[:3, 3] + T_link[:3, :3] @ robot.com[i]
            U += robot.link_mass[i] * GRAVITY * p_com[2]
        return U

    E0 = kinetic(q_hist[0], qdot_hist[0]) + potential(q_hist[0])
    E_end = kinetic(q_hist[-1], qdot_hist[-1]) + potential(q_hist[-1])
    assert abs(E_end - E0) / max(abs(E0), 1.0) < 1e-2


def test_closed_loop_computed_torque_tracks_trajectory(robot):
    """The strongest integration test: FK + Jacobians + M/C/G + controller +
    RK4 integration all exercised together. If computed_torque_control or
    any of the dynamics terms had a sign error, this would fail to track
    (typically by diverging or drifting, not just being slightly off)."""
    n = robot.n
    q0 = np.array([0.0, -1.2, 1.2, -0.3, 0.5, 0.0])
    qf = q0 + np.array([0.3, 0.2, -0.3, 0.15, -0.2, 0.4])
    T = 1.0

    Kp = np.eye(n) * 400.0
    Kd = np.eye(n) * 40.0

    def tau_fn(t, q, qdot):
        tt = min(t, T)
        q_des, qdot_des, qddot_des = quintic_trajectory(q0, qf, T, tt)
        return computed_torque_control(robot, q, qdot, q_des, qdot_des, qddot_des, Kp, Kd)

    dt = 0.01
    n_steps = int(1.3 / dt)
    q_hist, qdot_hist, t_hist = simulate(robot, q0, np.zeros(n), tau_fn, dt, n_steps)

    final_err = np.linalg.norm(q_hist[-1] - qf)
    assert final_err < 1e-3, f"final tracking error too large: {final_err}"


def test_pd_control_alone_does_not_perfectly_track(robot):
    """Sanity check on the test above: plain PD (no dynamics compensation)
    should NOT track as tightly as computed-torque control under the same
    gains, since it doesn't cancel gravity/Coriolis. This guards against a
    trivial bug where forward_dynamics silently ignores tau altogether
    (which would make every controller look perfect).

    Moderate gains (Kp=100/Kd=20 vs. the CTC test's Kp=400/Kd=40): raw PD
    at high gain is a genuinely stiff system now that M has real (small,
    ~0.1-2.6) values rather than placeholder zeros -- see
    test_wrist_dynamics_matches_mujoco below for the real values -- and
    high-gain PD with explicit RK4 at dt=0.01 is a control-tuning/numerical
    -stability issue at that point, not a sign of a dynamics bug.
    """
    n = robot.n
    q0 = np.array([0.0, -1.2, 1.2, -0.3, 0.5, 0.0])
    qf = q0 + np.array([0.3, 0.2, -0.3, 0.15, -0.2, 0.4])
    T = 1.0
    Kp = np.eye(n) * 100.0
    Kd = np.eye(n) * 20.0

    def tau_fn(t, q, qdot):
        tt = min(t, T)
        q_des, qdot_des, _ = quintic_trajectory(q0, qf, T, tt)
        return pd_control(q, qdot, q_des, qdot_des, Kp, Kd)

    dt = 0.01
    n_steps = int(1.3 / dt)
    q_hist, _, _ = simulate(robot, q0, np.zeros(n), tau_fn, dt, n_steps)
    assert not np.isnan(q_hist).any()
    final_err = np.linalg.norm(q_hist[-1] - qf)
    assert final_err > 1e-3  # gravity sag / steady-state error should show up
    assert final_err < 5.0   # ...but shouldn't be diverging either


def test_wrist_dynamics_matches_mujoco_order_of_magnitude(robot):
    """
    Regression guard for the gap this file used to document: robot.inertia
    was originally a zeros() placeholder, which (combined with link 6's COM
    sitting on its own rotation axis) made M[5,5] (wrist_3) collapse to
    ~1.3e-4 -- effectively singular -- and made forward_dynamics blow up to
    NaN for any ordinary torque at ordinary timesteps (confirmed: ~-1565
    rad/s^2 vs. MuJoCo's ground-truth ~-2.03 rad/s^2 for the same tau, q,
    qdot). robot.inertia + robot.armature are now populated from the real
    MuJoCo Menagerie model (see robot.py's comment for how), which fixes
    this. This test just checks the fix didn't regress: M[5,5] should be a
    normal-sized number now (MuJoCo's own value is ~0.1), and forward
    dynamics under a real torque shouldn't explode.
    """
    q = np.array([0.3, -0.9, 1.1, -0.4, 0.6, 0.2])
    qdot = np.array([0.1, -0.2, 0.05, 0.0, 0.1, -0.1])
    tau = np.array([5.0, -10.0, 3.0, 1.0, 0.5, -0.2])

    M = compute_M(robot, q)
    assert 0.05 < M[5, 5] < 0.5, f"M[5,5]={M[5,5]} -- expected an armature-scale value, not near-zero"

    qddot = forward_dynamics(robot, q, qdot, tau)
    assert np.max(np.abs(qddot)) < 100, f"qddot={qddot} -- forward dynamics blowing up again"
