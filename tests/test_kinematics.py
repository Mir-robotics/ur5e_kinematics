import numpy as np
import pytest

from ur5e_kinematics import (
    UR5e, forward_kinematics, numerical_ik, analytical_ik, verify_solution,
    geometric_jacobian, forward_velocity_kinematics, inverse_velocity_kinematics,
    compute_M, compute_C_qdot, compute_G,
)


@pytest.fixture
def robot():
    return UR5e()


def random_q(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, 6)


@pytest.mark.parametrize("seed", range(10))
def test_analytical_ik_matches_fk(robot, seed):
    q_true = random_q(seed)
    _, T = forward_kinematics(robot, q_true)
    sols = analytical_ik(robot, T)
    assert len(sols) > 0, "analytical_ik found no verified branch"
    for q in sols:
        assert verify_solution(robot, q, T)
    # at least one branch should reproduce q_true (mod 2*pi)
    errs = [np.linalg.norm(np.mod(q - q_true + np.pi, 2 * np.pi) - np.pi) for q in sols]
    assert min(errs) < 1e-4


@pytest.mark.parametrize("seed", range(5))
def test_numerical_ik_converges(robot, seed):
    q_true = random_q(seed)
    _, T = forward_kinematics(robot, q_true)
    q_sol, ok, _ = numerical_ik(robot, T, q0=np.zeros(6))
    assert ok
    assert verify_solution(robot, q_sol, T)


@pytest.mark.parametrize("seed", range(5))
def test_jacobian_matches_finite_difference(robot, seed):
    q = random_q(seed)
    eps = 1e-6
    J = geometric_jacobian(robot, q)
    _, T0 = forward_kinematics(robot, q)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        _, T1 = forward_kinematics(robot, q + dq)
        dp_numeric = (T1[:3, 3] - T0[:3, 3]) / eps
        dp_jacobian = J[:3, i]
        assert np.allclose(dp_numeric, dp_jacobian, atol=1e-3)


def test_forward_inverse_velocity_roundtrip(robot):
    q = random_q(0)
    qdot = np.array([0.1, -0.2, 0.05, 0.0, 0.1, -0.1])
    xdot = forward_velocity_kinematics(robot, q, qdot)
    qdot_back = inverse_velocity_kinematics(robot, q, xdot)
    assert np.allclose(qdot, qdot_back, atol=1e-6)


def test_mass_matrix_symmetric_positive_semidefinite(robot):
    q = random_q(1)
    M = compute_M(robot, q)
    assert np.allclose(M, M.T)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > -1e-8)


def test_gravity_zero_at_zero_mass_would_be_zero():
    # sanity check on compute_G's sign/units: increasing a link's mass
    # should monotonically increase the magnitude of its gravity torque.
    robot = UR5e()
    q = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])
    G1 = compute_G(robot, q)
    robot.link_mass[1] *= 2
    G2 = compute_G(robot, q)
    assert abs(G2[1]) > abs(G1[1])


def test_coriolis_matches_passivity_property(robot):
    """Standard robotics identity: Mdot(q) - 2*C(q,qdot) is skew-symmetric."""
    q = random_q(2)
    qdot = np.array([0.1, -0.2, 0.05, 0.0, 0.1, -0.1])
    dt = 1e-6
    M0 = compute_M(robot, q)
    M1 = compute_M(robot, q + qdot * dt)
    Mdot = (M1 - M0) / dt

    n = robot.n
    dM = np.zeros((n, n, n))
    for k in range(n):
        dq = np.zeros(n)
        dq[k] = 1e-6
        dM[:, :, k] = (compute_M(robot, q + dq) - compute_M(robot, q - dq)) / (2e-6)
    C = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            C[k, j] = 0.5 * sum(
                (dM[k, j, i] + dM[k, i, j] - dM[i, j, k]) * qdot[i] for i in range(n)
            )
    skew = Mdot - 2 * C
    assert np.max(np.abs(skew + skew.T)) < 1e-4
