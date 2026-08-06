"""Lab 2 — Velocity Kinematics / Jacobian"""
import numpy as np
try:
    from .fk import forward_kinematics
except ImportError:
    from fk import forward_kinematics


def geometric_jacobian(robot, q):
    transforms, T0n = forward_kinematics(robot, q)
    o_n = T0n[:3, 3]
    J = np.zeros((6, robot.n))
    for i in range(robot.n):
        if i == 0:
            z_im1 = np.array([0., 0., 1.])
            o_im1 = np.zeros(3)
        else:
            z_im1 = transforms[i-1][:3, 2]
            o_im1 = transforms[i-1][:3, 3]
        J[:3, i] = np.cross(z_im1, o_n - o_im1)
        J[3:, i] = z_im1
    return J


def forward_velocity_kinematics(robot, q, qdot):
    """xdot (6,) = J(q) @ qdot"""
    J = geometric_jacobian(robot, q)
    return J @ qdot


def inverse_velocity_kinematics(robot, q, xdot, damping=1e-6):
    """qdot from desired twist xdot, damped least squares (robust near singularities)."""
    J = geometric_jacobian(robot, q)
    JJt = J @ J.T + (damping**2) * np.eye(6)
    return J.T @ np.linalg.solve(JJt, xdot)


if __name__ == "__main__":
    from robot import UR5e
    r = UR5e()
    q = np.array([0.1, -1.0, 1.2, 0.3, 0.7, 0.0])
    qdot = np.array([0.1, 0, 0, 0, 0, 0])
    xdot = forward_velocity_kinematics(r, q, qdot)
    print("End-effector twist:", xdot)
    qdot_back = inverse_velocity_kinematics(r, q, xdot)
    print("Recovered qdot:", qdot_back)  # should match qdot away from singularities
