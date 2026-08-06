"""Lab 1.1 — Forward Kinematics"""
import numpy as np


def dh_transform(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,      sa,     ca,    d],
        [0.,       0.,      0.,   1.]
    ])


def forward_kinematics(robot, q, up_to_joint=None):
    """Returns (list_of_T0_i, T0_n)."""
    n = robot.n if up_to_joint is None else up_to_joint
    T = np.eye(4)
    transforms = []
    for i in range(n):
        a, alpha, d, off = robot.dh[i]
        A = dh_transform(a, alpha, d, q[i] + off)
        T = T @ A
        transforms.append(T.copy())
    return transforms, T


def end_effector_pose(robot, q):
    _, T = forward_kinematics(robot, q)
    return T[:3, 3], T[:3, :3]


if __name__ == "__main__":
    from robot import UR5e
    r = UR5e()
    q_test = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])
    pos, R = end_effector_pose(r, q_test)
    print("Position:", pos)
    print("Orientation:\n", R)
