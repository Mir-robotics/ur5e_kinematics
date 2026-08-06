"""Lab-adjacent — simple controllers to exercise fk/ik/dynamics together."""
import numpy as np
try:
    from .dynamics import compute_M, compute_C_qdot, compute_G
except ImportError:
    from dynamics import compute_M, compute_C_qdot, compute_G


def pd_control(q, qdot, q_des, qdot_des, Kp, Kd):
    return Kp @ (q_des - q) + Kd @ (qdot_des - qdot)


def computed_torque_control(robot, q, qdot, q_des, qdot_des, qddot_des, Kp, Kd):
    """tau = M(q)[qddot_des + Kd*e_dot + Kp*e] + C(q,qdot)qdot + G(q)"""
    e = q_des - q
    e_dot = qdot_des - qdot
    M = compute_M(robot, q)
    C_qdot = compute_C_qdot(robot, q, qdot)
    G = compute_G(robot, q)
    return M @ (qddot_des + Kd @ e_dot + Kp @ e) + C_qdot + G
