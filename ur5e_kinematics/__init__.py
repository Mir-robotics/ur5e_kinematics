"""
ur5e_kinematics — a small kinematics/dynamics/control library for the UR5e.

    from ur5e_kinematics import UR5e, forward_kinematics, numerical_ik, analytical_ik

See README.md for the full API and a quickstart.
"""
from .robot import UR5e
from .fk import forward_kinematics, end_effector_pose, dh_transform
from .jacobian import (
    geometric_jacobian,
    forward_velocity_kinematics,
    inverse_velocity_kinematics,
)
from .ik import numerical_ik, analytical_ik, verify_solution
from .dynamics import compute_M, compute_C_qdot, compute_G, compute_dynamics, forward_dynamics, simulate
from .controller import pd_control, computed_torque_control
from .trajectory import quintic_coeffs, quintic_trajectory

__version__ = "0.1.0"

__all__ = [
    "UR5e",
    "forward_kinematics", "end_effector_pose", "dh_transform",
    "geometric_jacobian", "forward_velocity_kinematics", "inverse_velocity_kinematics",
    "numerical_ik", "analytical_ik", "verify_solution",
    "compute_M", "compute_C_qdot", "compute_G", "compute_dynamics", "forward_dynamics", "simulate",
    "pd_control", "computed_torque_control",
    "quintic_coeffs", "quintic_trajectory",
]
