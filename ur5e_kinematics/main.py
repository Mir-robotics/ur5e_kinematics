import numpy as np
try:
    from .robot import UR5e
    from .fk import forward_kinematics, end_effector_pose
    from .ik import numerical_ik, verify_solution, analytical_ik
    from .jacobian import forward_velocity_kinematics, inverse_velocity_kinematics
    from .dynamics import compute_M, compute_C_qdot, compute_G
    from .trajectory import quintic_trajectory
except ImportError:
    from robot import UR5e
    from fk import forward_kinematics, end_effector_pose
    from ik import numerical_ik, verify_solution, analytical_ik
    from jacobian import forward_velocity_kinematics, inverse_velocity_kinematics
    from dynamics import compute_M, compute_C_qdot, compute_G
    from trajectory import quintic_trajectory

if __name__ == "__main__":
    robot = UR5e()
    robot.sanity_check()

    q = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])

    # Lab 1.1
    pos, R = end_effector_pose(robot, q)
    print("\n[Lab 1.1] EE position:", pos)

    # Lab 1.2
    _, T_target = forward_kinematics(robot, q)
    q_sol, ok, it = numerical_ik(robot, T_target, q0=np.zeros(6))
    print(f"\n[Lab 1.2] Numerical IK converged={ok} in {it} iters, "
          f"valid={verify_solution(robot, q_sol, T_target)}")

    a_sols = analytical_ik(robot, T_target)
    print(f"[Lab 1.2] Analytical IK: {len(a_sols)} verified branch(es); "
          f"first = {np.round(a_sols[0], 4) if a_sols else None}")

    # Lab 2
    qdot = np.array([0.1, 0, 0, 0, 0, 0])
    xdot = forward_velocity_kinematics(robot, q, qdot)
    print("\n[Lab 2] EE twist:", xdot)

    # Lab 3
    print("\n[Lab 3] M(q):\n", compute_M(robot, q))
    print("[Lab 3] G(q):", compute_G(robot, q))
