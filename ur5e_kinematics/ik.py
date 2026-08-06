"""
Inverse Kinematics for the UR5e (offset-wrist 6R arm).

- numerical_ik(): damped-least-squares Jacobian solver. Always converges from
  a reasonable seed; used to cross-check the analytical solver below.
- analytical_ik(): closed-form geometric decoupling (wrist-center -> theta1,
  theta5, theta6, then planar law-of-cosines -> theta2, theta3, theta4).
  Every branch is verified against verify_solution() before being returned,
  so only geometrically valid solutions come back.
"""
import numpy as np
try:
    from .fk import forward_kinematics, end_effector_pose, dh_transform
    from .jacobian import geometric_jacobian
except ImportError:
    from fk import forward_kinematics, end_effector_pose, dh_transform
    from jacobian import geometric_jacobian


def numerical_ik(robot, T_target, q0, max_iter=500, tol=1e-6, damping=1e-3):
    q = np.array(q0, dtype=float)
    p_target = T_target[:3, 3]
    R_target = T_target[:3, :3]
    for it in range(max_iter):
        pos, R = end_effector_pose(robot, q)
        pos_err = p_target - pos
        R_err = R_target @ R.T
        rot_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        err = np.hstack([pos_err, rot_err])
        if np.linalg.norm(err) < tol:
            return q, True, it
        J = geometric_jacobian(robot, q)
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        q = q + dq
    return q, False, max_iter


def analytical_ik(robot, T_target, pos_tol=1e-4, rot_tol=1e-4):
    """
    Closed-form geometric IK for a UR5e-type arm (offset-wrist 6R, standard
    DH, joint axes matching robot.dh: a=[0,a2,a3,0,0,0], alpha=[pi/2,0,0,pi/2,-pi/2,0]).

    Method (matches the standard UR5/UR5e decoupling):
      1) theta1 from the wrist-center (origin of frame 5) projected onto the
         base XY plane -> two branches (left/right shoulder).
      2) theta5 from the target position relative to the theta1 frame ->
         two branches (wrist up/down).
      3) theta6 from matching the target x/y axes given theta1, theta5.
      4) theta2, theta3 from the wrist-center position relative to frame 1
         via the law of cosines with a2, a3 -> two branches (elbow up/down).
      5) theta4 as the remaining rotation to match full orientation.

    All 8 branch combinations are generated, then each is checked with
    forward_kinematics + verify_solution(); only verified solutions are
    returned (silently discards branches that hit a NaN/singularity or a
    combination that doesn't reconstruct T_target).
    """
    d1 = robot.dh[0, 2]
    a2 = robot.dh[1, 0]
    a3 = robot.dh[2, 0]
    d4 = robot.dh[3, 2]
    d5 = robot.dh[4, 2]
    d6 = robot.dh[5, 2]
    off = robot.dh[:, 3]

    T06 = np.asarray(T_target, dtype=float)

    # ---- wrist center (origin of frame 5), expressed in base frame ----
    p05 = T06 @ np.array([0.0, 0.0, -d6, 1.0])
    p05 = p05[:3]

    solutions = []

    # ---- theta1: two shoulder branches ----
    r_xy = np.hypot(p05[0], p05[1])
    if r_xy < abs(d4):
        return []  # unreachable wrist-center offset
    psi = np.arctan2(p05[1], p05[0])
    phi = np.arccos(np.clip(d4 / r_xy, -1.0, 1.0))
    theta1_options = [psi + phi + np.pi / 2, psi - phi + np.pi / 2]

    p06 = T06[:3, 3]

    for theta1 in theta1_options:
        # ---- theta5: two wrist branches ----
        num5 = p06[0] * np.sin(theta1) - p06[1] * np.cos(theta1) - d4
        c5 = np.clip(num5 / d6, -1.0, 1.0)
        theta5_options = [np.arccos(c5), -np.arccos(c5)]

        for theta5 in theta5_options:
            # ---- theta6 ----
            if abs(np.sin(theta5)) < 1e-8:
                theta6_options = [0.0]  # singular: theta4/theta6 axes align, pick 0
            else:
                T60 = np.linalg.inv(T06)
                zy = -T60[1, 0] * np.sin(theta1) + T60[1, 1] * np.cos(theta1)
                zx = T60[0, 0] * np.sin(theta1) - T60[0, 1] * np.cos(theta1)
                theta6_options = [np.arctan2(zy / np.sin(theta5), zx / np.sin(theta5))]

            for theta6 in theta6_options:
                # ---- theta2, theta3 via planar law of cosines ----
                T01 = dh_transform(0.0, np.pi / 2, d1, theta1 + off[0])
                T56 = dh_transform(0.0, -np.pi / 2, d5, theta5 + off[4])
                T67 = dh_transform(0.0, 0.0, d6, theta6 + off[5])

                T14 = np.linalg.inv(T01) @ T06 @ np.linalg.inv(T56 @ T67)
                p14 = T14[:3, 3]
                # planar distance from joint-2 origin to joint-4 origin
                px, py = p14[0], p14[1]
                dist2 = px**2 + py**2
                c3 = (dist2 - a2**2 - a3**2) / (2 * a2 * a3)
                c3 = np.clip(c3, -1.0, 1.0)
                for sign in (1.0, -1.0):
                    theta3 = sign * np.arccos(c3)
                    theta2 = np.arctan2(-py, -px) + np.arcsin(
                        np.clip(a3 * np.sin(theta3) / max(np.sqrt(dist2), 1e-9), -1.0, 1.0)
                    )

                    # theta4: remaining rotation about the shared axis, taken
                    # directly from T14's rotation once theta2/theta3 fixed.
                    T02_ = T01 @ dh_transform(a2, 0.0, 0.0, theta2 + off[1])
                    T03_ = T02_ @ dh_transform(a3, 0.0, 0.0, theta3 + off[2])
                    T34_ = np.linalg.inv(T03_) @ (T01 @ T14)
                    theta4 = np.arctan2(T34_[1, 0], T34_[0, 0]) - off[3]

                    q = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                    if np.any(~np.isfinite(q)):
                        continue
                    solutions.append(q)

    verified = [q for q in solutions if verify_solution(robot, q, T_target, pos_tol, rot_tol)]
    return verified


def verify_solution(robot, q, T_target, pos_tol=1e-4, rot_tol=1e-4):
    _, T = forward_kinematics(robot, q)
    return (np.allclose(T[:3, 3], T_target[:3, 3], atol=pos_tol) and
            np.allclose(T[:3, :3], T_target[:3, :3], atol=rot_tol))


if __name__ == "__main__":
    from robot import UR5e
    r = UR5e()
    q_true = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])
    _, T_target = forward_kinematics(r, q_true)

    q_sol, ok, iters = numerical_ik(r, T_target, q0=np.zeros(6))
    print("Numerical IK converged:", ok, "in", iters, "iters, valid?",
          verify_solution(r, q_sol, T_target))

    sols = analytical_ik(r, T_target)
    print(f"Analytical IK: {len(sols)} verified branch(es)")
    for i, q in enumerate(sols):
        print(f"  [{i}] q =", np.round(q, 4))
