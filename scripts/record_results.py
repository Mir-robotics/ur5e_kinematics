#!/usr/bin/env python3
"""
Record validation results for ur5e_kinematics, for a README/report/GitHub.

Produces:
  results/results.json   -- machine-readable, everything below
  results/RESULTS.md      -- human-readable summary, paste-ready for GitHub

Run from the repo root:
    python3 scripts/record_results.py

MuJoCo cross-validation sections are skipped automatically if `mujoco` isn't
installed or models/ur5e/scene.xml isn't found -- everything else still runs.
"""
import json
import os
import time
from datetime import datetime

import numpy as np

from ur5e_kinematics import (
    UR5e, forward_kinematics, analytical_ik, numerical_ik, verify_solution,
    geometric_jacobian, compute_M, compute_G, compute_C_qdot,
    forward_dynamics, simulate, pd_control, computed_torque_control,
    quintic_trajectory,
)
from ur5e_kinematics.dynamics import GRAVITY

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Kinematics: FK timing, analytical IK (verified branches, ground-truth
#    match), numerical IK convergence.
# ---------------------------------------------------------------------------
def test_kinematics(robot, results, n_trials=20):
    section("1. Kinematics")
    rng = np.random.default_rng(0)

    q_probe = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])
    t0 = time.time()
    for _ in range(1000):
        _, T_probe = forward_kinematics(robot, q_probe)
    fk_time = (time.time() - t0) / 1000

    analytical_ok, analytical_branch_counts, analytical_pos_errs = 0, [], []
    numerical_ok, numerical_iters, numerical_pos_errs = 0, [], []

    for _ in range(n_trials):
        q_true = rng.uniform(-np.pi, np.pi, 6)
        _, T = forward_kinematics(robot, q_true)

        sols = analytical_ik(robot, T)
        analytical_branch_counts.append(len(sols))
        if sols:
            errs = [np.linalg.norm(np.mod(q - q_true + np.pi, 2 * np.pi) - np.pi) for q in sols]
            if min(errs) < 1e-4:
                analytical_ok += 1
            analytical_pos_errs.append(min(
                np.linalg.norm(forward_kinematics(robot, q)[1][:3, 3] - T[:3, 3]) for q in sols
            ))

        q_sol, converged, iters = numerical_ik(robot, T, q0=np.zeros(6))
        if converged:
            numerical_ok += 1
            numerical_iters.append(iters)
            numerical_pos_errs.append(np.linalg.norm(forward_kinematics(robot, q_sol)[1][:3, 3] - T[:3, 3]))

    results["kinematics"] = {
        "fk_time_per_call_s": fk_time,
        "n_trials": n_trials,
        "analytical_ik": {
            "success_rate": analytical_ok / n_trials,
            "avg_branches_found": float(np.mean(analytical_branch_counts)),
            "max_position_error_m": float(np.max(analytical_pos_errs)) if analytical_pos_errs else None,
        },
        "numerical_ik": {
            "convergence_rate": numerical_ok / n_trials,
            "avg_iterations": float(np.mean(numerical_iters)) if numerical_iters else None,
            "max_position_error_m": float(np.max(numerical_pos_errs)) if numerical_pos_errs else None,
        },
    }
    print(f"FK: {fk_time*1e6:.1f} us/call")
    print(f"Analytical IK: {analytical_ok}/{n_trials} recovered ground truth, "
          f"avg {np.mean(analytical_branch_counts):.1f} verified branches")
    print(f"Numerical IK: {numerical_ok}/{n_trials} converged, "
          f"avg {np.mean(numerical_iters):.1f} iterations")


# ---------------------------------------------------------------------------
# 2. Dynamics: mass matrix conditioning, self-consistency, energy balance
#    (fixed version of the free-fall demo -- tracks TOTAL energy, not just
#    KE, and reports the actual dissipation so the numbers are self-
#    explanatory instead of "why is kinetic energy growing?").
# ---------------------------------------------------------------------------
def potential_energy(robot, q):
    transforms, _ = forward_kinematics(robot, q)
    U = 0.0
    for i in range(robot.n):
        R, p = transforms[i][:3, :3], transforms[i][:3, 3]
        com_world = p + R @ robot.com[i]
        U += robot.link_mass[i] * GRAVITY * com_world[2]
    return U


def test_dynamics(robot, results):
    section("2. Dynamics")
    q = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])
    qdot = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.15])

    M = compute_M(robot, q)
    G = compute_G(robot, q)
    C_qdot = compute_C_qdot(robot, q, qdot)
    cond_M = float(np.linalg.cond(M))
    eigs = np.linalg.eigvalsh(M)

    # forward-dynamics self-consistency: M @ qddot should equal tau - C - G
    tau = np.array([1.0, -2.0, 0.5, 0.1, 0.2, -0.1])
    qddot = forward_dynamics(robot, q, qdot, tau)
    consistency_error = float(np.linalg.norm(M @ qddot - (tau - C_qdot - G)))

    results["dynamics"] = {
        "probe_q": q.tolist(),
        "mass_matrix_diagonal": np.diag(M).tolist(),
        "mass_matrix_condition_number": cond_M,
        "mass_matrix_eigenvalues": eigs.tolist(),
        "mass_matrix_symmetric": bool(np.allclose(M, M.T)),
        "gravity_norm_Nm": float(np.linalg.norm(G)),
        "coriolis_term_norm_Nm": float(np.linalg.norm(C_qdot)),
        "forward_dynamics_self_consistency_error": consistency_error,
    }
    print(f"cond(M) = {cond_M:.2f}   eigenvalues in [{eigs.min():.4f}, {eigs.max():.4f}]  (all > 0: {bool(np.all(eigs > 0))})")
    print(f"|G(q)| = {np.linalg.norm(G):.2f} N*m   |C(q,qdot)qdot| = {np.linalg.norm(C_qdot):.4f} N*m")
    print(f"forward-dynamics self-consistency error: {consistency_error:.2e}")

    # --- energy balance under free fall + light damping ---
    q0, qdot0 = np.zeros(6), np.zeros(6)
    tau_fn = lambda t, q, qdot: -0.1 * qdot
    dt, n_steps = 0.001, 800  # 0.8s, enough for damping to visibly bite without a slow re-run
    q_traj, qdot_traj, t_hist = simulate(robot, q0, qdot0, tau_fn, dt, n_steps)

    KE = np.array([0.5 * qdot_traj[i] @ compute_M(robot, q_traj[i]) @ qdot_traj[i] for i in range(len(t_hist))])
    PE = np.array([potential_energy(robot, q_traj[i]) for i in range(len(t_hist))])
    E = KE + PE

    results["energy_balance"] = {
        "duration_s": dt * n_steps,
        "dt": dt,
        "t_hist": t_hist.tolist(),
        "kinetic_energy_J": KE.tolist(),
        "potential_energy_J": PE.tolist(),
        "total_energy_J": E.tolist(),
        "total_energy_drift_J": float(E[-1] - E[0]),
        "note": "damping = -0.1*qdot is dissipative, so total energy should "
                "monotonically decrease (not stay exactly constant); a large "
                "*increase* would indicate a sign error somewhere in G/C.",
    }
    print(f"Energy balance over {dt*n_steps:.1f}s: E[0]={E[0]:.3f} J -> E[-1]={E[-1]:.3f} J "
          f"(monotonic decrease expected from damping)")


# ---------------------------------------------------------------------------
# 3. Closed-loop control: computed-torque vs. plain PD tracking a quintic
#    trajectory, integrated with the library's own forward_dynamics.
# ---------------------------------------------------------------------------
def test_closed_loop(robot, results):
    section("3. Closed-loop control")
    q0 = np.array([0.0, -1.2, 1.2, -0.3, 0.5, 0.0])
    qf = q0 + np.array([0.3, 0.2, -0.3, 0.15, -0.2, 0.4])
    T = 1.0
    dt, n_steps = 0.01, 130

    Kp_ctc, Kd_ctc = np.eye(6) * 400.0, np.eye(6) * 40.0
    def tau_ctc(t, q, qdot):
        tt = min(t, T)
        q_des, qdot_des, qddot_des = quintic_trajectory(q0, qf, T, tt)
        return computed_torque_control(robot, q, qdot, q_des, qdot_des, qddot_des, Kp_ctc, Kd_ctc)
    q_ctc, _, t_hist = simulate(robot, q0, np.zeros(6), tau_ctc, dt, n_steps)
    err_ctc = np.linalg.norm(q_ctc - qf, axis=1)

    Kp_pd, Kd_pd = np.eye(6) * 100.0, np.eye(6) * 20.0
    def tau_pd(t, q, qdot):
        tt = min(t, T)
        q_des, qdot_des, _ = quintic_trajectory(q0, qf, T, tt)
        return pd_control(q, qdot, q_des, qdot_des, Kp_pd, Kd_pd)
    q_pd, _, _ = simulate(robot, q0, np.zeros(6), tau_pd, dt, n_steps)
    err_pd = np.linalg.norm(q_pd - qf, axis=1)

    results["closed_loop"] = {
        "t_hist": t_hist.tolist(),
        "computed_torque_control": {"gains": "Kp=400, Kd=40", "tracking_error": err_ctc.tolist(),
                                     "final_error": float(err_ctc[-1])},
        "pd_control": {"gains": "Kp=100, Kd=20", "tracking_error": err_pd.tolist(),
                        "final_error": float(err_pd[-1])},
    }
    print(f"Computed-torque control: final tracking error = {err_ctc[-1]:.2e} rad")
    print(f"Plain PD control:        final tracking error = {err_pd[-1]:.4f} rad "
          f"(nonzero as expected -- no dynamics compensation)")


# ---------------------------------------------------------------------------
# 4. MuJoCo cross-validation (skipped gracefully if unavailable).
# ---------------------------------------------------------------------------
def test_mujoco_crossvalidation(robot, results):
    section("4. MuJoCo cross-validation")
    model_path = os.environ.get(
        "MUJOCO_UR5E_SCENE", os.path.join(REPO_ROOT, "models", "ur5e", "scene.xml")
    )
    try:
        import mujoco
    except ImportError:
        print("mujoco not installed -- skipping (pip install ur5e_kinematics[sim])")
        results["mujoco_crossvalidation"] = {"skipped": "mujoco not installed"}
        return
    if not os.path.exists(model_path):
        print(f"model not found at {model_path} -- skipping")
        results["mujoco_crossvalidation"] = {"skipped": f"model not found at {model_path}"}
        return

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    rng = np.random.default_rng(42)
    pos_errs, rot_errs, mass_matrix_errs, bias_errs = [], [], [], []
    for _ in range(20):
        q = rng.uniform(-3.0, 3.0, 6)
        qdot = rng.uniform(-1.0, 1.0, 6)

        d.qpos[:6], d.qvel[:6] = q, qdot
        mujoco.mj_forward(m, d)

        _, T = forward_kinematics(robot, q)
        pos_errs.append(float(np.linalg.norm(d.site_xpos[site_id] - T[:3, 3])))
        rot_errs.append(float(np.linalg.norm(d.site_xmat[site_id].reshape(3, 3) - T[:3, :3])))

        M_mj = np.zeros((6, 6))
        mujoco.mj_fullM(m, d, M_mj)
        mass_matrix_errs.append(float(np.max(np.abs(M_mj - compute_M(robot, q)))))

        bias_ours = compute_C_qdot(robot, q, qdot) + compute_G(robot, q)
        bias_errs.append(float(np.max(np.abs(d.qfrc_bias - bias_ours))))

    results["mujoco_crossvalidation"] = {
        "model": "google-deepmind/mujoco_menagerie universal_robots_ur5e",
        "n_trials": 20,
        "fk_position_error_m": {"max": max(pos_errs), "mean": float(np.mean(pos_errs))},
        "fk_rotation_error": {"max": max(rot_errs), "mean": float(np.mean(rot_errs))},
        "mass_matrix_max_abs_error": {"max": max(mass_matrix_errs), "mean": float(np.mean(mass_matrix_errs))},
        "bias_force_max_abs_error_Nm": {"max": max(bias_errs), "mean": float(np.mean(bias_errs))},
    }
    print(f"FK position error:  max {max(pos_errs)*1000:.3f} mm, mean {np.mean(pos_errs)*1000:.3f} mm")
    print(f"FK rotation error:  max {max(rot_errs):.2e}")
    print(f"Mass matrix error:  max {max(mass_matrix_errs):.4f}, mean {np.mean(mass_matrix_errs):.4f}")
    print(f"Bias force error:   max {max(bias_errs):.4f} N*m, mean {np.mean(bias_errs):.4f} N*m")


# ---------------------------------------------------------------------------
def write_markdown_summary(results):
    lines = [f"# ur5e_kinematics -- validation results",
             f"", f"Generated {results['timestamp']} by `scripts/record_results.py`.", ""]

    k = results["kinematics"]
    lines += [
        "## Kinematics", "",
        f"- FK: **{k['fk_time_per_call_s']*1e6:.1f} us/call**",
        f"- Analytical IK: **{k['analytical_ik']['success_rate']*100:.0f}%** of {k['n_trials']} random "
        f"configs recovered exactly (avg **{k['analytical_ik']['avg_branches_found']:.1f}** verified branches, "
        f"max position error **{k['analytical_ik']['max_position_error_m']:.2e} m**)",
        f"- Numerical IK: **{k['numerical_ik']['convergence_rate']*100:.0f}%** convergence, "
        f"avg **{k['numerical_ik']['avg_iterations']:.1f}** iterations", "",
    ]

    dy = results["dynamics"]
    lines += [
        "## Dynamics", "",
        f"- Mass matrix: symmetric = **{dy['mass_matrix_symmetric']}**, "
        f"condition number = **{dy['mass_matrix_condition_number']:.1f}**, "
        f"all eigenvalues positive = **{all(e > 0 for e in dy['mass_matrix_eigenvalues'])}**",
        f"- Forward-dynamics self-consistency error: **{dy['forward_dynamics_self_consistency_error']:.2e}**", "",
    ]
    eb = results["energy_balance"]
    lines += [
        f"- Energy balance ({eb['duration_s']:.1f}s free-fall + light damping): total mechanical energy "
        f"drift = **{eb['total_energy_drift_J']:.3f} J** (monotonic decrease expected; a large *increase* "
        "would indicate a sign error)", "",
    ]

    cl = results["closed_loop"]
    lines += [
        "## Closed-loop control (RK4-simulated against the library's own dynamics)", "",
        f"- Computed-torque control ({cl['computed_torque_control']['gains']}): final tracking error = "
        f"**{cl['computed_torque_control']['final_error']:.2e} rad**",
        f"- Plain PD ({cl['pd_control']['gains']}): final tracking error = "
        f"**{cl['pd_control']['final_error']:.4f} rad** (nonzero as expected)", "",
    ]

    mj = results.get("mujoco_crossvalidation", {})
    if "skipped" in mj:
        lines += ["## MuJoCo cross-validation", "", f"Skipped: {mj['skipped']}", ""]
    else:
        lines += [
            "## MuJoCo cross-validation", "",
            f"Against the official [mujoco_menagerie UR5e model]"
            f"(https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e), "
            f"{mj['n_trials']} random configurations, **no calibration transform applied**:", "",
            f"| Quantity | Max error | Mean error |",
            f"|---|---|---|",
            f"| FK position | {mj['fk_position_error_m']['max']*1000:.3f} mm | {mj['fk_position_error_m']['mean']*1000:.3f} mm |",
            f"| FK rotation | {mj['fk_rotation_error']['max']:.2e} | {mj['fk_rotation_error']['mean']:.2e} |",
            f"| Mass matrix M(q) | {mj['mass_matrix_max_abs_error']['max']:.4f} | {mj['mass_matrix_max_abs_error']['mean']:.4f} |",
            f"| Bias force C(q,qdot)qdot+G(q) | {mj['bias_force_max_abs_error_Nm']['max']:.4f} N*m | {mj['bias_force_max_abs_error_Nm']['mean']:.4f} N*m |",
            "",
        ]

    path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {path}")


def main():
    section(f"ur5e_kinematics validation run -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    robot = UR5e()
    results = {"timestamp": datetime.now().isoformat()}

    test_kinematics(robot, results)
    test_dynamics(robot, results)
    test_closed_loop(robot, results)
    test_mujoco_crossvalidation(robot, results)

    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {json_path}")
    write_markdown_summary(results)


if __name__ == "__main__":
    main()
