#!/usr/bin/env python3
"""
Generate report-ready figures for ur5e_kinematics.

Run from the repo root, after scripts/record_results.py (some figures reuse
results/results.json rather than re-simulating):
    python3 scripts/record_results.py
    python3 scripts/plot_results.py

Writes PNGs to results/figures/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; drop this line if you want plt.show()
import matplotlib.pyplot as plt
import numpy as np

from ur5e_kinematics import UR5e, compute_M, forward_kinematics, simulate
from ur5e_kinematics.dynamics import GRAVITY

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run scripts/record_results.py first")
    with open(path) as f:
        return json.load(f)


def potential_energy(robot, q):
    transforms, _ = forward_kinematics(robot, q)
    U = 0.0
    for i in range(robot.n):
        R, p = transforms[i][:3, :3], transforms[i][:3, 3]
        com_world = p + R @ robot.com[i]
        U += robot.link_mass[i] * GRAVITY * com_world[2]
    return U


# ---------------------------------------------------------------------------
def fig_joint_trajectories():
    """Joint positions/velocities under free fall + light damping."""
    robot = UR5e()
    q0, qdot0 = np.zeros(6), np.zeros(6)
    tau_fn = lambda t, q, qdot: -0.1 * qdot
    dt, n_steps = 0.001, 800
    q_traj, qdot_traj, t_hist = simulate(robot, q0, qdot0, tau_fn, dt, n_steps)

    labeled = [("Position", "rad", "joint_positions.png", q_traj),
               ("Velocity", "rad/s", "joint_velocities.png", qdot_traj)]
    for label, unit, fname, data in labeled:
        plural = "Positions" if label == "Position" else "Velocities"
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"UR5e Joint {plural} -- Free Fall (q0=0) + Light Damping (-0.1*qdot)",
                     fontsize=15, fontweight="bold")
        for i in range(6):
            ax = axes[i // 3, i % 3]
            ax.plot(t_hist, data[:, i], linewidth=2, color=f"C{i}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"Joint {i+1} {label} ({unit})")
            ax.set_title(f"Joint {i+1}")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, fname)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    return q_traj, qdot_traj, t_hist


def fig_energy_balance(robot, q_traj, qdot_traj, t_hist):
    """
    The key fix vs. a KE-only plot: KE rising under free fall isn't a bug
    (it's PE converting to KE), but a bare KE curve invites exactly that
    question. Plotting KE + PE + total makes the physics self-evident:
    total energy should decrease slowly (damping is dissipative), not swing
    wildly, and PE's drop should almost match KE's rise.
    """
    KE = np.array([0.5 * qdot_traj[i] @ compute_M(robot, q_traj[i]) @ qdot_traj[i]
                   for i in range(len(t_hist))])
    PE = np.array([potential_energy(robot, q_traj[i]) for i in range(len(t_hist))])
    E = KE + PE

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hist, KE, label="Kinetic energy", linewidth=2, color="tab:red")
    ax.plot(t_hist, PE, label="Potential energy", linewidth=2, color="tab:blue")
    ax.plot(t_hist, E, label="Total mechanical energy", linewidth=2.5, color="black", linestyle="--")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Energy (J)", fontsize=12)
    ax.set_title("UR5e Energy Balance -- Free Fall + Light Damping\n"
                 "(PE converts to KE under gravity; total energy decreases slowly from damping)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "energy_balance.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}  (total energy drift: {E[-1]-E[0]:.3f} J over {t_hist[-1]:.1f}s)")


def fig_dynamics_analysis(robot):
    q = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])
    M = compute_M(robot, q)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("UR5e Dynamics Analysis", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    im = ax1.imshow(M, cmap="viridis")
    ax1.set_title("Mass matrix M(q)")
    ax1.set_xlabel("Joint"); ax1.set_ylabel("Joint")
    ax1.set_xticks(range(6)); ax1.set_yticks(range(6))
    ax1.set_xticklabels(range(1, 7)); ax1.set_yticklabels(range(1, 7))
    plt.colorbar(im, ax=ax1, label="kg*m^2")

    ax2 = axes[0, 1]
    ax2.bar(range(1, 7), np.diag(M), color="skyblue", edgecolor="navy")
    ax2.set_title("Mass matrix diagonal")
    ax2.set_xlabel("Joint"); ax2.set_ylabel("Inertia (kg*m^2)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    eigs = np.linalg.eigvalsh(M)
    ax3.bar(range(1, 7), sorted(eigs), color="lightgreen", edgecolor="darkgreen")
    ax3.axhline(0, color="red", linestyle="--", linewidth=1)
    ax3.set_title(f"Eigenvalues of M(q)  (cond = {np.linalg.cond(M):.1f})")
    ax3.set_xlabel("index"); ax3.set_ylabel("eigenvalue")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, f"Symmetric: {np.allclose(M, M.T)}\nAll eigenvalues > 0: {bool(np.all(eigs>0))}\n"
              f"Condition number: {np.linalg.cond(M):.1f}",
              fontsize=15, ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow"))
    ax4.axis("off")
    ax4.set_title("Sanity checks")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "dynamics_analysis.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_closed_loop_tracking(results):
    cl = results["closed_loop"]
    t = np.array(cl["t_hist"])
    err_ctc = np.array(cl["computed_torque_control"]["tracking_error"])
    err_pd = np.array(cl["pd_control"]["tracking_error"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(t, np.maximum(err_ctc, 1e-12), label=f"Computed-torque control ({cl['computed_torque_control']['gains']})",
                linewidth=2, color="tab:green")
    ax.semilogy(t, np.maximum(err_pd, 1e-12), label=f"Plain PD ({cl['pd_control']['gains']})",
                linewidth=2, color="tab:orange")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("||q - q_desired|| (rad, log scale)", fontsize=12)
    ax.set_title("Closed-loop trajectory tracking\n(RK4-simulated against the library's own forward_dynamics)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "closed_loop_tracking.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_mujoco_crossvalidation(results):
    mj = results.get("mujoco_crossvalidation", {})
    if "skipped" in mj:
        print(f"skipping MuJoCo cross-validation figure: {mj['skipped']}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cross-validation vs. the official MuJoCo UR5e model (20 random configs, no calibration)",
                 fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.bar(["max", "mean"], [mj["fk_position_error_m"]["max"]*1000, mj["fk_position_error_m"]["mean"]*1000],
            color=["salmon", "lightblue"])
    ax1.set_title("FK position error (mm)")
    ax1.set_ylabel("mm")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(["max", "mean"], [mj["mass_matrix_max_abs_error"]["max"], mj["mass_matrix_max_abs_error"]["mean"]],
            color=["salmon", "lightblue"])
    ax2.set_title("Mass matrix M(q) abs. error")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.bar(["max", "mean"], [mj["bias_force_max_abs_error_Nm"]["max"], mj["bias_force_max_abs_error_Nm"]["mean"]],
            color=["salmon", "lightblue"])
    ax3.set_title("Bias force C*qdot+G error (N*m)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "mujoco_crossvalidation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    robot = UR5e()
    results = load_results()

    q_traj, qdot_traj, t_hist = fig_joint_trajectories()
    fig_energy_balance(robot, q_traj, qdot_traj, t_hist)
    fig_dynamics_analysis(robot)
    fig_closed_loop_tracking(results)
    fig_mujoco_crossvalidation(results)

    print(f"\nAll figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
