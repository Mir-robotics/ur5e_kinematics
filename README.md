# ur5e_kinematics

A small, dependency-light Python library for UR5e forward/inverse kinematics,
Jacobians, dynamics, and joint-space control — packaged from a set of lab
scaffolds into something installable, tested, and cross-validated against
the official [MuJoCo](https://mujoco.org/) model of the real robot.

![status](https://img.shields.io/badge/status-active--development-yellow)
![python](https://img.shields.io/badge/python-3.8%2B-blue)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-orange)
![license](https://img.shields.io/badge/license-MIT-green)


## Install

```bash
pip install -e .            # core (numpy only)
pip install -e ".[sim]"     # + mujoco, for simulation / cross-validation
pip install -e ".[test]"    # + pytest, to run tests/
```

## Quickstart

```python
import numpy as np
from ur5e_kinematics import UR5e, forward_kinematics, analytical_ik, numerical_ik

robot = UR5e()
q = np.array([0.2, -1.0, 1.3, -0.5, 0.9, 0.4])

_, T = forward_kinematics(robot, q)          # end-effector pose

sols = analytical_ik(robot, T)                # closed-form, up to 8 verified branches
q_sol, converged, iters = numerical_ik(robot, T, q0=np.zeros(6))  # DLS fallback
```

## What's in here

| Module | Contents |
|---|---|
| `robot.py` | `UR5e` — DH table + mass/COM/inertia/armature (real, from the official model) |
| `fk.py` | `forward_kinematics`, `end_effector_pose`, `dh_transform` |
| `jacobian.py` | `geometric_jacobian`, forward/inverse velocity kinematics |
| `ik.py` | `numerical_ik` (damped least squares) and `analytical_ik` (closed-form, geometric decoupling) |
| `dynamics.py` | `compute_M`, `compute_C_qdot`, `compute_G`, `forward_dynamics`, `simulate` |
| `controller.py` | `pd_control`, `computed_torque_control` |
| `trajectory.py` | `quintic_trajectory` for joint-space motion profiles |
| `mujoco_interface.py` | wrapper around `mujoco` for stepping/reading state, with a torque-mode fix (see below) |

## Analytical IK

`analytical_ik(robot, T_target)` implements the standard UR-series geometric
decoupling for an offset-wrist 6R arm: `theta1` from wrist-center geometry
(two shoulder branches), `theta5`/`theta6` from wrist orientation (two
wrist branches), `theta2`/`theta3` via the planar law of cosines (two elbow
branches), `theta4` as the remainder. All (up to 8) branch combinations are
computed, then each is checked with forward kinematics via
`verify_solution()` — only geometrically valid solutions are returned.

Validated three ways: round-trip against `numerical_ik` and ground-truth FK
across randomized configs (`tests/test_kinematics.py`), and directly against
MuJoCo's own FK (`tests/test_mujoco_crossvalidation.py`).

## Dynamics

`dynamics.py` was referenced by `controller.py`/`main.py` in the original
scaffold but wasn't included in the uploaded files, so it's new here. It
uses the per-link center-of-mass Jacobian formulation (Siciliano et al.,
*Robotics: Modelling, Planning and Control*, ch. 7) rather than recursive
Newton-Euler:

- `compute_M(robot, q)` — mass matrix, from Σ (mᵢ Jvᵢᵀ Jvᵢ + Jwᵢᵀ Iᵢ Jwᵢ),
  plus `diag(robot.armature)` for reflected motor/gearbox inertia
- `compute_G(robot, q)` — gravity vector, from Σ mᵢ g Jvᵢᵀ ẑ
- `compute_C_qdot(robot, q, qdot)` — Coriolis/centrifugal term, via
  Christoffel symbols computed by central finite differences of `M(q)`
- `forward_dynamics(robot, q, qdot, tau)` — `qddot = M⁻¹(tau - C·qdot - G)`
- `simulate(robot, q0, qdot0, tau_fn, dt, n_steps)` — fixed-step RK4
  integration under a torque policy `tau_fn(t, q, qdot) -> tau`

This is easier to verify than hand-rolled Newton-Euler, at the cost of being
slower per call — fine for simulation and offline analysis, not optimized
for a real-time control loop (`compute_C_qdot` alone calls `compute_M` 12
times per evaluation, for its central-difference Jacobian).

### `robot.inertia` / `robot.armature`: real values, sourced and cross-checked

`link_mass`, `com`, and `inertia` are sourced from the official
[MuJoCo Menagerie UR5e model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e)
(itself derived from the public UR5e URDF) — not hand-entered or guessed.
`com`/`inertia` needed a frame transform: MuJoCo's per-body inertial frames
don't share an origin/orientation convention with this file's DH frames, so
each link's COM and inertia tensor were converted by comparing this
module's own `forward_kinematics()` against MuJoCo's body poses at matched
joint angles, and confirming the transformed values came out **q-independent**
across multiple random configurations (a correctness check on the transform
itself, not just the source data — see git history / the extraction script
for the derivation if you need to redo this for a different model).
`robot.armature` (reflected motor+gearbox inertia, 0.1 kg·m² flat across all
6 joints) is likewise the model's real `dof_armature`, now added directly
into `compute_M`'s diagonal.

Net effect, confirmed against MuJoCo directly (`tests/test_mujoco_crossvalidation.py`):
- `compute_M` matches MuJoCo's `mj_fullM` to within ~0.05 across the whole
  matrix (previously off by **up to 1000×** on the wrist joints, when
  `inertia` was still a `zeros((6,6))` placeholder — link 6's COM sits on
  its own rotation axis, so a zero inertia tensor there made `M[5,5]`
  collapse to a near-singular ~1.3e-4, which made `forward_dynamics` explode
  to nonsense values — e.g. `-1565 rad/s²` vs. MuJoCo's ground-truth
  `-2.03 rad/s²` for the same torque/state).
- `compute_C_qdot(...) + compute_G(...)` matches MuJoCo's `qfrc_bias` to
  within ~0.1 N·m.
- `forward_dynamics` now matches MuJoCo's `qacc` closely under identical
  torque/state, including on the wrist joints.

**What's still not real:** `motor_inertia` and `gear_ratio` (kept as the
scaffold's original placeholders) don't uniquely decompose from
`armature = gear_ratio² × motor_inertia` (one equation, two unknowns) — use
`robot.armature` for anything that needs reflected inertia; don't trust
`motor_inertia`/`gear_ratio` individually. `friction_viscous`/`friction_coulomb`
are also still unverified placeholders and aren't used anywhere in
`dynamics.py` or `controller.py`.

## Closed-loop / forward-dynamics tests

`tests/test_dynamics_simulation.py` exercises `forward_dynamics` +
`simulate` together with the controllers — not just single-point M/C/G
evaluation:

- `test_forward_dynamics_self_consistent` — `M @ forward_dynamics(...)`
  reproduces `tau - C·qdot - G`
- `test_free_fall_conserves_energy_short_horizon` — no control torque, no
  friction: total mechanical energy is conserved to integrator error
- `test_closed_loop_computed_torque_tracks_trajectory` — the strongest
  self-consistency test: FK + Jacobians + M/C/G + `computed_torque_control`
  + RK4 integration all exercised together; converges to <1e-3 rad final
  tracking error (and to ~1e-10 in the pure self-consistency variant in
  `main`/ad hoc testing, since a model-based controller built from the same
  M/C/G it's tested against should cancel the dynamics almost exactly)
- `test_pd_control_alone_does_not_perfectly_track` — sanity check on the
  above: plain PD (no dynamics compensation) leaves real steady-state error,
  guarding against a trivial bug where `forward_dynamics` silently ignores
  `tau`
- `test_wrist_dynamics_matches_mujoco_order_of_magnitude` — regression
  guard for the fixed placeholder-inertia gap described above

Note the self-consistency tests above (built from this library's own M/C/G)
can't catch a *globally consistent* error — e.g. every sign flipped the same
way in both the "controller" and the "plant" would still pass. Only the
MuJoCo cross-validation is independent of this library's own dynamics code.

## MuJoCo

The official [MuJoCo Menagerie UR5e model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e)
(with mesh assets) ships in this repo under `models/ur5e/`, so cross-validation
and simulation work out of the box:

```python
from ur5e_kinematics.mujoco_interface import MuJoCoInterface

sim = MuJoCoInterface("models/ur5e/scene.xml")
sim.reset(q0=np.zeros(6))
q, qdot = sim.get_state()
sim.step(tau)   # tau = joint torques (N*m)
```

**Actuator gotcha (found and fixed during cross-validation):** the official
model's actuators are `general` actuators with `biastype="affine"` — built-in
PD *position* servos (gains ~2000/400 for the big joints, ~500/100 for the
wrist), not torque motors. Writing `tau` straight into `ctrl` (what an
earlier version of this file did) silently means "target angle", not
"torque". `MuJoCoInterface` defaults to `torque_mode=True`, which zeroes the
model's actuator gain/bias terms at load time and injects `tau` via
`data.qfrc_applied` instead, bypassing the built-in servos for genuine
open-loop torque control. Pass `torque_mode=False` if you want the model's
native position-servo behavior instead.

Check that the MJCF's actuated joint order matches `robot.dh`'s joint order
(shoulder_pan → wrist_3) before trusting the `q`/`qdot` mapping if you use a
different model — the official one matches directly (see below).

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -v

# MuJoCo cross-validation tests use the bundled models/ur5e/scene.xml by
# default; point MUJOCO_UR5E_SCENE elsewhere to test against a different copy.
pytest tests/test_mujoco_crossvalidation.py -v
```

- `test_kinematics.py` — analytical IK against FK ground truth, numerical IK
  convergence, Jacobian vs. finite-difference FK, forward/inverse velocity
  round-trip, mass matrix symmetry/PSD, Coriolis passivity identity
- `test_dynamics_simulation.py` — forward-dynamics self-consistency,
  energy conservation, closed-loop tracking (see above)
- `test_mujoco_crossvalidation.py` — the one set of tests that's independent
  of this library's own code: FK, mass matrix, and gravity/Coriolis bias
  force, all checked directly against the official MuJoCo model. **This is
  the test to run first if you distrust anything else in here** — everything
  else can pass while sharing the same bug (e.g. one globally flipped sign),
  since it's only checking self-consistency; this file can't.

## results

Generated by `scripts/record_results.py` + `scripts/plot_results.py` — see
the repo root README for what each check actually validates.

- `RESULTS.md` — human-readable summary
- `results.json` — same data, machine-readable
- `figures/` — PNGs
  - `joint_positions.png`, `joint_velocities.png` — free-fall + light-damping demo
  - `energy_balance.png` — KE and PE trade off cleanly, total mechanical energy decreases slowly
    (damping is dissipative) instead of drifting arbitrarily
  - `closed_loop_tracking.png` — computed-torque control (which uses this
    library's own FK + Jacobians + M/C/G) drives tracking error down ~10
    orders of magnitude; plain PD (no dynamics compensation) plateaus, as
    it should
  - `dynamics_analysis.png` — mass matrix heatmap, diagonal, eigenvalues,
    conditioning
  - `mujoco_crossvalidation.png` — **the strongest evidence, since it's the
    only comparison against an independent implementation** rather than
    this library checking itself: FK, mass matrix, and gravity/Coriolis
    bias force, all against the official MuJoCo Menagerie UR5e model

## Reproduce

```bash
pip install -e ".[sim,test]"
python scripts/record_results.py   # ~1-2 min (dynamics finite-differencing is the slow part)
python scripts/plot_results.py     # a few seconds
```


## Known placeholders (carried over, not silently fixed)

- `motor_inertia` / `gear_ratio` — individually unverified (see Dynamics
  section above); use `robot.armature` instead for reflected inertia.
- `friction_viscous` / `friction_coulomb` — not officially published by UR;
  current values are a starting point, not verified, and aren't wired into
  `dynamics.py` or `controller.py` at all yet.
- `robot.joint_limits` exists but isn't enforced anywhere (IK solutions,
  trajectories aren't filtered against it).
- No collision/self-collision checking.
- No forward/inverse dynamics validated against real hardware — this is a
  kinematics/control library cross-checked against a simulator, not a
  system-identified digital twin.

## License

MIT — see [LICENSE](LICENSE).
