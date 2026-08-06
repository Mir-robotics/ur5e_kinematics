# ur5e_kinematics -- validation results

Generated 2026-08-06T15:37:24.203160 by `scripts/record_results.py`.

## Kinematics

- FK: **949.3 us/call**
- Analytical IK: **100%** of 20 random configs recovered exactly (avg **7.4** verified branches, max position error **2.48e-16 m**)
- Numerical IK: **95%** convergence, avg **30.3** iterations

## Dynamics

- Mass matrix: symmetric = **True**, condition number = **27.1**, all eigenvalues positive = **True**
- Forward-dynamics self-consistency error: **7.33e-15**

- Energy balance (0.8s free-fall + light damping): total mechanical energy drift = **-4.200 J** (monotonic decrease expected; a large *increase* would indicate a sign error)

## Closed-loop control (RK4-simulated against the library's own dynamics)

- Computed-torque control (Kp=400, Kd=40): final tracking error = **7.71e-11 rad**
- Plain PD (Kp=100, Kd=20): final tracking error = **0.4715 rad** (nonzero as expected)

## MuJoCo cross-validation

Against the official [mujoco_menagerie UR5e model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e), 20 random configurations, **no calibration transform applied**:

| Quantity | Max error | Mean error |
|---|---|---|
| FK position | 1.360 mm | 1.004 mm |
| FK rotation | 2.55e-15 | 1.36e-15 |
| Mass matrix M(q) | 0.0025 | 0.0016 |
| Bias force C(q,qdot)qdot+G(q) | 0.0223 N*m | 0.0119 N*m |
