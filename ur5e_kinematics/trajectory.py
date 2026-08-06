"""Lab 3 support — joint-space trajectory generation for dynamics testing."""
import numpy as np


def quintic_coeffs(q0, qf, T):
    a0, a1, a2 = q0, 0.0, 0.0
    a3 = 10*(qf - q0)/T**3
    a4 = -15*(qf - q0)/T**4
    a5 = 6*(qf - q0)/T**5
    return a0, a1, a2, a3, a4, a5


def quintic_trajectory(q0, qf, T, t):
    """q0, qf: (n,) arrays. Returns q, qdot, qddot at time t (0<=t<=T)."""
    q0, qf = np.array(q0), np.array(qf)
    n = len(q0)
    q, qd, qdd = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        a0, a1, a2, a3, a4, a5 = quintic_coeffs(q0[i], qf[i], T)
        q[i] = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        qd[i] = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        qdd[i] = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return q, qd, qdd
