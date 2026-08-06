"""
UR5e — pure data container. No kinematics/dynamics computation here.
Convention: Standard Denavit-Hartenberg (a, alpha, d, theta_offset)
"""
import numpy as np


class UR5e:
    def __init__(self):
        self.n = 6

        # [a (m), alpha (rad), d (m), theta_offset (rad)]
        # Official UR5e kinematic parameters (verify against your lecture DH table)
        self.dh = np.array([
            [0.0,      np.pi/2,   0.1625,   0.0],
            [-0.425,   0.0,       0.0,      0.0],
            [-0.3922,  0.0,       0.0,      0.0],
            [0.0,      np.pi/2,   0.1333,   0.0],
            [0.0,     -np.pi/2,   0.0997,   0.0],
            [0.0,      0.0,       0.0996,   0.0],
        ])

        self.joint_limits = np.array([[-2*np.pi, 2*np.pi]] * 6)

        # ------------------------------------------------------------------
        # DYNAMIC PARAMETERS — sourced from the official MuJoCo Menagerie
        # UR5e model (google-deepmind/mujoco_menagerie, universal_robots_ur5e/
        # ur5e.xml), which the Menagerie README states is derived from the
        # publicly available UR5e URDF. mass/com/inertia are the real
        # manufacturer-sourced values (previously zeros/placeholders here).
        # com and inertia were transformed from the MJCF's body-local /
        # principal-axis frames into this file's DH-frame convention by
        # comparing this module's own forward_kinematics() against MuJoCo's
        # body poses at matched joint angles (see dynamics cross-validation
        # in tests/) — off-diagonal inertia terms and small (<1mm) COM
        # components came out ~0 after that transform, consistent with the
        # links' physical symmetry about their DH axes.
        # ------------------------------------------------------------------
        self.link_mass = np.array([3.7, 8.393, 2.275, 1.219, 1.219, 0.1889])

        # COM of each link, expressed in that link's own DH frame (m)
        self.com = np.array([
            [0.0,    0.0,    0.0],
            [0.2121, 0.0,    0.138],
            [0.1963, 0.0,    0.007],
            [0.0,    0.0,    0.0],
            [0.0,    0.0,    0.0],
            [0.0,    0.0,   -0.0216],
        ])

        # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about COM, in link's own DH frame (kg*m^2)
        self.inertia = np.array([
            [0.010267, 0.00666,   0.010267, 0.0, 0.0, 0.0],
            [0.015107, 0.133886,  0.133886, 0.0, 0.0, 0.0],
            [0.004095, 0.03118,   0.03118,  0.0, 0.0, 0.0],
            [0.00256,  0.00256,   0.002194, 0.0, 0.0, 0.0],
            [0.00256,  0.002194,  0.00256,  0.0, 0.0, 0.0],
            [9.90e-05, 9.90e-05,  1.32e-04, 0.0, 0.0, 0.0],
        ])

        # Reflected motor+gearbox inertia at each joint. MuJoCo's UR5e model
        # uses a single lumped "armature" term (its own name for this exact
        # quantity) rather than separate motor_inertia/gear_ratio, and its
        # value is real (from the official model, not a placeholder) —
        # 0.1 kg*m^2 flat across all 6 joints.
        self.armature = np.array([0.1] * 6)

        # motor_inertia / gear_ratio below are the scaffold's original
        # placeholders. armature = gear_ratio**2 * motor_inertia has one
        # equation and two unknowns, so it doesn't uniquely determine these
        # two from the armature value above — keep using self.armature
        # (which dynamics.compute_M now adds directly) for anything that
        # needs reflected inertia; treat these two as unverified.
        self.motor_inertia = np.array([0.0021, 0.0021, 0.0021, 0.0005, 0.0005, 0.0005])
        self.gear_ratio = np.array([101, 101, 101, 101, 101, 101])

        # NOT officially published by UR — use system ID or literature, then justify in report
        self.friction_viscous = np.array([0.1] * 6)
        self.friction_coulomb = np.array([0.5] * 6)

    # ----------------------------------------------------------------------
    # Import hook — fill this in once you have real numbers (URDF, datasheet,
    # or your own system identification), so the rest of the package never
    # needs to change.
    # ----------------------------------------------------------------------
    def load_dynamics_from_dict(self, data: dict):
        """
        data keys (all optional, only overwrites what's given):
          'link_mass': (6,), 'com': (6,3), 'inertia': (6,6), 'armature': (6,),
          'motor_inertia': (6,), 'gear_ratio': (6,),
          'friction_viscous': (6,), 'friction_coulomb': (6,)
        Use this after parsing ur_description URDF or a datasheet/CSV.
        """
        for key, val in data.items():
            arr = np.array(val, dtype=float)
            if not hasattr(self, key):
                raise KeyError(f"Unknown field: {key}")
            expected_shape = getattr(self, key).shape
            if arr.shape != expected_shape:
                raise ValueError(f"{key}: expected shape {expected_shape}, got {arr.shape}")
            setattr(self, key, arr)

    def inertia_matrix(self, link_idx):
        """3x3 inertia tensor about link i's COM, in that link's own frame,
        built from the [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] row in self.inertia."""
        ixx, iyy, izz, ixy, ixz, iyz = self.inertia[link_idx]
        return np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ])

    def sanity_check(self):
        assert self.dh.shape == (self.n, 4)
        assert self.link_mass.shape == (self.n,)
        assert self.com.shape == (self.n, 3)
        assert self.inertia.shape == (self.n, 6)
        assert self.armature.shape == (self.n,)
        assert np.all(self.link_mass > 0), "link masses must be positive"
        print("robot.py: shapes OK. mass/com/inertia/armature sourced from mujoco_menagerie's "
              "UR5e MJCF; motor_inertia/gear_ratio/friction are still unverified placeholders.")


if __name__ == "__main__":
    r = UR5e()
    r.sanity_check()
