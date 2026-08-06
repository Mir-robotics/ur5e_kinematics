"""
MuJoCo integration for simulating this library's controllers/trajectories
against a UR5e model.

Requires: pip install mujoco
Requires a UR5e MJCF (model_path). This library doesn't ship one -- use the
official model from MuJoCo Menagerie:
    https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e

IMPORTANT -- actuator type: the official Menagerie ur5e.xml actuators are
NOT torque motors. They're MuJoCo `general` actuators with
`biastype="affine"`, i.e. built-in PD position servos: writing `ctrl[i]`
sets a *target joint angle* (gains ~2000/400 for the big joints, ~500/100
for the wrist), not a torque. Passing this class's `step(tau)` output
straight into `ctrl` would silently apply the wrong thing.

To get genuine open-loop torque control (which is what computed_torque_control
etc. expect), `torque_mode=True` (default) zeroes out the model's actuator
gain/bias terms at load time and instead injects `tau` directly as a
generalized force via `data.qfrc_applied`, bypassing the built-in servos
entirely. Set `torque_mode=False` if you want the model's native
position-servo behavior instead (then `step()`'s argument is interpreted as
a target joint position passed through `ctrl`, not a torque).
"""
import numpy as np

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


class MuJoCoInterface:
    def __init__(self, model_path: str, n_joints: int = 6, torque_mode: bool = True):
        if not _HAS_MUJOCO:
            raise ImportError("mujoco is not installed. Run: pip install mujoco")
        self.model_path = model_path
        self.n = n_joints
        self.torque_mode = torque_mode
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nu < n_joints:
            raise ValueError(
                f"Model at {model_path} has {self.model.nu} actuators, "
                f"expected at least {n_joints}."
            )

        if self.torque_mode:
            # Neutralize the model's built-in position-servo actuators so
            # step(tau) behaves as pure open-loop torque control via
            # qfrc_applied instead of fighting a hidden PD controller.
            self.model.actuator_gainprm[:] = 0
            self.model.actuator_biasprm[:] = 0

    def reset(self, q0=None):
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            q0 = np.asarray(q0, dtype=float)
            self.data.qpos[:self.n] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self, tau):
        """
        Apply joint torques `tau` (n,), advance one simulation step.
        (If constructed with torque_mode=False, `tau` is instead treated as
        a target joint position for the model's native position servos.)
        """
        tau = np.asarray(tau, dtype=float)
        if self.torque_mode:
            self.data.qfrc_applied[:self.n] = tau
            self.data.ctrl[:self.n] = 0
        else:
            self.data.ctrl[:self.n] = tau
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        """Return (q, qdot), each shape (n,)."""
        q = self.data.qpos[:self.n].copy()
        qdot = self.data.qvel[:self.n].copy()
        return q, qdot

    def render(self):
        """
        Launches (or syncs) an interactive viewer. For headless/offscreen
        rendering instead, use mujoco.Renderer directly on self.model/self.data
        rather than this method.
        """
        if not hasattr(self, "_viewer"):
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer.sync()


if __name__ == "__main__":
    if not _HAS_MUJOCO:
        print("mujoco not installed; run `pip install mujoco` to try this demo.")
    else:
        print(
            "mujoco is installed. To run a real simulation, pass the path to "
            "a UR5e MJCF (e.g. from mujoco_menagerie) to MuJoCoInterface(...)."
        )
