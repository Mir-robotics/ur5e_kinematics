"""
Cross-validation against the official MuJoCo Menagerie UR5e model:
    https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e

This is the strongest check in the whole test suite: everything else only
verifies internal self-consistency (our FK agrees with our Jacobian agrees
with our IK, etc.) — a globally-flipped sign convention would sail through
all of it. This file checks our numbers against an independent
implementation of the same physical robot.

Skipped automatically if `mujoco` isn't installed or the model isn't found.
The official UR5e model (with mesh assets, from mujoco_menagerie) ships in
this repo under models/ur5e/ -- point MUJOCO_UR5E_SCENE elsewhere if you
want to test against a different copy.
"""
import os
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ur5e_kinematics import UR5e
from ur5e_kinematics.fk import forward_kinematics
from ur5e_kinematics.dynamics import compute_M, compute_G

_MODEL_PATH = os.environ.get(
    "MUJOCO_UR5E_SCENE",
    os.path.join(os.path.dirname(__file__), "..", "models", "ur5e", "scene.xml"),
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(_MODEL_PATH),
    reason=f"UR5e MJCF not found at {_MODEL_PATH} (see README for how to fetch it)",
)


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


@pytest.fixture
def robot():
    return UR5e()


def _set_and_forward(mj, q):
    model, data = mj
    data.qpos[:6] = q
    data.qvel[:6] = 0
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize("seed", range(8))
def test_ee_pose_matches_mujoco(robot, mj, seed):
    """Our forward_kinematics vs. MuJoCo's own FK, at the same q.
    Validates DH table + joint order + joint sign conventions end to end."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, 6)

    _set_and_forward(mj, q)
    model, data = mj
    site_id = model.site("attachment_site").id
    p_mj = data.site_xpos[site_id].copy()
    R_mj = data.site_xmat[site_id].reshape(3, 3).copy()

    _, T = forward_kinematics(robot, q)

    assert np.allclose(T[:3, 3], p_mj, atol=2e-3)   # ~1mm-scale DH rounding
    assert np.allclose(T[:3, :3], R_mj, atol=1e-6)  # orientation should match tightly


@pytest.mark.parametrize("seed", range(5))
def test_mass_matrix_matches_mujoco(robot, mj, seed):
    """compute_M vs. MuJoCo's mj_fullM, including armature/reflected inertia
    on both sides (compute_M always adds robot.armature -- see dynamics.py
    -- since that's what real forward dynamics needs; MuJoCo's own M
    includes its dof_armature by default too, so this is apples-to-apples
    without needing to zero anything out)."""
    model, data = mj
    rng = np.random.default_rng(100 + seed)
    q = rng.uniform(-np.pi, np.pi, 6)
    _set_and_forward(mj, q)

    M_mj = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M_mj)
    M_ours = compute_M(robot, q)

    assert np.max(np.abs(M_mj - M_ours)) < 0.05


@pytest.mark.parametrize("seed", range(5))
def test_gravity_matches_mujoco(robot, mj, seed):
    """compute_G vs. MuJoCo's qfrc_bias at qvel=0 (bias force = C@qdot + G;
    at qdot=0 that's just G)."""
    model, data = mj
    rng = np.random.default_rng(200 + seed)
    q = rng.uniform(-np.pi, np.pi, 6)
    _set_and_forward(mj, q)

    G_mj = data.qfrc_bias[:6].copy()
    G_ours = compute_G(robot, q)
    assert np.allclose(G_mj, G_ours, atol=0.05)  # N*m
