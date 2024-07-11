import numpy as np
import pybullet as p
import pytest

from config import Config
from sim import ArmEnv


def test_sim_can_be_created():
    config = Config()
    sim = ArmEnv(config)
    sim.disconnect()


def test_sim_can_be_created_no_gui():
    config = Config()
    config.USE_GUI = False
    sim = ArmEnv(config)
    sim.disconnect()


@pytest.fixture
def sim_fixture():
    config = Config()
    config.USE_GUI = False
    sim = ArmEnv(config)
    yield sim, config
    sim.disconnect()


def test_move_arm_to_home_position(sim_fixture):
    sim, config = sim_fixture
    sim.move_arm_to_home_position()
    assert np.allclose([x[0] for x in p.getJointStates(sim.objects["arm"], range(p.getNumJoints(sim.objects["arm"])))],
                       config.ARM_HOME_POSITION, atol=config.MOVE_TO_TARGET_ERROR_THRESHOLD)


def test_load_object_loads_object(sim_fixture):
    sim, config = sim_fixture
    config.RANDOM_OBJECT_POSITION = False
    config.RANDOM_OBJECT_ROTATION = False
    sim.load_object()
    assert "object" in sim.objects
    pos, rot = p.getBasePositionAndOrientation(sim.objects["object"])
    x, y, z = config.OBJECT_X_Y_Z_BASE
    assert np.allclose(pos[0], x, atol=0.01)
    assert np.allclose(pos[1], y, atol=0.01)
    assert pos[2] <= z
    assert np.allclose(p.getEulerFromQuaternion(rot), (0, 0, 0), atol=1e-3)


def test_load_object_with_random_position(sim_fixture):
    sim, config = sim_fixture
    config.RANDOM_OBJECT_POSITION = True
    config.RANDOM_OBJECT_ROTATION = False
    sim.load_object()
    assert "object" in sim.objects
    pos, rot = p.getBasePositionAndOrientation(sim.objects["object"])
    x, y, z = config.OBJECT_X_Y_Z_BASE
    assert np.allclose(pos[0], x, atol=0.1)
    assert np.allclose(pos[1], y, atol=0.1)
    assert pos[2] <= z
    assert np.allclose(p.getEulerFromQuaternion(rot), (0, 0, 0), atol=1e-3)


def test_load_object_with_random_rotation(sim_fixture):
    sim, config = sim_fixture
    config.RANDOM_OBJECT_ROTATION = True
    sim.load_object()
    assert "object" in sim.objects
    pos, rot = p.getBasePositionAndOrientation(sim.objects["object"])
    x, y, z = config.OBJECT_X_Y_Z_BASE
    # The rotation introduces more noise
    assert np.allclose(pos[0], x, atol=0.05)
    assert np.allclose(pos[1], y, atol=0.05)
    assert pos[2] <= z
    assert np.allclose(p.getEulerFromQuaternion(rot)[:2], (0, 0), atol=1e-3)
