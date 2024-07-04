import numpy as np
import pybullet as p

from config import Config
from sim import ArmEnv


def move_to_home(func):
    def wrapper(*args, **kwargs):
        env = args[0]
        env.move_arm_to_home_position()
        func(*args, **kwargs)
        env.move_arm_to_home_position()

    return wrapper


@move_to_home
def test_move_and_rotate(env):
    env.move_in_camera_frame(np.array([-0.3, 0, 0]), np.eye(3))
    env.move_in_camera_frame(np.array([0.3, 0, 0]), np.eye(3))
    env.move_in_camera_frame(
        np.array([-0.3, 0, 0]),
        np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        ).reshape(3, 3),
    )
    env.move_in_camera_frame(
        np.array([0.3, 0, 0]),
        np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        ).reshape(3, 3),
    )


@move_to_home
def test_load_object(env):
    env.load_object("objects/mug.urdf")


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 0
    environment = ArmEnv(config, False)

    # test_move_and_rotate(environment)
    test_load_object(environment)
