import os

import numpy as np
import pybullet as p
from pybullet_object_models import ycb_objects

from config import Config
from sim import ArmEnv


def move_to_home(func):
    def wrapper(*args, **kwargs):
        env = args[0]
        env.move_arm_to_home_position()
        func(*args, **kwargs)
        env.move_arm_to_home_position()

    return wrapper


def reset(func):
    def wrapper(*args, **kwargs):
        env = args[0]
        func(*args, **kwargs)
        env.reset()

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


@reset
@move_to_home
def test_load_normal_pybullet_objects(env):
    env.load_object("objects/mug.urdf")


@reset
@move_to_home
def test_load_pybullet_object_models(env):
    obj_name = "YcbBanana"
    path_to_urdf = os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf")
    env.load_object(path_to_urdf)
    # TODO look through and play around with those objects to find a set of candidates for DINOBot


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 1
    environment = ArmEnv(config)

    test_move_and_rotate(environment)
    test_load_normal_pybullet_objects(environment)
    test_load_pybullet_object_models(environment)
