import os

import numpy as np
import pybullet as p
from pybullet_object_models import ycb_objects

from config import Config
from sim_env import SimEnv
from task_types import Task


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
def move_and_rotate_test(env):
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
def load_normal_pybullet_objects_test(env):
    env.load_object(Task.GRASPING.value, "objects/mug.urdf")


@reset
@move_to_home
def load_pybullet_object_models_test(env):
    obj_name = "YcbBanana"
    path_to_urdf = os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf")
    env.load_object(Task.GRASPING.value, path_to_urdf)


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 2
    environment = SimEnv(config)

    move_and_rotate_test(environment)
    load_normal_pybullet_objects_test(environment)
    load_pybullet_object_models_test(environment)
