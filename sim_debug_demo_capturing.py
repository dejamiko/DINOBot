import time

import numpy as np
import pybullet as p

from config import Config
from sim import ArmEnv


# TODO this doesn't record anything, just moves as requested with the debug parameters
class DemoSim(ArmEnv):
    def __init__(self, task_name):
        super(DemoSim, self).__init__(task_name)
        self.debug_parameters = self.create_debug_parameters()
        self.create_target_balls()

    def fast_move(self, position, orientation):
        """
        Move the arm to a specific position and orientation.
        :param position: The position to move to.
        :param orientation: The orientation to move to.
        """
        joint_positions = p.calculateInverseKinematics(
            self.objects["arm"], 11, position, orientation
        )
        p.setJointMotorControlArray(
            self.objects["arm"],
            [i for i in range(9)],
            p.POSITION_CONTROL,
            targetPositions=joint_positions,
        )

    def fast_move_to_home(self):
        """
        Move the arm to the home position.
        """
        p.setJointMotorControlArray(
            self.objects["arm"],
            list(range(p.getNumJoints(self.objects["arm"]))),
            p.POSITION_CONTROL,
            targetPositions=self.config.ARM_HOME_POSITION,
        )
        for _ in range(100):
            p.stepSimulation()

    def create_debug_parameters(self):
        """
        Create debug parameters for the simulation. Inspired by Matis's code.
        :return: The debug parameters dictionary.
        """
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        current_x, current_y, current_z = pos
        current_roll, current_pitch, current_yaw = p.getEulerFromQuaternion(rot)

        params_dictionary = {
            "x": {
                "min": -0.5 + current_x,
                "max": 0.5 + current_x,
                "default": current_x,
                "previous_value": current_x,
            },
            "placeholder": {"min": -1, "max": 1, "default": 0},
            "y": {
                "min": -0.5 + current_y,
                "max": 0.5 + current_y,
                "default": current_y,
                "previous_value": current_y,
            },
            "z": {
                "min": -0.5 + current_z,
                "max": 0.5 + current_z,
                "default": current_z,
                "previous_value": current_z,
            },
            "roll": {
                "min": -np.pi,
                "max": np.pi,
                "default": current_roll,
                "previous_value": current_roll,
            },
            "pitch": {
                "min": -np.pi,
                "max": np.pi,
                "default": current_pitch,
                "previous_value": current_pitch,
            },
            "yaw": {
                "min": -np.pi,
                "max": np.pi,
                "default": current_yaw,
                "previous_value": current_yaw,
            },
            "home_button": {"min": 1, "max": 0, "default": 0},
        }
        for param in params_dictionary.keys():
            param_id = p.addUserDebugParameter(
                param,
                params_dictionary[param]["min"],
                params_dictionary[param]["max"],
                params_dictionary[param]["default"],
            )
            params_dictionary[param]["id"] = param_id
        return params_dictionary

    def create_target_balls(self):
        """
        Create debug visualisation of a target ball.
        """
        if "orange_dot" in self.objects:
            p.removeBody(self.objects["orange_dot"])
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        orange_dot = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0.5, 0, 1],
            visualFramePosition=pos,
        )

        # Initialize lists for additional spheres (links) and their positions
        additional_balls = []
        link_positions = []

        for i in range(-4, 5):
            for axis in range(3):
                offset = [0, 0, 0]
                offset[axis] = 0.1 * i
                additional_balls.append(
                    p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=0.01,
                        rgbaColor=[1, 0.5, 0, 1],
                        visualFramePosition=np.array(pos) + np.array(offset),
                    )
                )
                link_positions.append(offset)

        # Create the multi body with the central sphere as the base and additional spheres as links
        self.objects["orange_dot"] = p.createMultiBody(
            baseVisualShapeIndex=orange_dot,
            baseInertialFramePosition=pos,
            baseInertialFrameOrientation=rot,
            baseMass=0,
            linkMasses=[0] * len(additional_balls),  # Assuming the links are massless
            linkCollisionShapeIndices=[-1]
            * len(additional_balls),  # No collision shapes for the links
            linkVisualShapeIndices=additional_balls,
            linkPositions=link_positions,
            linkOrientations=[(0, 0, 0, 1)]
            * len(additional_balls),  # No orientation needed for the links
            linkInertialFramePositions=[(0, 0, 0)] * len(additional_balls),
            linkInertialFrameOrientations=[(0, 0, 0, 1)] * len(additional_balls),
            linkParentIndices=[0]
            * len(additional_balls),  # All links are children of the base
            linkJointTypes=[p.JOINT_FIXED]
            * len(additional_balls),  # Fixed joints to keep the shape rigid
            linkJointAxis=[(0, 0, 0)]
            * len(additional_balls),  # No axis needed for fixed joints
        )

    def move_target_balls(self, position, rotation):
        """
        Move the target balls to a specific position and orientation.
        :param position: The position to move to.
        :param rotation: The orientation to move to.
        :return: The target balls moved to the specified position and orientation.
        """
        p.resetBasePositionAndOrientation(
            self.objects["orange_dot"], position, rotation
        )

    def update_parameters(self):
        """
        Update the user provided debug parameters of the simulation.
        """
        try:
            if (
                p.readUserDebugParameter(self.debug_parameters["home_button"]["id"])
                == 1
            ):
                self.fast_move_to_home()
                p.removeAllUserParameters()
                self.debug_parameters = self.create_debug_parameters()
                self.create_target_balls()
                return
        except p.error:
            pass

        value_changed = False
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        current_roll, current_pitch, current_yaw = p.getEulerFromQuaternion(rot)
        current_values = {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "roll": current_roll,
            "pitch": current_pitch,
            "yaw": current_yaw,
        }

        for param in self.debug_parameters.keys():
            if param in ["placeholder", "home_button"]:
                continue
            try:
                value = p.readUserDebugParameter(self.debug_parameters[param]["id"])
            except p.error:
                value = self.debug_parameters[param]["previous_value"]

            self.debug_parameters[param]["previous_value"] = value

            current_value = current_values[param]
            if not np.allclose(value, current_value, atol=1e-3):
                value_changed = True

        if value_changed:
            position = [
                self.debug_parameters["x"]["previous_value"],
                self.debug_parameters["y"]["previous_value"],
                self.debug_parameters["z"]["previous_value"],
            ]
            rotation = p.getQuaternionFromEuler(
                [
                    self.debug_parameters["roll"]["previous_value"],
                    self.debug_parameters["pitch"]["previous_value"],
                    self.debug_parameters["yaw"]["previous_value"],
                ]
            )

            self.move_target_balls(position, rotation)

            self.fast_move(position, rotation)


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 0
    sim = DemoSim(config)
    while True:
        sim.update_parameters()
        p.stepSimulation()
        time.sleep(1.0 / 2400.0)
