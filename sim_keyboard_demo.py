import json
import os
import time

import numpy as np
import pybullet as p

from config import Config
from sim import ArmEnv


class DemoSim(ArmEnv):
    def __init__(self, task_name):
        super(DemoSim, self).__init__(task_name)
        self.index = 0
        self.gripper_open = False
        self.recording = False
        self.recorded_data = []
        self.recently_triggered = 10
        self.load_object()
        self.create_target_balls()
        self.movement_key_mappings = {
            ord("s"): lambda: (0.01, 0, 0),  # positive x
            ord("x"): lambda: (-0.01, 0, 0),  # negative x
            ord("z"): lambda: (0, 0.01, 0),  # positive y
            ord("c"): lambda: (0, -0.01, 0),  # negative y
            ord("a"): lambda: (0, 0, 0.01),  # positive z
            ord("d"): lambda: (0, 0, -0.01),  # negative z
            ord("i"): lambda: (0.02, 0, 0),  # positive roll
            ord("k"): lambda: (-0.02, 0, 0),  # negative roll
            ord("j"): lambda: (0, 0.02, 0),  # positive pitch
            ord("l"): lambda: (0, -0.02, 0),  # negative pitch
            ord("u"): lambda: (0, 0, 0.02),  # positive yaw
            ord("o"): lambda: (0, 0, -0.02),  # negative yaw
            ord("n"): lambda: self.open_gripper(),  # open gripper
            ord("m"): lambda: self.close_gripper(),  # close gripper
        }
        self.other_key_mappings = {
            ord("h"): lambda: self.fast_move_to_home(),  # move to home position
            ord("q"): lambda: self.start_recording(),  # start recording
            ord("e"): lambda: self.stop_recording(),  # stop recording
            ord("r"): lambda: self.replay_last_demo(),  # replay demo
            ord("t"): lambda: self.reset(),  # reset
        }

    def fast_move(self, position, orientation):
        """
        Move the arm to a specific position and orientation.
        :param position: The position to move to.
        :param orientation: The orientation to move to.
        """
        joint_positions = p.calculateInverseKinematics(
            self.objects["arm"],
            11,
            position,
            orientation
        )
        p.setJointMotorControlArray(
            self.objects["arm"],
            [i for i in range(9)],
            p.POSITION_CONTROL,
            targetPositions=joint_positions
        )
        self.move_target_balls(position, orientation)

    def fast_move_to_home(self):
        """
        Move the arm to the home position.
        """
        p.setJointMotorControlArray(
            self.objects["arm"],
            list(range(p.getNumJoints(self.objects["arm"]))),
            p.POSITION_CONTROL,
            targetPositions=self.config.ARM_HOME_POSITION
        )
        for _ in range(100):
            p.stepSimulation()

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
            linkCollisionShapeIndices=[-1] * len(additional_balls),  # No collision shapes for the links
            linkVisualShapeIndices=additional_balls,
            linkPositions=link_positions,
            linkOrientations=[(0, 0, 0, 1)] * len(additional_balls),  # No orientation needed for the links
            linkInertialFramePositions=[(0, 0, 0)] * len(additional_balls),
            linkInertialFrameOrientations=[(0, 0, 0, 1)] * len(additional_balls),
            linkParentIndices=[0] * len(additional_balls),  # All links are children of the base
            linkJointTypes=[p.JOINT_FIXED] * len(additional_balls),  # Fixed joints to keep the shape rigid
            linkJointAxis=[(0, 0, 0)] * len(additional_balls)  # No axis needed for fixed joints
        )

    def move_target_balls(self, position, rotation):
        """
        Move the target balls to a specific position and orientation.
        :param position: The position to move to.
        :param rotation: The orientation to move to.
        :return: The target balls moved to the specified position and orientation.
        """
        p.resetBasePositionAndOrientation(
            self.objects["orange_dot"],
            position,
            rotation
        )

    def move_in_xyz(self, x, y, z):
        """
        Move the arm in the simulation in the x, y, z directions.
        :param x: The translation in the x direction.
        :param y: The translation in the y direction.
        :param z: The translation in the z direction.
        """
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        current_x, current_y, current_z = pos
        new_x, new_y, new_z = current_x + x, current_y + y, current_z + z
        self.fast_move((new_x, new_y, new_z), rot)

    def move_in_rpy(self, roll, pitch, yaw):
        """
        Move the arm in the simulation in the roll, pitch, yaw directions.
        :param roll: The rotation in the roll direction.
        :param pitch: The rotation in the pitch direction.
        :param yaw: The rotation in the yaw direction.
        """
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        current_roll, current_pitch, current_yaw = p.getEulerFromQuaternion(rot)
        new_roll, new_pitch, new_yaw = current_roll + roll, current_pitch + pitch, current_yaw + yaw
        new_rot = p.getQuaternionFromEuler([new_roll, new_pitch, new_yaw])
        self.fast_move(pos, new_rot)

    def close_gripper(self):
        """
        Close the gripper in the simulation.
        """
        # make sure this closes with a lot of force to ensure the object is grasped
        p.setJointMotorControl2(
            self.objects["arm"],
            9,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=1000
        )
        p.setJointMotorControl2(
            self.objects["arm"],
            10,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=1000
        )
        self.gripper_open = False

    def open_gripper(self):
        """
        Open the gripper in the simulation.
        """
        if self.gripper_open:
            return
        p.setJointMotorControl2(
            self.objects["arm"],
            9,
            p.POSITION_CONTROL,
            targetPosition=0.04
        )
        p.setJointMotorControl2(
            self.objects["arm"],
            10,
            p.POSITION_CONTROL,
            targetPosition=0.04
        )
        self.gripper_open = True

    def start_recording(self):
        """
        Start recording the simulation.
        """
        if self.recording:
            return
        self.recording = True
        self.index = 0
        # display "Recording" on the screen
        self.objects["recording_text"] = p.addUserDebugText(
            "Recording",
            [0, 0, 2],
            textColorRGB=[1, 0, 0],
            textSize=2
        )
        self.recorded_data = []

    def stop_recording(self):
        """
        Stop recording the simulation and save the recorded data to a file.
        """
        if not self.recording:
            return
        self.recording = False
        p.removeUserDebugItem(self.objects["recording_text"])
        self.objects.pop("recording_text")
        # find an unused filename
        directory = "demonstrations/"
        i = 0
        while True:
            filename = f"{directory}demonstration_{str(i).zfill(3)}.json"
            if not os.path.exists(filename):
                break
            i += 1
        with open(filename, "w") as f:
            json.dump({"recorded_data": self.recorded_data}, f)
        if self.config.VERBOSITY > 0:
            print(f"Saved demonstration with {len(self.recorded_data)} frames to {filename}")
        self.recorded_data = []

    def control_arm(self):
        """
        Control the arm in the simulation using the keyboard.
        """
        keys = p.getKeyboardEvents()
        # I will ignore the flags for now
        keys = list(keys.keys())
        self.control_movement_with_keys(keys)
        self.control_other_with_keys(keys)

    def control_other_with_keys(self, keys):
        """
        Control the simulation with keys that are not movement keys.
        :param keys: The keys pressed.
        """
        for key in keys:
            if key in self.other_key_mappings:
                self.other_key_mappings[key]()
        if self.recording:
            self.index += 1
            if len(keys) > 0:
                self.recorded_data.append((self.index, keys))
        if self.recently_triggered > 0:
            self.recently_triggered -= 1
        if not self.gripper_open:
            self.close_gripper()

    def control_movement_with_keys(self, keys):
        """
        Control the movement of the arm in the simulation with keys.
        :param keys: The keys pressed.
        :return: The arm moved in the simulation.
        """
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.array([0.0, 0.0, 0.0])
        for key in keys:
            if key in self.movement_key_mappings:
                if key in [ord("s"), ord("x"), ord("z"), ord("c"), ord("a"), ord("d")]:
                    translation += np.array(self.movement_key_mappings[key]())
                elif key in [ord("i"), ord("k"), ord("j"), ord("l"), ord("u"), ord("o")]:
                    rotation += np.array(self.movement_key_mappings[key]())
                else:
                    self.movement_key_mappings[key]()
        if np.any(translation):
            self.move_in_xyz(*translation)
        if np.any(rotation):
            self.move_in_rpy(*rotation)
        if not self.gripper_open:
            self.close_gripper()

    def replay_last_demo(self):
        """
        Replay the last recorded demonstration.
        """
        if self.recently_triggered > 0:
            return

        # find the last recorded demo by finding the highest number
        directory = "demonstrations/"
        i = 0
        while True:
            filename = f"{directory}demonstration_{str(i).zfill(3)}.json"
            if not os.path.exists(filename):
                break
            i += 1
        i -= 1
        if i < 0:
            raise FileNotFoundError("No recorded demos found.")
        demo = f"{directory}demonstration_{str(i).zfill(3)}.json"
        if self.config.VERBOSITY > 0:
            print(f"Replaying demo {demo}")
        self.replay_demo(demo)

    def replay_demo(self, demo):
        """
        Replay a demonstration from a file.
        :param demo: The file to replay.
        """
        self.reset()
        # create text on the screen to show that the demo is playing
        self.objects["replaying_text"] = p.addUserDebugText(
            "Replaying demo",
            [0, 0, 2],
            textColorRGB=[1, 0, 0],
            textSize=2
        )
        with open(demo, "r") as f:
            data = json.load(f)["recorded_data"]
        current_index = 0
        for index, keys in data:
            if self.config.VERBOSITY > 0:
                print(f"Replaying index {index}, with keys {keys}")
            while current_index < int(index):
                if self.config.VERBOSITY > 0:
                    print(f"Skipping index {current_index}")
                p.stepSimulation()
                time.sleep(1. / 120.)
                current_index += 1
            keys = [int(k) for k in keys]
            keys = [k for k in keys if k in self.movement_key_mappings]
            self.control_movement_with_keys(keys)
            p.stepSimulation()
            time.sleep(1. / 120.)
            current_index += 1
        p.removeUserDebugItem(self.objects["replaying_text"])
        self.objects.pop("replaying_text")

    def reset(self):
        """
        Reset the simulation.
        """
        if self.recently_triggered > 0:
            return
        super(DemoSim, self).reset()
        self.gripper_open = False
        self.recording = False
        self.recorded_data = []
        self.recently_triggered = 10
        self.load_object()
        self.create_target_balls()


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 0
    sim = DemoSim(config)
    while True:
        sim.control_arm()
        p.stepSimulation()
        time.sleep(1. / 120.)
