import json
import os
import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from config import Config
from database import create_and_populate_db
from sim_env import SimEnv


class DemoSimEnv(SimEnv):
    def __init__(self, config, object_path, scale=1.0, offset=(0, 0, 0), rot=(0, 0, 0)):
        super(DemoSimEnv, self).__init__(config)
        self.gripper_open = False
        self.object_info = (object_path, scale, offset, rot)
        self.recording = False
        self.recording_start_pos_and_rot = None
        self.recorded_data = []
        self.recently_triggered = 10
        self.load_object(*self.object_info)
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
            ord("t"): lambda: self.reset_keyboard(),  # reset
            ord("."): lambda: self.store_state_keyboard(),
            ord(","): lambda: self.load_last_state(),
        }

    def fast_move(self, position, rotation):
        """
        Move the arm to a specific position and rotation.
        :param position: The position to move to.
        :param rotation: The rotation to move to.
        """
        joint_positions = p.calculateInverseKinematics(
            self.objects["arm"], 11, position, rotation
        )
        self.set_joint_positions_and_velocities(joint_positions)

    def set_joint_positions_and_velocities(
        self, joint_positions, joint_velocities=None
    ):
        if joint_velocities is not None:
            p.setJointMotorControlArray(
                self.objects["arm"],
                list(range(len(joint_positions))),
                p.POSITION_CONTROL,
                targetPositions=joint_positions,
                targetVelocities=joint_velocities,
            )
        else:
            p.setJointMotorControlArray(
                self.objects["arm"],
                list(range(len(joint_positions))),
                p.POSITION_CONTROL,
                targetPositions=joint_positions,
            )

    def fast_move_to_home(self):
        """
        Move the arm to the home position.
        """
        self.set_joint_positions_and_velocities(self.config.ARM_HOME_POSITION)
        for _ in range(100):
            p.stepSimulation()

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
        new_roll, new_pitch, new_yaw = (
            current_roll + roll,
            current_pitch + pitch,
            current_yaw + yaw,
        )
        new_rot = p.getQuaternionFromEuler([new_roll, new_pitch, new_yaw])
        self.fast_move(pos, new_rot)

    def close_gripper(self):
        """
        Close the gripper in the simulation.
        """
        # TODO Try out the velocity control and lower the force
        p.setJointMotorControl2(
            self.objects["arm"], 9, p.POSITION_CONTROL, targetPosition=0.0, force=1000
        )
        p.setJointMotorControl2(
            self.objects["arm"], 10, p.POSITION_CONTROL, targetPosition=0.0, force=1000
        )
        self.gripper_open = False

    def open_gripper(self):
        """
        Open the gripper in the simulation.
        """
        if self.gripper_open:
            return
        p.setJointMotorControl2(
            self.objects["arm"], 9, p.POSITION_CONTROL, targetPosition=0.04
        )
        p.setJointMotorControl2(
            self.objects["arm"], 10, p.POSITION_CONTROL, targetPosition=0.04
        )
        self.gripper_open = True

    def start_recording(self):
        """
        Start recording the simulation.
        """
        if self.recording:
            return
        self.recording = True
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        self.recording_start_pos_and_rot = np.array(pos), np.array(
            p.getMatrixFromQuaternion(rot)
        ).reshape(3, 3)
        # display "Recording" on the screen
        self.objects["recording_text"] = p.addUserDebugText(
            "Recording", [0, 0, 2], textColorRGB=[1, 0, 0], textSize=2
        )
        self.recorded_data = []
        images, depth_buffers = self.take_demo_images()
        self.recorded_data.append(len(images))
        for img, depth in zip(images, depth_buffers):
            self.recorded_data.append((img, depth))

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
            # the first frame contains the number of images taken
            num_images = self.recorded_data.pop(0)
            images = []
            depth_buffers = []
            # we then get all the images
            for i in range(num_images):
                img, depth_buffer = self.recorded_data.pop(0)
                images.append(img.tolist())
                depth_buffers.append(depth_buffer.tolist())
            # the rest of the recorded data is movement in every frame
            json.dump(
                {
                    "recorded_data": self.recorded_data,
                    "images": images,
                    "depth_buffers": depth_buffers,
                },
                f,
            )
        if self.config.VERBOSITY > 1:
            print(
                f"Saved demonstration with {len(self.recorded_data)} frames to {filename}"
            )
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
            pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
            rot = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3)

            f_pos, f_rot = self.recording_start_pos_and_rot

            self.recorded_data.append(
                (
                    (np.array(pos) - f_pos).tolist(),
                    (rot @ np.linalg.inv(f_rot)).tolist(),
                    self.gripper_open,
                )
            )
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
                elif key in [
                    ord("i"),
                    ord("k"),
                    ord("j"),
                    ord("l"),
                    ord("u"),
                    ord("o"),
                ]:
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

        # find the last recorded demo by inspecting the creation times
        directory = "demonstrations/"
        latest_time = 0
        latest_path = None
        for f in os.listdir(directory):
            if not f.endswith(".json"):
                continue
            if os.path.getmtime(directory + f) > latest_time:
                latest_time = os.path.getmtime(directory + f)
                latest_path = directory + f

        if self.config.VERBOSITY > 1:
            print(f"Replaying demo {latest_path}")
        data = self.load_demonstration(latest_path)["demo_velocities"]
        self.replay_demo(data)

    def replay_demo(self, demo):
        """
        Replay a demonstration from a list of keystrokes.
        :param demo: The demonstration to replay.
        """

        self.recently_triggered = 10
        # create text on the screen to show that the demo is playing
        self.objects["replaying_text"] = p.addUserDebugText(
            "Replaying demo", [0, 0, 2], textColorRGB=[1, 0, 0], textSize=2
        )

        success = False
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        pos = np.array(pos)
        rot = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
        for pos_delta, rot_delta, gripper_open in demo:
            # calculate the actual position and rotation from the current one
            new_pos = pos + pos_delta
            new_rot = rot_delta @ rot
            new_rot = Rotation.from_matrix(new_rot).as_quat(canonical=True)

            if self.config.VERBOSITY > 1:
                print(f"Replay, move to {new_pos}, {new_rot}")

            self.fast_move(new_pos, new_rot)
            if gripper_open:
                self.open_gripper()
            else:
                self.close_gripper()
            p.stepSimulation()
            if self.config.USE_GUI:
                time.sleep(1.0 / 2400.0)
            success = success or self.determine_grasp_success()

        p.removeUserDebugItem(self.objects["replaying_text"])
        self.objects.pop("replaying_text")
        return success

    def reset_keyboard(self):
        if self.recently_triggered > 0:
            return
        self.recently_triggered = 10
        self.reset()

    def reset(self):
        """
        Reset the simulation.
        """
        super(DemoSimEnv, self).reset()
        self.gripper_open = False
        self.recording = False
        self.recorded_data = []
        self.load_object(*self.object_info)

    @staticmethod
    def load_demonstration(filename):
        """
        Load a demonstration from a file.
        :param filename: The filename of the demonstration.
        :return: The demonstration as a dictionary.
        """
        with open(filename, "r") as file:
            demonstration = json.load(file)
        images = demonstration["images"]
        depth_buffers = demonstration["depth_buffers"]
        for i in range(len(images)):
            img = np.array(images[i], dtype=np.uint8).reshape(-1, 3)
            images[i] = img.reshape(int(np.sqrt(img.shape[0])), -1, 3)
            depth_buffers[i] = np.array(depth_buffers[i])
        data = {
            "rgb_bn": images,
            "depth_bn": depth_buffers,
            "demo_velocities": demonstration["recorded_data"],
        }
        return data

    def determine_grasp_success(self):
        """
        Determine if the grasp was successful.
        :return: True if the grasp was successful, False otherwise.
        """
        # check if the object is higher than the table
        pos, _ = p.getBasePositionAndOrientation(self.objects["object"])
        if pos[2] < self.config.OBJECT_X_Y_Z_BASE[2] + 0.2:
            return False
        return True

    def take_demo_images(self):
        images, depth_buffers = [], []
        img, depth = self.get_rgbd_image()
        images.append(img)
        depth_buffers.append(depth)
        # move to 4 different viewpoints by moving an angle on a sphere in the positive and negative x and y axes
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        x, y, z = pos
        roll, pitch, yaw = p.getEulerFromQuaternion(rot)
        r = depth[len(depth) // 2][
            len(depth) // 2
        ]  # the distance to the object (approximately)
        new_z = z - r * (1.0 - np.cos(self.config.DEMO_ADDITIONAL_IMAGE_ANGLE))
        offset = r * np.sin(self.config.DEMO_ADDITIONAL_IMAGE_ANGLE)

        # go in the negative x direction
        new_pos = (x - offset, y, new_z)
        new_rot = p.getQuaternionFromEuler(
            (roll, pitch - self.config.DEMO_ADDITIONAL_IMAGE_ANGLE, yaw)
        )
        self._move_with_debug_dot(new_pos, new_rot)
        img, depth = self.get_rgbd_image()
        images.append(img)
        depth_buffers.append(depth)

        # go in the positive x direction
        new_pos = (x + offset, y, new_z)
        new_rot = p.getQuaternionFromEuler(
            (roll, pitch + self.config.DEMO_ADDITIONAL_IMAGE_ANGLE, yaw)
        )
        self._move_with_debug_dot(new_pos, new_rot)
        img, depth = self.get_rgbd_image()
        images.append(img)
        depth_buffers.append(depth)

        # go in the negative y direction
        new_pos = (x, y - offset, new_z)
        new_rot = p.getQuaternionFromEuler(
            (roll + self.config.DEMO_ADDITIONAL_IMAGE_ANGLE, pitch, yaw)
        )
        self._move_with_debug_dot(new_pos, new_rot)
        img, depth = self.get_rgbd_image()
        images.append(img)
        depth_buffers.append(depth)

        # go in the positive y direction
        new_pos = (x, y + offset, new_z)
        new_rot = p.getQuaternionFromEuler(
            (roll - self.config.DEMO_ADDITIONAL_IMAGE_ANGLE, pitch, yaw)
        )
        self._move_with_debug_dot(new_pos, new_rot)
        img, depth = self.get_rgbd_image()
        images.append(img)
        depth_buffers.append(depth)

        # return to the initial position
        self._move_with_debug_dot(pos, rot)

        return images, depth_buffers

    def store_state_keyboard(self):
        if self.recently_triggered > 0:
            return
        self.recently_triggered = 10
        self.store_state()

    def store_state(self, base_object=None, target_object=None):
        # store a JSON file in BASE_DIR/transfers which allows for loading the env and quick replay
        data = {
            "object": p.getBasePositionAndOrientation(self.objects["object"]),
            "arm": p.getBasePositionAndOrientation(self.objects["arm"]),
            "joints": p.getJointStates(
                self.objects["arm"], range(p.getNumJoints(self.objects["arm"]))
            ),
            "seed": self.config.SEED,
        }
        i = 0
        while True:
            if base_object is not None:
                filename = os.path.join(
                    self.config.BASE_DIR,
                    "transfers",
                    f"transfer_{base_object}_{target_object}_{str(i).zfill(3)}.json",
                )
            else:
                filename = os.path.join(
                    self.config.BASE_DIR,
                    "transfers",
                    f"transfer_{str(i).zfill(3)}.json",
                )
            if not os.path.exists(filename):
                break
            i += 1
        with open(filename, "w") as f:
            json.dump(data, f)

    def load_last_state(self):
        directory = os.path.join(self.config.BASE_DIR, "transfers/")
        latest_time = 0
        latest_path = None
        for f in os.listdir(directory):
            if not f.endswith(".json"):
                continue
            if os.path.getmtime(directory + f) > latest_time:
                latest_time = os.path.getmtime(directory + f)
                latest_path = directory + f
        self.load_state(latest_path)

    def load_state(self, state):
        self.reset()
        with open(state) as f:
            data = json.load(f)
        self.config.SEED = data["seed"]
        p.resetBasePositionAndOrientation(self.objects["object"], *data["object"])
        p.resetBasePositionAndOrientation(self.objects["arm"], *data["arm"])
        positions, velocities = [], []
        for i, j in enumerate(data["joints"]):
            positions.append(j[0])
            velocities.append(j[1])

        self.set_joint_positions_and_velocities(positions, velocities)
        self.pause()

    @staticmethod
    def pause():
        for _ in range(10000):
            p.stepSimulation()


def record_demo():
    while True:
        sim.control_arm()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    config = Config()
    config.RANDOM_OBJECT_ROTATION = False
    config.RANDOM_OBJECT_POSITION_FOLLOWING = True
    config.VERBOSITY = 0
    db = create_and_populate_db(config)
    config.VERBOSITY = 1
    target_object = "tomato_soup_can"
    sim = DemoSimEnv(
        config,
        db.get_urdf_path(target_object),
        db.get_urdf_scale(target_object),
        (0, 0, 0),
        (0, 0, 0),
    )

    record_demo()

    # data = sim.load_demonstration("demonstrations/demonstration_mini_cheetah.json")
    # sim.replay_demo(data["demo_velocities"])
