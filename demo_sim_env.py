import json
import os
import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from config import Config
from sim_env import SimEnv
from task_types import Task


class DemoSimEnv(SimEnv):
    def __init__(
        self,
        config,
        task_type,
        object_path,
        scale=1.0,
        offset=(0, 0, 0),
        rot=(0, 0, 0),
        adj_rot=(0, 0, 0),
        nail_path=None,
    ):
        super(DemoSimEnv, self).__init__(config)
        self.object_info = (object_path, scale, offset, rot, adj_rot, nail_path)
        self.task = task_type

        self.gripper_open = False
        self.recording = False
        self.recording_start_pos_and_rot = None
        self.recorded_data = []
        self.recently_triggered = 10

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
            ord("n"): lambda: self._open_gripper(),  # open gripper
            ord("m"): lambda: self._close_gripper(),  # close gripper
        }
        self.other_key_mappings = {
            ord("h"): lambda: self._fast_move_to_home(),  # move to home position
            ord("q"): lambda: self._start_recording(),  # start recording
            ord("e"): lambda: self._stop_recording(),  # stop recording
            ord("r"): lambda: self._replay_last_demo(),  # replay the latest demo
            ord("t"): lambda: self._reset_from_keyboard(),  # reset
            ord("."): lambda: self._store_state_keyboard(),  # store the current state
            ord(","): lambda: self._load_last_state(),  # load the last state
        }

        self.load_object(task_type, *self.object_info)

        self._create_debug_dot()

    def control_arm_with_keyboard(self):
        """
        Control the arm in the simulation using the keyboard.
        """
        keys = p.getKeyboardEvents()
        # I will ignore the flags for now
        keys = list(keys.keys())
        self._control_movement_with_keys(keys)
        self._control_flow_with_keys(keys)
        self._update_debug_dot()

    def replay_demo(self, demo):
        """
        Replay a demonstration using relative positions.
        :param demo: The demonstration to replay.
        :return: the success of the replayed demonstration.
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
            new_rot = rot_delta @ rot
            new_pos = pos + np.dot(pos_delta, rot.T)
            new_rot = Rotation.from_matrix(new_rot).as_quat(canonical=True)

            if self.config.VERBOSITY > 1:
                print(f"Replay, move to {new_pos}, {new_rot}")

            self._fast_move(new_pos, new_rot)
            if gripper_open:
                self._open_gripper()
            else:
                self._close_gripper()
            p.stepSimulation()
            self._update_debug_dot()
            if self.config.USE_GUI:
                time.sleep(1.0 / 2400.0)
            success = success or self._evaluate_success()

        p.removeUserDebugItem(self.objects["replaying_text"])
        self.objects.pop("replaying_text")
        return success

    def reset(self, task_type=None):
        """
        Reset the simulation.
        :param task_type: Optionally provided type of task to perform.
        """
        super(DemoSimEnv, self).reset()
        self.gripper_open = False
        self.recording = False
        self.recorded_data = []
        if task_type is not None:
            self.task = task_type
        self.load_object(self.task, *self.object_info)
        self._create_debug_dot()

    @staticmethod
    def load_demonstration(filename):
        """
        Load a demonstration from a file.
        :param filename: The filename of the demonstration.
        :return: The demonstration as a dictionary {"images": images, "depth_bn": depth buffers,
        "demo_positions": the offset positions}
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
            "images": images,
            "depth_buffers": depth_buffers,
            "demo_positions": demonstration["recorded_data"],
        }
        return data

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

        self._set_joint_positions_and_velocities(positions, velocities)
        self.pause()

        self.task = data["task"]
        self.object_initial_pos_and_rot = (
            np.array(p.getBasePositionAndOrientation(self.objects["object"])[0]),
            # TODO add the rotation from DB to make sure the axes are correct
            np.array(p.getBasePositionAndOrientation(self.objects["object"])[1]),
        )

    def store_state(self, base_object=None, target_object=None):
        # store a JSON file in BASE_DIR/transfers which allows for loading the env and quick replay
        data = {
            "object": p.getBasePositionAndOrientation(self.objects["object"]),
            "arm": p.getBasePositionAndOrientation(self.objects["arm"]),
            "joints": p.getJointStates(
                self.objects["arm"], range(p.getNumJoints(self.objects["arm"]))
            ),
            "seed": self.config.SEED,
            "task": self.task,
        }
        i = 0
        while True:
            if base_object is not None:
                filename = os.path.join(
                    self.config.BASE_DIR,
                    "transfers",
                    self.task,
                    f"transfer_{base_object}_{target_object}_{str(i).zfill(3)}.json",
                )
            else:
                filename = os.path.join(
                    self.config.BASE_DIR,
                    "transfers",
                    self.task,
                    f"transfer_{str(i).zfill(3)}.json",
                )
            if not os.path.exists(filename):
                break
            i += 1
        with open(filename, "w") as f:
            json.dump(data, f)

    @staticmethod
    def pause():
        for _ in range(10000):
            p.stepSimulation()

    def _control_flow_with_keys(self, keys):
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
                    (np.dot(np.linalg.inv(f_rot), np.array(pos) - f_pos)).tolist(),
                    (np.dot(np.linalg.inv(f_rot), rot)).tolist(),
                    self.gripper_open,
                )
            )
        if self.recently_triggered > 0:
            self.recently_triggered -= 1
        if not self.gripper_open:
            self._close_gripper()

    def _control_movement_with_keys(self, keys):
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
            self._move_in_xyz(*translation)
        if np.any(rotation):
            self._move_in_rpy(*rotation)
        if not self.gripper_open:
            self._close_gripper()

    def _fast_move(self, position, rotation):
        """
        Move the arm to a specific position and rotation.
        :param position: The position to move to.
        :param rotation: The rotation to move to.
        """
        joint_positions = p.calculateInverseKinematics(
            self.objects["arm"], 11, position, rotation
        )
        self._set_joint_positions_and_velocities(joint_positions)

    def _set_joint_positions_and_velocities(
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

    def _fast_move_to_home(self):
        """
        Move the arm to the home position.
        """
        self._set_joint_positions_and_velocities(self.config.ARM_HOME_POSITION)
        for _ in range(100):
            p.stepSimulation()

    def _move_in_xyz(self, x, y, z):
        """
        Move the arm in the simulation in the x, y, z directions.
        :param x: The translation in the x direction.
        :param y: The translation in the y direction.
        :param z: The translation in the z direction.
        """
        pos, rot = p.getLinkState(self.objects["arm"], 11)[:2]
        current_x, current_y, current_z = pos
        new_x, new_y, new_z = current_x + x, current_y + y, current_z + z
        self._fast_move((new_x, new_y, new_z), rot)

    def _move_in_rpy(self, roll, pitch, yaw):
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
        self._fast_move(pos, new_rot)

    def _close_gripper(self):
        """
        Close the gripper in the simulation.
        """
        p.setJointMotorControl2(
            self.objects["arm"], 9, p.POSITION_CONTROL, targetPosition=0.0, force=1000
        )
        p.setJointMotorControl2(
            self.objects["arm"], 10, p.POSITION_CONTROL, targetPosition=0.0, force=1000
        )
        self.gripper_open = False

    def _open_gripper(self):
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

    def _start_recording(self):
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
        images, depth_buffers = self._take_demo_images()
        self.recorded_data.append(len(images))
        for img, depth in zip(images, depth_buffers):
            self.recorded_data.append((img, depth))

    def _stop_recording(self):
        """
        Stop recording the simulation and save the recorded data to a file.
        """
        if not self.recording:
            return
        self.recording = False
        p.removeUserDebugItem(self.objects["recording_text"])
        self.objects.pop("recording_text")
        # find an unused filename
        directory = f"demonstrations/{self.task}/"
        i = 0
        while True:
            filename = f"{directory}demonstration_{str(i).zfill(2)}.json"
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

    def _replay_last_demo(self):
        """
        Replay the last recorded demonstration.
        """
        if self.recently_triggered > 0:
            return

        # find the last recorded demo by inspecting the creation times
        directory = f"demonstrations/{self.task}/"
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
        data = self.load_demonstration(latest_path)["demo_positions"]
        self.replay_demo(data)

    def _reset_from_keyboard(self):
        if self.recently_triggered > 0:
            return
        self.recently_triggered = 10
        self.reset()

    def _determine_grasp_success(self):
        """
        Determine if the grasp was successful.
        :return: True if the grasp was successful, False otherwise.
        """
        # check if the object is higher than the table
        pos, _ = p.getBasePositionAndOrientation(self.objects["object"])
        if pos[2] < self.config.OBJECT_X_Y_Z_BASE[2] + self.config.GRASP_SUCCESS_HEIGHT:
            return False
        return True

    def _determine_push_success(self):
        """
        See if the object is moved in a specific direction by some distance
        :return: True if the push was successful, false otherwise
        """
        pos, rot = p.getBasePositionAndOrientation(self.objects["object"])
        init_pos, init_rot = self.object_initial_pos_and_rot

        translation = pos - init_pos

        translation_local = np.dot(
            translation, np.array(p.getMatrixFromQuaternion(init_rot)).reshape(3, -1)
        )

        total_dist = np.linalg.norm(translation_local)

        if total_dist > 0:
            angle_from_positive_x = np.arccos(
                np.clip(translation_local[0] / total_dist, -1.0, 1.0)
            )
            angle_from_negative_x = np.arccos(
                np.clip(-translation_local[0] / total_dist, -1.0, 1.0)
            )
        else:
            angle_from_positive_x = np.pi / 2
            angle_from_negative_x = np.pi / 2

        return total_dist > self.config.PUSH_SUCCESS_DIST and (
            angle_from_positive_x <= self.config.PUSH_SUCCESS_ANGLE
            or angle_from_negative_x <= self.config.PUSH_SUCCESS_ANGLE
        )

    def _determine_hammering_success(self):
        # see if the nail was struck
        contact_points = p.getContactPoints(
            self.objects["object"], self.objects["nail"]
        )

        if contact_points is None or len(contact_points) == 0:
            return False

        def is_pointing_downwards(vector):
            z_vector = (0, 0, 1)
            alignment = np.dot(z_vector, vector)

            return alignment > self.config.HAMMERING_SUCCESS_ALIGNMENT

        for point in contact_points:
            if point[9] > self.config.HAMMERING_SUCCESS_FORCE and is_pointing_downwards(
                point[7]
            ):
                return True

        return False

    def _evaluate_success(self):
        if self.task == Task.GRASPING.value:
            return self._determine_grasp_success()
        elif self.task == Task.PUSHING.value:
            return self._determine_push_success()
        else:
            return self._determine_hammering_success()

    def _take_demo_images(self):
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

    def _store_state_keyboard(self):
        if self.recently_triggered > 0:
            return
        self.recently_triggered = 10
        self.store_state()

    def _load_last_state(self):
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

    def _update_debug_dot(self):
        suc = self._evaluate_success()
        if suc and self.dot_is_red:
            self.dot_is_red = False
            p.changeVisualShape(
                self.objects["dot"],
                -1,
                rgbaColor=[0, 1, 0, 1],
            )
        elif not suc and not self.dot_is_red:
            self.dot_is_red = True
            p.changeVisualShape(
                self.objects["dot"],
                -1,
                rgbaColor=[1, 0, 0, 1],
            )

    def _create_debug_dot(self):
        dot = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 1],
            visualFramePosition=(0, 1, 2),
        )
        self.objects["dot"] = p.createMultiBody(
            baseVisualShapeIndex=dot,
            baseMass=0,
            baseInertialFramePosition=(0, 1, 2),
        )
        self.dot_is_red = True


def record_demo_with_keyboard():
    while True:
        sim.control_arm_with_keyboard()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    config = Config()
    config.RANDOM_OBJECT_ROTATION = False
    config.RANDOM_OBJECT_POSITION_FOLLOWING = True
    # db = create_and_populate_db(config)
    config.VERBOSITY = 1

    # obj_name = "YcbHammer"
    # object_path = os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf")
    i = 157
    object_path = f"random_urdfs/{str(i).zfill(3)}/{str(i).zfill(3)}.urdf"

    sim = DemoSimEnv(
        config,
        Task.HAMMERING.value,
        object_path,
        offset=(0, 0, 0),
        rot=(0, 0, 6 * np.pi / 4),
        adj_rot=(0, 0, np.pi),
        scale=1.2,
    )

    # rotation which simplifies the demonstration

    record_demo_with_keyboard()

    # data = sim.load_demonstration(db.get_demo_for_object("banana"))
    # sim.replay_demo(data["demo_positions"])
