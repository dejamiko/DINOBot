import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from config import Config
from task_types import Task


def to_euler(angles):
    if len(angles) == 3:
        angles = Rotation.from_matrix(angles).as_quat(canonical=True)
    angles = p.getEulerFromQuaternion(angles)
    s = ""
    for a in angles:
        s += f"{a * 180 / np.pi} "
    return s


class SimEnv:
    """
    A class that sets up the simulation environment and provides functions to interact with it. It contains a table, an
    arm, and some objects that can be placed on the table.
    """

    def __init__(self, config):
        """
        Initialize the simulation environment.
        """
        self.config = config
        self.objects = {}
        # set seed
        np.random.seed(self.config.SEED)
        if self.config.USE_GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._setup_simulation_basic()
        self.move_arm_to_home_position()

    def _setup_simulation_basic(self):
        """
        Set up the simulation environment with a table and an arm.
        """
        p.resetSimulation()
        p.setGravity(0, 0, self.config.GRAVITY)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf")

        self.objects["arm"] = p.loadURDF(
            "franka_panda/panda.urdf", self.config.ARM_BASE_POSITION, useFixedBase=True
        )

        # use a table with no texture to make it easier for DINOBot to detect the objects
        self.objects["table"] = p.loadURDF(
            "/additional_urdfs/table/table.urdf",
            self.config.TABLE_BASE_POSITION,
            useFixedBase=False,
            globalScaling=2,
        )

        focus_position = p.getLinkState(self.objects["arm"], 11)[0]
        p.resetDebugVisualizerCamera(
            cameraDistance=self.config.DEBUG_CAMERA_VIEW[0],
            cameraYaw=self.config.DEBUG_CAMERA_VIEW[1],
            cameraPitch=self.config.DEBUG_CAMERA_VIEW[2],
            cameraTargetPosition=focus_position,
        )

        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        self.rest_poses = []
        for i in range(p.getNumJoints(self.objects["arm"])):
            joint_info = p.getJointInfo(self.objects["arm"], i)
            if joint_info[2] != p.JOINT_FIXED:
                self.lower_limits.append(joint_info[8])
                self.upper_limits.append(joint_info[9])
                self.joint_ranges.append(self.upper_limits[-1] - self.lower_limits[-1])
                self.rest_poses.append(joint_info[8] + joint_info[9] / 2)

    def move_arm_to_home_position(self):
        """
        Move the arm to the home position.
        """
        home_position = self.config.ARM_HOME_POSITION
        self.move_to_target_joint_position(home_position)

    def load_object(
        self, task_type, object_path, scale=1.0, offset=(0, 0, 0), rot=(0, 0, 0)
    ):
        """
        Load an object in the appropriate place depending on the task
        :param task_type: The type of the task for which to load the object
        :param object_path: The path to the object URDF file
        :param scale: The scale to be used for the object model
        :param offset: The offset to the initial object position
        :param rot: The rotation applied to the initial object rotation
        """
        # load the object on the table somewhere random (within a certain range)
        if task_type == Task.GRASPING.value or task_type == Task.PUSHING.value:
            self._load_object_on_the_table(object_path, offset, rot, scale)
        elif task_type == Task.HAMMERING.value:
            self._load_object_in_the_gripper(object_path, offset, rot, scale)
        else:
            raise ValueError(f"Unknown task type {task_type}")

    def get_rgbd_image(self):
        """
        Take a picture with the wrist camera for the purpose of the simulation.
        """
        width, height, projection_matrix, view_matrix, _, _, _ = self._get_camera_info()
        _, _, rgb_image, depth_buffer, _ = p.getCameraImage(
            width, height, view_matrix, projection_matrix
        )
        rgb_image = np.array(rgb_image).reshape(height, width, -1).astype(np.uint8)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB)

        depth_buffer = np.array(depth_buffer).reshape(height, width).astype(np.float32)

        return rgb_image, depth_buffer

    def reset(self):
        """
        Reset the simulation environment.
        """
        self._setup_simulation_basic()
        self.move_arm_to_home_position()

    def project_to_3d(self, points, depth):
        """
        Inputs: points: list of [x,y] pixel coordinates,
                depth (H,W,1) observations from camera.
                intrinsics: intrinsics of the camera, used to
                project pixels to 3D space.
        Outputs: point_with_depth: list of [x,y,z] coordinates in the camera frame.

        Projects the selected pixels to 3D space using intrinsics and
        depth value. Based on your setup the implementation may vary,
        but here you can find a simple example or the explicit formula:
        https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html.
        """
        width, height, _, _, fov, near, far = self._get_camera_info()
        z_n = 2.0 * depth - 1.0
        depth = 2.0 * near * far / (far + near - z_n * (far - near))
        fx, fy, cx, cy = self._get_intrinsics(fov, width, height)
        projected_points = []
        for u, v in points:
            z = depth[u, v]
            x = (cx - v) * z / fx
            y = (cy - u) * z / fy
            projected_points.append([x, y, z])
        projected_points = np.array(projected_points)
        return projected_points

    def move_in_camera_frame(self, t, rot):
        """
        Move the robot in the camera frame.
        :param t: The translation in the camera frame
        :param rot: The rotation in the camera frame
        """
        if self.config.VERBOSITY > 1:
            print("Moving in camera frame")
            print(
                f"Camera rotation {to_euler(self._get_camera_position_and_rotation()[1])}"
            )
            print(
                f"EEF rotation {to_euler(p.getLinkState(self.objects['arm'], 11)[1])}"
            )
        camera_position, camera_rotation = self._get_camera_position_and_rotation()
        camera_to_eef_translation, camera_to_eef_rotation = (
            self._get_eef_to_camera_transform()
        )
        eef_rotation = np.dot(np.linalg.inv(camera_to_eef_rotation), camera_rotation)
        # post-multiply row vector with transposed rotation matrix
        eef_position = camera_position - np.dot(
            camera_to_eef_translation, eef_rotation.T
        )
        eef_position_actual, eef_orientation_actual = p.getLinkState(
            self.objects["arm"], 11
        )[:2]
        assert np.allclose(
            eef_position, eef_position_actual
        ), f"{eef_position} != {eef_position_actual}"
        assert np.allclose(
            eef_rotation,
            np.array(p.getMatrixFromQuaternion(eef_orientation_actual)).reshape(-1, 3),
        ), f"{eef_rotation} != {np.array(p.getMatrixFromQuaternion(eef_orientation_actual)).reshape(-1, 3)}"
        # post-multiply row vector with transposed rotation matrix
        new_camera_position = camera_position + np.dot(t, camera_rotation.T)
        new_camera_rotation = np.dot(camera_rotation, rot)

        new_eef_rotation = np.dot(
            np.linalg.inv(camera_to_eef_rotation), new_camera_rotation
        )
        # post-multiply row vector with transposed rotation matrix
        new_eef_position = new_camera_position - np.dot(
            camera_to_eef_translation, new_eef_rotation.T
        )

        if self.config.VERBOSITY > 1:
            print(
                "Old camera position:",
                camera_position,
                "Old camera rotation:",
                to_euler(camera_rotation),
            )
            print(
                "New camera position:",
                new_camera_position,
                "New camera rotation:",
                to_euler(new_camera_rotation),
            )
            print(
                "Old eef position:",
                eef_position,
                "Old eef rotation:",
                to_euler(eef_rotation),
            )
            print(
                "New eef position:",
                new_eef_position,
                "New eef rotation:",
                to_euler(new_eef_rotation),
            )

        new_eef_rotation = Rotation.from_matrix(new_eef_rotation).as_quat(
            canonical=True
        )

        self._move_with_debug_dot(new_eef_position, new_eef_rotation)

    @staticmethod
    def disconnect():
        """
        Disconnect the simulation environment.
        """
        p.disconnect()

    def move_to_target_joint_position(self, target_joint_positions):
        """
        Move the arm to the target joint positions.
        :param target_joint_positions: the target joint positions
        """
        p.setJointMotorControlArray(
            self.objects["arm"],
            list(range(p.getNumJoints(self.objects["arm"]))),
            p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
        )
        error = np.inf
        step = 0
        while (
            error > self.config.MOVE_TO_TARGET_ERROR_THRESHOLD
            and step < self.config.MOVEMENT_ITERATION_MAX_STEPS
        ):
            current_joint_positions = np.array(
                [
                    p.getJointState(self.objects["arm"], i)[0]
                    for i in range(p.getNumJoints(self.objects["arm"]))
                ]
            )
            error = np.linalg.norm(current_joint_positions - target_joint_positions)
            p.stepSimulation()
            if self.config.USE_GUI:
                if self.config.TAKE_IMAGE_AT_EVERY_STEP:
                    live_img, depth_buffer = self.get_rgbd_image()
                    if self.config.VERBOSITY > 1:
                        self._draw_points_in_3d(live_img, depth_buffer)
                else:
                    time.sleep(1.0 / 240.0)
            step += 1
        if self.config.VERBOSITY > 1:
            if "points_debug_3d" in self.objects:
                p.removeUserDebugItem(self.objects["points_debug_3d"])
                self.objects.pop("points_debug_3d")

    def get_current_joint_positions(self):
        return [
            x[0]
            for x in p.getJointStates(
                self.objects["arm"], range(p.getNumJoints(self.objects["arm"]))
            )
        ]

    def _move_to_target_position_and_rotation(
        self, target_position, target_rotation=None
    ):
        """
        Move the arm to the target position.
        :param target_position: the position of the target in the world frame
        :param target_rotation: the rotation of the target in the world frame
        """
        if target_rotation is None:
            target_rotation = p.getLinkState(self.objects["arm"], 11)[1]
        target_joint_positions = p.calculateInverseKinematics(
            self.objects["arm"],
            11,
            target_position,
            target_rotation,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
        )

        if len(target_joint_positions) != p.getNumJoints(self.objects["arm"]):
            i, j = 0, 0
            target_joint_positions_full = [0] * p.getNumJoints(self.objects["arm"])
            while i < len(target_joint_positions):
                if self.lower_limits[i] < self.upper_limits[i]:
                    target_joint_positions_full[i] = target_joint_positions[j]
                    j += 1
                i += 1
            target_joint_positions = tuple(target_joint_positions_full)

        self.move_to_target_joint_position(target_joint_positions)

    def _load_object_on_the_table(self, object_path, offset, rot, scale):
        """
        Load the object on the table and move the arm so the camera can see it.
        :param object_path: The path to the object URDF file
        :param scale: The scale to be used for the object model
        :param offset: The offset to the initial object position
        :param rot: The rotation applied to the initial object rotation
        """
        x_base, y_base, z_base = self.config.OBJECT_X_Y_Z_BASE
        pos_x = (
            np.random.uniform(-0.05, 0.05) + x_base
            if self.config.RANDOM_OBJECT_POSITION
            else x_base
        )
        pos_y = (
            np.random.uniform(-0.05, 0.05) + y_base
            if self.config.RANDOM_OBJECT_POSITION
            else y_base
        )
        angle = (
            np.random.uniform(0, np.pi / 2) if self.config.RANDOM_OBJECT_ROTATION else 0
        )
        self.objects[f"object"] = p.loadURDF(
            object_path,
            offset + np.array((pos_x, pos_y, z_base)),
            p.getQuaternionFromEuler(rot + np.array((0, 0, angle))),
            globalScaling=scale,
        )
        # let the object drop
        for i in range(100):
            p.stepSimulation()
        self.object_initial_pos_and_rot = (
            np.array(p.getBasePositionAndOrientation(self.objects["object"])[0]),
            np.array(p.getBasePositionAndOrientation(self.objects["object"])[1]),
        )
        # move the arm so the camera can see the object
        eef_pos = p.getLinkState(self.objects["arm"], 11)[0]
        if self.config.RANDOM_OBJECT_POSITION_FOLLOWING:
            pos = [pos_x, pos_y, eef_pos[2]]
        else:
            pos = [x_base, y_base, eef_pos[2]]
        self._move_to_target_position_and_rotation(pos)

    def _load_object_in_the_gripper(self, object_path, offset, rot, scale):
        raise NotImplemented

    def _move_with_debug_dot(self, desired_pos, desired_rot):
        """
        Move the robot to the desired position and orientation, and add a red point at the desired position.
        :param desired_pos: The desired position
        :param desired_rot: The desired rotation
        """
        if self.config.VERBOSITY > 1:
            # add a red point at the desired position
            red_dot = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[1, 0, 0, 1],
                visualFramePosition=desired_pos,
            )
            self.objects["red_dot_id"] = p.createMultiBody(
                baseVisualShapeIndex=red_dot,
                baseMass=0,
                baseInertialFramePosition=desired_pos,
            )
            p.resetBasePositionAndOrientation(
                self.objects["red_dot_id"], desired_pos, desired_rot
            )
        # move the robot
        self._move_to_target_position_and_rotation(desired_pos, desired_rot)
        if self.config.VERBOSITY > 1:
            if "red_dot_id" in self.objects:
                p.removeBody(self.objects["red_dot_id"])
                self.objects.pop("red_dot_id")

    def _get_camera_info(self):
        """
        Get the camera information for the wrist camera.
        :return: the width, height, projection matrix, and view matrix of the camera
        """
        width = self.config.LOAD_SIZE
        height = self.config.LOAD_SIZE

        fov = self.config.CAMERA_FOV
        aspect = width / height
        near = self.config.CAMERA_NEAR_PLANE
        far = self.config.CAMERA_FAR_PLANE
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # find the camera position to be at the wrist of the robot
        camera_position, rot_matrix = self._get_camera_position_and_rotation()
        init_camera_vector = self.config.CAMERA_INIT_VECTOR
        init_up_vector = self.config.CAMERA_INIT_UP
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(
            camera_position, camera_position + 0.1 * camera_vector, up_vector
        )

        if self.config.VERBOSITY > 1:
            self._draw_debug_camera_axis(camera_position, rot_matrix)

        return width, height, projection_matrix, view_matrix, fov, near, far

    @staticmethod
    def _draw_debug_camera_axis(camera_position, rot_matrix):
        """
        Draw the camera axes for debugging purposes.
        :param camera_position: The position of the camera
        :param rot_matrix: The rotation matrix of the camera
        """
        # clear the previous camera axes
        p.removeAllUserDebugItems()
        # draw the camera x, y, z axes
        p.addUserDebugLine(
            camera_position,
            camera_position + 0.5 * rot_matrix[:, 0],
            [1, 0, 0],
            lineWidth=5,
        )
        p.addUserDebugLine(
            camera_position,
            camera_position + 0.5 * rot_matrix[:, 1],
            [0, 1, 0],
            lineWidth=5,
        )
        p.addUserDebugLine(
            camera_position,
            camera_position + 0.5 * rot_matrix[:, 2],
            [0, 0, 1],
            lineWidth=5,
        )

    def _get_camera_position_and_rotation(self):
        """
        Get the position and rotation of the camera.
        :return: the position and rotation of the camera
        """
        eef_position, eef_rotation = p.getLinkState(self.objects["arm"], 11)[:2]
        eef_rotation = np.array(p.getMatrixFromQuaternion(eef_rotation)).reshape(3, 3)
        eef_to_camera_translation, eef_to_camera_rotation = (
            self._get_eef_to_camera_transform()
        )
        # post-multiply row vector with transposed rotation matrix
        camera_position = np.array(eef_position) + np.dot(
            eef_to_camera_translation, eef_rotation.T
        )
        camera_rotation = np.dot(eef_to_camera_rotation, eef_rotation)

        return camera_position, camera_rotation

    @staticmethod
    def _get_intrinsics(fov, width, height):
        """
        Get the intrinsic parameters of the camera.
        Adapted from https://stackoverflow.com/questions/60430958/
        understanding-the-view-and-projection-matrix-from-pybullet/60450420#60450420
        """
        aspect = width / height
        fx = width / (2 * aspect * np.tan(np.radians(fov / 2)))
        fy = height / (2 * np.tan(np.radians(fov / 2)))
        cx = width / 2
        cy = height / 2
        return fx, fy, cx, cy

    def _get_eef_to_camera_transform(self):
        """
        Get the transformation matrix from camera frame to end-effector frame.
        """
        camera_to_eef_position = np.array(self.config.CAMERA_TO_EEF_TRANSLATION)
        camera_to_eef_rotation = np.array(
            p.getMatrixFromQuaternion(
                p.getQuaternionFromEuler(self.config.CAMERA_TO_EEF_ROTATION)
            )
        ).reshape(3, 3)
        return camera_to_eef_position, camera_to_eef_rotation

    def _project_points_to_world_frame(self, points):
        """
        Project points from the camera frame to the world frame.
        :param points: The points to project
        :return: The points in the world frame
        """
        camera_position, camera_rotation = self._get_camera_position_and_rotation()
        world_points = []
        for i in range(len(points)):
            point = camera_position + np.dot(camera_rotation, points[i])
            world_points.append(point)
        return world_points

    def _draw_points_in_3d(self, live_img, depth_buffer):
        """
        Draw projections of the points in 3D space.
        :param live_img: The live image from the camera
        :param depth_buffer: The depth buffer from the camera
        """
        all_pixels_in_camera_feed = []
        colours = []
        for x in range(0, self.config.LOAD_SIZE, 8):
            for y in range(0, self.config.LOAD_SIZE, 8):
                all_pixels_in_camera_feed.append([x, y])
                colours.append(live_img[x, y] / 255.0)
        points3d = self.project_to_3d(all_pixels_in_camera_feed, depth_buffer)
        world_points = self._project_points_to_world_frame(points3d)
        # draw the points in 3D using debug points
        if self.objects.get("points_debug_3d") is not None:
            p.removeUserDebugItem(self.objects["points_debug_3d"])
            self.objects.pop("points_debug_3d")
        self.objects["points_debug_3d"] = p.addUserDebugPoints(world_points, colours, 5)


if __name__ == "__main__":
    """
    A sample script to spawn some cubes on a table and move the robot arm towards one of them.
    """
    config = Config()
    env = SimEnv(config)
