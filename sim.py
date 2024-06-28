"""
This file contains the code related to the simulation environment. It can be used within the DINOBot framework
or for small independent experiments with the pybullet simulation environment.
"""

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from config import Config
from environment import Environment


class ArmEnv(Environment):
    """
    A class that sets up the simulation environment and provides functions to interact with it. It contains a table,
    an arm, and some objects that can be placed on the table.
    """

    def __init__(self, config):
        """
        Initialize the simulation environment.
        """
        super().__init__(config)
        self.objects = {}
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._setup_simulation_basic()

    def _setup_simulation_basic(self):
        """
        Set up the simulation environment with a table and an arm.
        """
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf", globalScaling=100)

        self.objects["arm"] = p.loadURDF(
            "franka_panda/panda.urdf", [1, 1, 0], useFixedBase=True
        )

        focus_position, _ = p.getBasePositionAndOrientation(self.objects["arm"])
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=focus_position,
        )

    def load_object(self, x=None, y=None, object_path="jenga/jenga.urdf"):
        """
        Load an object on the table and move the arm so the camera can see it.
        :param x: the x-coordinate of the object. If None, a random x-coordinate is chosen.
        :param y: the y-coordinate of the object. If None, a random y-coordinate is chosen.
        :param object_path: the path to the object URDF file
        """
        # load some object on the table somewhere random (within a certain range)
        pos_x = x if x is not None else np.random.uniform(1.6, 1.8)
        pos_y = y if y is not None else np.random.uniform(0.9, 1.1)
        angle = np.random.uniform(0, 2 * np.pi)
        self.objects[f"object"] = p.loadURDF(
            object_path, [pos_x, pos_y, 0.5], p.getQuaternionFromEuler([0, 0, angle])
        )
        colour = np.random.uniform(0, 1, 3)
        p.changeVisualShape(
            self.objects[f"object"], -1, rgbaColor=colour.tolist() + [1]
        )

        # let the object drop
        for i in range(100):
            p.stepSimulation()

        # move the arm a little bit so the camera can see the object
        if x is not None and y is not None:
            pos = [x, y, 0.5]
        else:
            pos = [1.7, 1, 0.5]
        while not self.is_target_reached(pos):
            self.step_to_target(pos)

    def get_target_position(self, name="object"):
        """
        Get the position of the target object.
        :param name: The name of the object
        :return: the position of the object
        """
        return p.getBasePositionAndOrientation(self.objects[name])[0]

    def move_to_target(self, target_position, target_orientation=None):
        """
        Move the arm to the target position.
        :param target_position: the position of the target in the world frame
        :param target_orientation: the orientation of the target in the world frame
        """
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = []
        for i in range(p.getNumJoints(self.objects["arm"])):
            joint_info = p.getJointInfo(self.objects["arm"], i)
            if joint_info[2] != p.JOINT_FIXED:
                lower_limits.append(joint_info[8])
                upper_limits.append(joint_info[9])
                joint_ranges.append(upper_limits[-1] - lower_limits[-1])
                rest_poses.append(joint_info[8] + joint_info[9] / 2)

        if target_orientation is not None:
            target_joint_positions = p.calculateInverseKinematics(
                self.objects["arm"],
                11,
                target_position,
                target_orientation,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
            )
        else:
            target_joint_positions = p.calculateInverseKinematics(
                self.objects["arm"],
                11,
                target_position,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
            )

        p.setJointMotorControlArray(
            self.objects["arm"],
            range(9),
            p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
        )

        error, steps = np.inf, 0
        if self.config.VERBOSITY > 0:
            points_debug_3d = -1
        while error > 0.01 and steps < 100:
            current_pos, current_rot = p.getLinkState(self.objects["arm"], 11)[:2]
            error = np.linalg.norm(
                np.array(target_position) - np.array(current_pos)
            ) + np.linalg.norm(np.array(target_orientation) - np.array(current_rot))

            p.stepSimulation()
            live_img, depth_buffer = self.get_rgbd_image()
            if self.config.VERBOSITY > 0:
                points_debug_3d = self.draw_points_in_3d(
                    live_img, depth_buffer, points_debug_3d
                )
            steps += 1
        if self.config.VERBOSITY > 0:
            p.removeUserDebugItem(points_debug_3d)

    def step_to_target(self, target_position, target_orientation=None):
        """
        Move the arm to the target position by performing one step.
        :param target_orientation: the orientation of the target
        :param target_position: the position of the target
        """
        if target_orientation is not None:
            target_joint_positions = p.calculateInverseKinematics(
                self.objects["arm"], 11, target_position, target_orientation
            )
        else:
            target_joint_positions = p.calculateInverseKinematics(
                self.objects["arm"], 11, target_position
            )

        p.setJointMotorControlArray(
            self.objects["arm"],
            range(9),
            p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
        )
        p.stepSimulation()
        self.get_rgbd_image()

    def get_rgbd_image(self):
        """
        Take a picture with the wrist camera for the purpose of the simulation.
        """
        width, height, projection_matrix, view_matrix, _, _, _ = self.get_camera_info()
        _, _, rgb_image, depth_buffer, _ = p.getCameraImage(
            width, height, view_matrix, projection_matrix
        )
        rgb_image = np.array(rgb_image).reshape(height, width, -1).astype(np.uint8)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB)

        depth_buffer = np.array(depth_buffer).reshape(height, width).astype(np.float32)

        return rgb_image, depth_buffer

    def is_target_reached(self, target_position, target_orientation=None):
        """
        Check if the end-effector is close to the target position.
        :param target_orientation: The orientation of the target
        :param target_position: the position of the target
        :return: True if the end-effector is close to the target position, False otherwise
        """
        gripper_pos, gripper_orientation = p.getLinkState(self.objects["arm"], 11)[:2]
        if target_orientation is None:
            target_orientation = gripper_orientation
        if len(target_orientation) != 4:
            target_orientation = Rotation.from_matrix(target_orientation).as_quat()
        return np.allclose(target_position, gripper_pos, atol=0.05) and np.allclose(
            target_orientation, gripper_orientation, atol=0.05
        )

    def get_camera_info(self):
        """
        Get the camera information for the wrist camera.
        :return: the width, height, projection matrix, and view matrix of the camera
        """
        width = self.config.IMAGE_SIZE
        height = self.config.IMAGE_SIZE

        fov = 60
        aspect = width / height
        near = 0.01
        far = 100
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # find the camera position to be at the wrist of the robot
        camera_position, rot_matrix = self.get_camera_position_and_rotation()
        init_camera_vector = (0, 0, 1)
        init_up_vector = (0, 1, 0)
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(
            camera_position, camera_position + 0.1 * camera_vector, up_vector
        )

        if self.config.VERBOSITY > 0:
            self.draw_debug_camera_axis(camera_position, rot_matrix)

        return width, height, projection_matrix, view_matrix, fov, near, far

    @staticmethod
    def draw_debug_camera_axis(camera_position, rot_matrix):
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

    def get_camera_position_and_rotation(self):
        """
        Get the position and rotation of the camera.
        :return: the position and rotation of the camera
        """
        eef_position, eef_rotation = p.getLinkState(self.objects["arm"], 11)[:2]
        eef_rotation = np.array(p.getMatrixFromQuaternion(eef_rotation)).reshape(3, 3)
        eef_to_camera_position, eef_to_camera_orientation = (
            self.get_eef_to_camera_transform()
        )
        camera_position = np.array(eef_position) + np.dot(
            eef_rotation, eef_to_camera_position
        )
        camera_rotation = np.dot(eef_rotation, eef_to_camera_orientation)

        return camera_position, camera_rotation

    def move(self, t, R):
        """
        Inputs: t_meters: (x,y,z) translation in end-effector frame
                R: (3x3) array - rotation matrix in end-effector frame

        Moves and rotates the robot according to the input translation and rotation.
        """
        current_pos, current_rot = p.getLinkState(self.objects["arm"], 11)[:2]
        current_rot = np.array(p.getMatrixFromQuaternion(current_rot)).reshape(3, 3)

        # calculate the desired position and rotation in world frame
        if self.config.VERBOSITY > 0:
            print("Current position:", current_pos, "Current rotation:", current_rot)
        desired_pos = current_pos + np.dot(current_rot, t)
        desired_rot = np.dot(current_rot, R)
        if self.config.VERBOSITY > 0:
            print("Desired position:", desired_pos, "Desired rotation:", desired_rot)
        desired_rot = Rotation.from_matrix(desired_rot).as_quat()

        self.move_with_debug_dot(desired_pos, desired_rot)

    def move_with_debug_dot(self, desired_pos, desired_rot):
        """
        Move the robot to the desired position and orientation, and add a red point at the desired position.
        :param desired_pos: The desired position
        :param desired_rot: The desired rotation
        """
        if self.config.VERBOSITY > 0:
            # add a red point at the desired position
            red_dot = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[1, 0, 0, 1],
                visualFramePosition=desired_pos,
            )
            red_dot_id = p.createMultiBody(
                baseVisualShapeIndex=red_dot,
                baseMass=0,
                baseInertialFramePosition=desired_pos,
            )
            p.resetBasePositionAndOrientation(red_dot_id, desired_pos, desired_rot)
        # move the robot
        self.move_to_target(desired_pos, desired_rot)
        if self.config.VERBOSITY > 0:
            p.removeBody(red_dot_id)

    def replay_demo(self, demo):
        """
        Replays a demonstration by moving the end-effector according to the input velocities.
        :param demo: The demonstration to replay
        """
        if self.config.VERBOSITY > 0:
            print("Replaying demonstration", demo)
        for velocities in demo:
            p.setJointMotorControlArray(
                self.objects["arm"],
                range(7),
                p.VELOCITY_CONTROL,
                targetVelocities=velocities,
            )
            p.stepSimulation()
            self.get_rgbd_image()

    def record_demo(self):
        """
        Record a demonstration by moving the end-effector, and stores velocities
        that can then be replayed by the "replay_demo" function.
        """
        bn_image, bn_depth = self.get_rgbd_image()
        velocities = []
        target_pos = self.get_target_position()

        while not self.is_target_reached(target_pos):
            self.step_to_target(target_pos)
            joint_states = p.getJointStates(self.objects["arm"], range(7))
            velocities.append([link_state[1] for link_state in joint_states])

        return {"rgb_bn": bn_image, "depth_bn": bn_depth, "demo_vels": velocities}

    @staticmethod
    def get_intrinsics(fov, width, height):
        """
        Get the intrinsic parameters of the camera.
        Adapted from https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet/60450420#60450420
        """
        aspect = width / height
        fx = width / (2 * aspect * np.tan(np.radians(fov / 2)))
        fy = height / (2 * np.tan(np.radians(fov / 2)))
        cx = width / 2
        cy = height / 2
        return fx, fy, cx, cy

    @staticmethod
    def get_extrinsics(view_matrix):
        """
        Get the extrinsic parameters of the camera.
        Adapted from https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet/60450420#60450420
        """
        view_matrix = np.array(view_matrix).reshape(4, 4)
        Tc = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).reshape(4, 4)
        T = np.linalg.inv(view_matrix) @ Tc

        return T

    def reset(self):
        """
        Reset the simulation environment.
        """
        self._setup_simulation_basic()
        self.load_object()

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
        width, height, _, _, fov, near, far = self.get_camera_info()
        z_n = 2.0 * depth - 1.0
        depth = 2.0 * near * far / (far + near - z_n * (far - near))
        fx, fy, cx, cy = self.get_intrinsics(fov, width, height)
        projected_points = []
        for u, v in points:
            z = depth[u, v]
            x = (cx - v) * z / fx
            y = (cy - u) * z / fy
            projected_points.append([x, y, z])
        projected_points = np.array(projected_points)
        return projected_points

    @staticmethod
    def get_eef_to_camera_transform():
        """
        Get the transformation matrix from camera frame to end-effector frame.
        """
        camera_to_eef_position = np.array([0, 0, 0.05])
        camera_to_eef_orientation = np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        ).reshape(3, 3)
        return camera_to_eef_position, camera_to_eef_orientation

    def move_in_camera_frame(self, t, R):
        """
        Move the robot in the camera frame.
        :param t: The translation in the camera frame
        :param R: The rotation in the camera frame
        """
        if self.config.VERBOSITY > 0:
            print("Moving in camera frame")
        camera_position, camera_rotation = self.get_camera_position_and_rotation()
        camera_to_eef_position, camera_to_eef_orientation = (
            self.get_eef_to_camera_transform()
        )
        eef_position = camera_position - np.dot(camera_rotation, camera_to_eef_position)
        eef_rotation = np.dot(camera_rotation, np.linalg.inv(camera_to_eef_orientation))
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
        new_camera_position = camera_position + np.dot(camera_rotation, t)
        new_camera_rotation = np.dot(camera_rotation, R)

        new_eef_position = new_camera_position - np.dot(
            new_camera_rotation, camera_to_eef_position
        )
        new_eef_orientation = np.dot(
            new_camera_rotation, np.linalg.inv(camera_to_eef_orientation)
        )

        if self.config.VERBOSITY > 0:
            print(
                "Old camera position:",
                camera_position,
                "Old camera orientation:",
                camera_rotation,
            )
            print(
                "New camera position:",
                new_camera_position,
                "New camera orientation:",
                new_camera_rotation,
            )
            print(
                "Old eef position:", eef_position, "Old eef orientation:", eef_rotation
            )
            print(
                "New eef position:",
                new_eef_position,
                "New eef orientation:",
                new_eef_orientation,
            )

        new_eef_orientation = Rotation.from_matrix(new_eef_orientation).as_quat()

        self.move_with_debug_dot(new_eef_position, new_eef_orientation)

    def project_points_to_world_frame(self, points):
        """
        Project points from the camera frame to the world frame.
        :param points: The points to project
        :return: The points in the world frame
        """
        camera_position, camera_rotation = self.get_camera_position_and_rotation()
        world_points = []
        for i in range(len(points)):
            point = camera_position + np.dot(camera_rotation, points[i])
            world_points.append(point)
        return world_points

    def draw_points_in_3d(self, live_img, depth_buffer, points):
        """
        Draw projections of the points in 3D space.
        :param live_img: The live image from the camera
        :param depth_buffer: The depth buffer from the camera
        :param points: The points to draw
        :return: The points drawn in 3D space
        """
        all_pixels_in_camera_feed = []
        for x in range(self.config.IMAGE_SIZE):
            for y in range(self.config.IMAGE_SIZE):
                all_pixels_in_camera_feed.append([x, y])
        reduced_pixels = all_pixels_in_camera_feed[::8]
        points3d = self.project_to_3d(reduced_pixels, depth_buffer)
        world_points = self.project_points_to_world_frame(points3d)
        # draw the points in 3D using debug points
        colors = live_img.reshape(-1, 3) / 255.0  # Normalize BGR image values to [0, 1]
        reduced_colors = colors[::8]
        p.removeAllUserDebugItems()
        if points != -1:
            p.removeUserDebugItem(points)
        points = p.addUserDebugPoints(world_points, reduced_colors, 5)
        return points


if __name__ == "__main__":
    """
    A sample script to spawn some cubes on a table and move the robot arm towards one of them.
    """
    config = Config()
    env = ArmEnv(config)

    # env.load_object(x=0.6, y=0.3)
    # pos = env.get_target_position("object")
    # env.move_to_target(pos)

    # env.move(np.array([0, 0, 0]),
    #          np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, 0]))).reshape(3, 3))
    # env.move_in_camera_frame(np.array([0, 0.3, 0]), np.eye(3))
    # env.move(np.array([0, 0, 0]),
    #          np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, np.pi / 2]))).reshape(3, 3))
    # env.move(np.array([0, 0, 0]),
    #          np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, -np.pi / 2]))).reshape(3, 3))
    #
    # env.move_to_target(np.array([1, 1, 0.821]), np.eye(3))
    #
    # env.move(np.array([0.3, 0, 0]), np.eye(3))
    # env.move(np.array([0, 0.3, 0]), np.eye(3))
    # env.move(np.array([-0.3, 0, 0]), np.eye(3))
    # env.move(np.array([0, -0.3, 0]), np.eye(3))
    # env.move(np.array([0.3, 0, 0]), np.eye(3))
    # env.move(np.array([0, 0, 0]),
    #          np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, np.pi / 2]))).reshape(3, 3))
    # env.move(np.array([0, 0, 0]),
    #          np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, -np.pi / 2]))).reshape(3, 3))
    # env.move(np.array([-0.3, 0, 0]), np.eye(3))

    # env.move(np.array([0.25, 0, 0]), np.eye(3))
    # env.move(np.array([0, 0.25, 0]), np.eye(3))
    # env.move(np.array([-0.25, 0, 0]), np.eye(3))
    # env.move(np.array([0, -0.25, 0]), np.eye(3))
    # env.move(np.array([0, 0, 0]),
    #          np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, np.pi / 2]))).reshape(3, 3))
    # env.move(np.array([0, 0.25, 0]), np.eye(3))

    # env.move_in_camera_frame(np.array([0.3, 0, 0]), np.eye(3))
    # env.move_in_camera_frame(np.array([0, 0.3, 0]), np.eye(3))
    # env.move_in_camera_frame(np.array([-0.3, 0, 0]), np.eye(3))
    # env.move_in_camera_frame(np.array([0, -0.3, 0]), np.eye(3))
    # env.move_in_camera_frame(np.array([0.3, 0, 0]), np.eye(3))
    # env.move_in_camera_frame(np.array([0, 0, 0]), np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, np.pi / 2]))).reshape(3, 3))
    # env.move_in_camera_frame(np.array([0, 0, 0]), np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, -np.pi / 2]))).reshape(3, 3))
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

    # demo = env.record_demo()
    #
    # env.reset()
    # env.load_object(x=0.6, y=-0.3)
    #
    # env.replay_demo(demo)

    # target_ind = env.get_random_cube_index()
    # target_pos = env.get_target_position("cube" + str(target_ind))
    # print(f"Target index: {target_ind}")
    # print(f"Target position: {target_pos}")
    #
    # # let the cubes drop
    # for step in range(100):
    #     p.stepSimulation()
    #
    # # start moving the arm
    # for step in range(300):
    #     target_pos = env.get_target_position("cube" + str(target_ind))
    #     if step % 10 == 0:
    #         print(f"Target position: {target_pos}")
    #     env.step_to_target(target_pos)
    #     env.take_picture(None)
    #     # if the end effector is close to target, stop
    #     if env.is_target_reached(target_pos):
    #         print("Target reached!")
    #         break
