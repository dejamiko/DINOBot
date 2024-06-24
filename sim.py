"""
This file contains the code related to the simulation environment. It can be used within the DINOBot framework
or for small independent experiments with the pybullet simulation environment.
"""

import os

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation


class ArmEnv:
    """
    A class that sets up the simulation environment and provides functions to interact with it. It contains a table,
    an arm, and some objects that can be placed on the table.
    """

    def __init__(self, image_size=224):
        """
        Initialize the simulation environment.
        """
        self.objects = {}
        self.image_size = image_size
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

        # load a table
        # self.objects["table"] = p.loadURDF("table/table.urdf", [0.5, 0, 0])

        # load the arm on top of the table
        self.objects["arm"] = p.loadURDF("franka_panda/panda.urdf", [1, 1, 0], useFixedBase=True)

        focus_position, _ = p.getBasePositionAndOrientation(self.objects["arm"])
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=focus_position)

    def load_object(self, x=None, y=None, object_path="jenga/jenga.urdf"):
        """
        Load an object on the table and move the arm so the camera can see it.
        :param x: the x-coordinate of the object. If None, a random x-coordinate is chosen.
        :param y: the y-coordinate of the object. If None, a random y-coordinate is chosen.
        :param object_path: the path to the object URDF file. Default is "objects/mug.urdf".
        """
        # load some object on the table somewhere random (within a certain range)
        pos_x = x if x is not None else np.random.uniform(1.6, 1.8)
        pos_y = y if y is not None else np.random.uniform(0.9, 1.1)
        angle = np.random.uniform(0, 2 * np.pi)
        self.objects[f"object"] = p.loadURDF(object_path, [pos_x, pos_y, 0.5],
                                             p.getQuaternionFromEuler([0, 0, angle]))
        colour = np.random.uniform(0, 1, 3)
        p.changeVisualShape(self.objects[f"object"], -1, rgbaColor=colour.tolist() + [1])

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

    def load_cubes(self):
        """
        Load some cubes on the table and move the arm so the camera can see them.
        """
        min_num_cubes = 2
        max_num_cubes = 4
        num_cubes = np.random.randint(min_num_cubes, max_num_cubes + 1)
        for i in range(num_cubes):
            pos_x = np.random.uniform(0.6, 0.8)
            pos_y = np.random.uniform(-0.1, 0.1)
            angle = np.random.uniform(0, 2 * np.pi)
            self.objects[f"cube{i}"] = p.loadURDF("cube_small.urdf", [pos_x, pos_y, 1],
                                                  p.getQuaternionFromEuler([0, 0, angle]))
            colour = np.random.uniform(0, 1, 3)
            p.changeVisualShape(self.objects[f"cube{i}"], -1, rgbaColor=colour.tolist() + [1])

        # let the cubes drop
        for i in range(100):
            p.stepSimulation()

        # move the arm a little bit so the camera can see the cubes
        pos = [0.7, 0, 1]
        while not self.is_target_reached(pos):
            self.step_to_target(pos)

    def get_random_cube_index(self):
        """
        Get a random cube index.
        :return: the index of a random cube
        """
        # find how many cubes we have
        num_cubes = len(self.objects) - 2
        cube_index = np.random.randint(0, num_cubes)
        return cube_index

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
        :param target_position: the position of the target
        :param target_orientation: the orientation of the target
        """
        if target_orientation is not None:
            target_joint_positions = p.calculateInverseKinematics(self.objects["arm"], 11, target_position,
                                                                  target_orientation)
        else:
            target_joint_positions = p.calculateInverseKinematics(self.objects["arm"], 11, target_position)

        # interpolate the joint positions over a number of steps
        num_steps = 50
        joint_positions_now = p.getJointStates(self.objects["arm"], range(9))
        joint_positions_now = [joint_state[0] for joint_state in joint_positions_now]
        joint_positions_over_time = np.linspace(joint_positions_now, target_joint_positions, num_steps)
        for joint_positions in joint_positions_over_time:
            p.setJointMotorControlArray(self.objects["arm"], range(9), p.POSITION_CONTROL,
                                        targetPositions=joint_positions)
            p.stepSimulation()
            self.take_picture()

    def step_to_target(self, target_position, target_orientation=None):
        """
        Move the arm to the target position by performing one step.
        :param target_orientation: the orientation of the target
        :param target_position: the position of the target
        """
        if target_orientation is not None:
            target_joint_positions = p.calculateInverseKinematics(self.objects["arm"], 11, target_position,
                                                                  target_orientation)
        else:
            target_joint_positions = p.calculateInverseKinematics(self.objects["arm"], 11, target_position)

        for i in range(len(target_joint_positions)):
            p.setJointMotorControl2(self.objects["arm"], i, p.POSITION_CONTROL, target_joint_positions[i])

        p.stepSimulation()
        self.take_picture()

    def take_picture(self):
        """
        Take a picture with the wrist camera for the purpose of the simulation.
        """
        width, height, projection_matrix, view_matrix, _ = self.get_camera_info()
        p.getCameraImage(width, height, view_matrix, projection_matrix)

    def is_target_reached(self, target_position, target_orientation=None):
        """
        Check if the end-effector is close to the target position.
        :param target_position: the position of the target
        :return: True if the end-effector is close to the target position, False otherwise
        """
        gripper_pos, gripper_orientation = p.getLinkState(self.objects["arm"], 11)[:2]
        if target_orientation is None:
            target_orientation = gripper_orientation
        if np.allclose(target_position, gripper_pos, atol=0.05) and np.allclose(target_orientation, gripper_orientation,
                                                                                atol=0.05):
            print(gripper_pos, target_position, gripper_orientation, target_orientation)
        return np.allclose(target_position, gripper_pos, atol=0.05) and np.allclose(target_orientation,
                                                                                    gripper_orientation, atol=0.05)

    def get_camera_info(self):
        """
        Get the camera information for the wrist camera. This might require some more attention.
        :return: the width, height, projection matrix, and view matrix of the camera
        """
        width = self.image_size
        height = self.image_size

        fov = 60
        aspect = width / height
        near = 0.01
        far = 100
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # find the camera position to be at the wrist of the robot
        com_p, com_o = p.getLinkState(self.objects["arm"], 11)[:2]
        com_p = np.array(com_p)
        com_p[2] -= 0.05
        rot_matrix = p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        init_camera_vector = (0, 0, 1)
        init_up_vector = (0, 1, 0)
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        return width, height, projection_matrix, view_matrix, fov

    def take_picture_and_save(self, filename):
        """
        Take a picture with the wrist camera and save it to disk.
        :param filename: the filename of the saved image
        :return: the filename of the saved image
        :return: the depth buffer of the image
        """
        width, height, projection_matrix, view_matrix, _ = self.get_camera_info()

        # take a picture
        _, _, rgb_image, depth_buffer, _ = p.getCameraImage(width, height, view_matrix,
                                                            projection_matrix)
        rgb_image = np.array(rgb_image).reshape(height, width, -1).astype(np.uint8)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB)

        depth_buffer = np.array(depth_buffer).reshape(height, width).astype(np.float32)

        if filename is None:
            return

        # save the image to disk
        # first see if there already is an image with the same name
        im_name = f"images/rgb_image_{filename}_0.png"
        i = 0
        while os.path.exists(im_name):
            i += 1
            im_name = f"images/rgb_image_{filename}_{i}.png"

        cv2.imwrite(im_name, rgb_image)

        return im_name, depth_buffer

    def move(self, t_meters, R):
        """
        Inputs: t_meters: (x,y,z) translation in end-effector frame
                R: (3x3) array - rotation matrix in end-effector frame

        Moves and rotates the robot according to the input translation and rotation.
        """
        end_effector = p.getLinkState(self.objects["arm"], 11)
        current_pos = end_effector[0]
        current_rot = end_effector[1]
        current_rot = p.getMatrixFromQuaternion(current_rot)
        current_rot = np.array(current_rot).reshape(3, 3)

        # calculate the desired position and rotation in world frame
        print("Current position:", current_pos)
        desired_pos = current_pos + t_meters
        print("Desired position:", desired_pos)
        desired_rot = np.dot(current_rot, R)
        desired_rot = Rotation.from_matrix(desired_rot).as_quat()

        # move the robot
        self.move_to_target(desired_pos, desired_rot)

    def replay_demo(self, demo):
        """
        Inputs: demo: list of velocities

        Replays a demonstration by moving the end-effector according to the input velocities.
        """
        print("Replaying demonstration", demo)
        for velocities in demo:
            p.setJointMotorControlArray(self.objects["arm"], range(7), p.VELOCITY_CONTROL, targetVelocities=velocities)
            p.stepSimulation()
            self.take_picture()

    def record_demo(self):
        """
        Record a demonstration by moving the end-effector, and stores velocities
        that can then be replayed by the "replay_demo" function.

        :return: a list of velocities for each step and each joint
        """
        velocities = []
        target_pos = self.get_target_position()

        while not self.is_target_reached(target_pos):
            self.step_to_target(target_pos)
            joint_states = p.getJointStates(self.objects["arm"], range(7))
            velocities.append([link_state[1] for link_state in joint_states])

        return velocities

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
        Tc = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]]).reshape(4, 4)
        T = np.linalg.inv(view_matrix) @ Tc

        return T

    def convert_pixels_to_meters(self, t):
        """
        Inputs: t : (x,y,z) translation of the end-effector in pixel-space/frame
        Outputs: t_meters : (x,y,z) translation of the end-effector in world frame

        Requires camera calibration to go from a pixel distance to world distance
        """
        # to go from pixel distance to world distance we need to know the camera calibration
        width, height, projection_matrix, view_matrix, fov = self.get_camera_info()
        fx, fy, cx, cy = self.get_intrinsics(fov, width, height)
        T = self.get_extrinsics(view_matrix)
        t_meters = np.array([t[0] / fx, t[1] / fy, t[2]])
        t_meters = T[:3, :3] @ t_meters
        return t_meters

    def reset(self):
        """
        Reset the simulation environment.
        """
        self._setup_simulation_basic()


if __name__ == "__main__":
    """
    A sample script to spawn some cubes on a table and move the robot arm towards one of them.
    """
    env = ArmEnv()

    # env.load_object(x=0.6, y=0.3)
    # pos = env.get_target_position("object")
    # env.move_to_target(pos)

    env.move(np.array([0.5, 0, 0]), np.eye(3))
    env.move(np.array([0, 0.5, 0]), np.eye(3))
    env.move(np.array([-0.5, 0, 0]), np.eye(3))
    env.move(np.array([0, -0.5, 0]), np.eye(3))

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
