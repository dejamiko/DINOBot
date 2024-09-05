import os
import shutil
import time
import traceback

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as f
from DINOserver.dino_vit_features.extractor import ViTExtractor

from config import Config
from connector import get_correspondences
from database import DB
from demo_sim_env import DemoSimEnv
from task_types import Task


def find_transformation(x, y, config):
    """
    Find transformation given two sets of correspondences between 3D points.
    :param x: the first set of points
    :param y: the second set of points
    :param config: The configuration used
    :return: R - 3x3 rotation matrix, t - 3-dim translation array.
    """
    # Calculate centroids
    x_centroid = np.mean(x, axis=0)
    y_centroid = np.mean(y, axis=0)
    # Subtract centroids to obtain centered sets of points
    x_centered = x - x_centroid
    y_centered = y - y_centroid
    # Calculate covariance matrix
    covariance = np.dot(x_centered.T, y_centered)
    # Compute SVD
    U, S, Vt = np.linalg.svd(covariance)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)

    # Code taken from https://nghiaho.com/?page_id=671
    if np.linalg.det(R) < 0:
        if config.VERBOSITY > 0:
            print("det(R) < 0, reflection detected")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Determine translation vector
    # post-multiply row vector with transposed rotation matrix
    t = y_centroid - np.dot(x_centroid, R.T)
    t2 = y_centroid - np.dot(R, x_centroid)

    assert np.allclose(t, t2), f"{t} != {t2}"

    assert np.allclose(
        np.linalg.det(R), 1
    ), f"Expected the rotation matrix to describe a rigid transform, got det={np.linalg.det(R)}"

    return R, t


def find_transformation_lower_DOF(x, y, config):
    """
    Find transformation given two sets of correspondences between 3D points. Lower the degrees of freedom to only be
    x, y, and yaw.
    :param x: the first set of points
    :param y: the second set of points
    :param config: The configuration used
    :return: R - 3x3 rotation matrix, t - 3-dim translation array.
    """
    # Calculate centroids
    x_centroid = np.mean(x, axis=0)
    y_centroid = np.mean(y, axis=0)
    # Subtract centroids to obtain centered sets of points
    x_centered = x - x_centroid
    y_centered = y - y_centroid
    # Calculate covariance matrix in 2D
    covariance_2d = np.dot(x_centered[:, :2].T, y_centered[:, :2])
    # Compute SVD
    U, S, Vt = np.linalg.svd(covariance_2d)
    # Determine the rotation matrix in 2d
    R_2d = np.dot(Vt.T, U.T)

    # Code taken from https://nghiaho.com/?page_id=671
    if np.linalg.det(R_2d) < 0:
        if config.VERBOSITY > 0:
            print("det(R) < 0, reflection detected")
        Vt[1, :] *= -1
        R_2d = Vt.T @ U.T

    # Create the full rotation matrix
    R = np.eye(3)
    R[:2, :2] = R_2d

    # Determine the translation vector
    # post-multiply row vector with transposed rotation matrix
    t = y_centroid - np.dot(x_centroid, R.T)
    t2 = y_centroid - np.dot(R, x_centroid)
    assert np.allclose(t, t2), f"{t} != {t2}"

    assert np.allclose(
        np.linalg.det(R), 1
    ), f"Expected the rotation matrix to describe a rigid transform, got det={np.linalg.det(R)}"

    return R, t


def compute_error(points1, points2):
    """
    Compute the error between two sets of points for the purpose of stopping the alignment phase
    :param points1: The first set of points
    :param points2: The second set of points
    :return: The error between the two sets of points. Calculated as the norm of the difference between the two sets of
    points.
    """
    return np.linalg.norm(np.array(points1) - np.array(points2))


def save_rgb_image(image, filename, image_directory):
    """
    Take a picture with the wrist camera and save it to disk.
    :param image: The image to save
    :param filename: The filename to save the image as
    :param image_directory: The directory where the image should be saved
    :return: The path to the saved image
    """
    # first see if there already is an image with the same name
    im_name = f"{image_directory}/rgb_image_{filename}_0.png"
    i = 0
    while os.path.exists(im_name):
        i += 1
        im_name = f"{image_directory}/rgb_image_{filename}_{i}.png"

    cv2.imwrite(im_name, image)

    return im_name


def clear_images(image_directory):
    """
    Clear all images from the working directory and the directory itself.
    This is only used as temporary storage anyway.
    """
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    for filename in os.listdir(image_directory):
        file_path = os.path.join(image_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    os.rmdir(image_directory)


def set_up_images_directory(config):
    """
    Create a temporary images directory
    :param config: The config used
    :return: The temporary images directory
    """
    base_directory = config.IMAGE_DIR
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    new_directory = base_directory + "/" + str(time.time())
    try:
        os.makedirs(new_directory)
    except Exception as e:
        print("Failed to create %s. Reason: %s" % (new_directory, e))
    return new_directory


def deploy_dinobot(env, data, config, image_directory, base_object, target_object):
    """
    Run the main dinobot loop with a single object
    :param env: The environment in which the robot is operating
    :param data: The data containing the bottleneck images and the demonstration velocities (after semantic retrieval)
    :param config: The configuration object
    :param image_directory: The base image directory
    :param base_object: The object on which the original demonstration was recorded
    :param target_object: The target object to be solved
    """
    rgb_bn, depth_bn, demo_positions = (
        data["images"][0],
        data["depth_buffers"][0],
        data["demo_positions"],
    )

    if config.RUN_LOCALLY and config.USE_FAST_CORRESPONDENCES:
        extractor = ViTExtractor(config.MODEL_TYPE, config.STRIDE, device=config.DEVICE)
    else:
        extractor = None

    rgb_bn, depth_bn = transform_images(config, depth_bn, rgb_bn)

    rgb_bn_path = save_rgb_image(rgb_bn, "bn", image_directory)
    error = np.inf
    counter = 0
    num_patches, descriptor_vectors, points1_2d = None, None, None
    best_alignment_joint_position = None
    smallest_error = None
    while error > config.ERR_THRESHOLD:
        counter += 1
        if counter > config.TRIES_LIMIT:
            print("Reached tries limit")
            break

        # Collect observations at the current pose.
        rgb_live, depth_live = env.get_rgbd_image()
        rgb_live_path = save_rgb_image(rgb_live, "live", image_directory)

        # Compute pixel correspondences between new observation and bottleneck observation.
        if config.USE_FAST_CORRESPONDENCES:
            results = get_correspondences(
                config,
                counter,
                rgb_bn_path,
                rgb_live_path,
                num_patches=num_patches,
                descriptor_vectors=descriptor_vectors,
                points1_2d=points1_2d,
                extractor=extractor,
            )
            points1_2d, points2_2d, time_taken, num_patches, descriptor_vectors = (
                results["points1_2d"],
                results["points2_2d"],
                results["time_taken"],
                results["num_patches"],
                results["descriptor_vectors"],
            )
        else:
            results = get_correspondences(config, counter, rgb_bn_path, rgb_live_path)
            points1_2d, points2_2d, time_taken = (
                results["points1_2d"],
                results["points2_2d"],
                results["time_taken"],
            )

        if config.DRAW_CORRESPONDENCES:
            results["image1_correspondences"].save(
                f"{image_directory}/image1_correspondences_{counter}.png"
            )
            results["image2_correspondences"].save(
                f"{image_directory}/image2_correspondences_{counter}.png"
            )

        # Given the pixel coordinates of the correspondences, add the depth channel.
        points1 = env.project_to_3d(points1_2d, depth_bn)
        points2 = env.project_to_3d(points2_2d, depth_live)

        error = compute_error(points1, points2)
        if config.VERBOSITY > 0:
            print(f"Error: {error}, time taken: {time_taken}")

        if smallest_error is None or error < smallest_error:
            smallest_error = error
            best_alignment_joint_position = env.get_current_joint_positions()

        if error < config.ERR_THRESHOLD:
            # we can stop, the alignment is good enough
            break

        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation_lower_DOF(points1, points2, config)

        # Move robot
        env.move_in_camera_frame(t, R)

    env.move_to_target_joint_position(best_alignment_joint_position)
    env.pause()
    # before replay, store the current state, so it can be reloaded and replayed
    env.store_state(base_object, target_object)
    # Once error is small enough, replay demo.
    return env.replay_demo(demo_positions), counter


def transform_images(config, depth_bn, rgb_bn):
    """
    Transform images to have the same shapes
    :param config: The config used
    :param depth_bn: The depth buffer in the bottleneck pose
    :param rgb_bn: The image in the bottleneck pose
    :return: The depth buffer and image in the bottleneck pose after resizing
    """
    rgb_bn = torch.tensor(rgb_bn).swapaxes(0, 1).swapaxes(0, 2)
    depth_bn = torch.tensor(depth_bn).unsqueeze(0)
    rgb_bn = f.resize(rgb_bn, config.LOAD_SIZE)
    depth_bn = f.resize(depth_bn, config.LOAD_SIZE)
    rgb_bn = rgb_bn.swapaxes(0, 2).swapaxes(0, 1).numpy()
    depth_bn = depth_bn.squeeze(0).numpy()
    return rgb_bn, depth_bn


def run_dino_once(config, db, base_object, target_object, task):
    """
    Run dinobot once for a given target and base object pair
    :param config: The configuration used
    :param db: The database
    :param target_object: The target object on which the demo is to be performed
    :param base_object: The object on which the demo was recorded
    :param task: The task to be performed
    :return: The success of the dinobot deployment
    """
    try:
        image_directory = set_up_images_directory(config)

        env = DemoSimEnv(
            config, task, *db.get_load_info(target_object, task), db.get_nail_object()
        )
        data = env.load_demonstration(db.get_demo(base_object, task))

        success, tries = deploy_dinobot(
            env, data, config, image_directory, base_object, target_object
        )
        env.disconnect()
        if not config.USE_GUI:
            clear_images(image_directory)
        if config.VERBOSITY > 0:
            print(success, tries)
    except Exception as e:
        print(traceback.format_exc())
        if config.USE_GUI:
            raise e
        return False, -1
    return success, tries


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 1
    config.USE_GUI = True
    config.RUN_LOCALLY = False
    config.USE_FAST_CORRESPONDENCES = True
    config.DRAW_CORRESPONDENCES = True
    config.SEED = 0
    config.BASE_URL = "http://localhost:8080/"
    db = DB(config)
    success, tries = run_dino_once(
        config, db, "banana", "banana", Task.GRASPING.value
    )
