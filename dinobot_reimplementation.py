"""
This is a reimplementation of the DINOBot algorithm. It allows for (hopefully) effortless swapping in different
environments. A server running the DINO model is used to offload the feature extraction and correspondence finding
processes.
"""

import os
import shutil
import time

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms.functional as F
import wandb

from config import Config
from dino_features import get_correspondences
from sim_keyboard_demo_capturing import DemoSim


def find_transformation(X, Y, config):
    """
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)

    # Code taken from https://nghiaho.com/?page_id=671
    if np.linalg.det(R) < 0:
        if config.VERBOSITY > 0:
            print("det(R) < 0, reflection detected")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Determine translation vector
    t = cY - np.dot(R, cX)
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
    Clear all images from the working directory
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


def set_up_images_directory(config):
    base_directory = config.IMAGE_DIR
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    new_directory = base_directory + "/" + str(time.time())
    try:
        os.makedirs(new_directory)
    except Exception as e:
        print("Failed to create %s. Reason: %s" % (new_directory, e))
    return new_directory


def deploy_dinobot(env, data, config, image_directory):
    """
    Run the main dinobot loop with a single object
    :param env: The environment in which the robot is operating
    :param data: The data containing the bottleneck images and the demonstration velocities (after semantic retrieval)
    :param config: The configuration object
    :param image_directory: The base image directory
    """
    rgb_bn, depth_bn, demo_vels = (
        data["rgb_bn"],
        data["depth_bn"],
        data["demo_vels"],
    )
    # transform the images to match the necessary shapes
    rgb_bn = torch.tensor(rgb_bn).swapaxes(0, 1).swapaxes(0, 2)
    depth_bn = torch.tensor(depth_bn).unsqueeze(0)

    rgb_bn = F.resize(rgb_bn, config.LOAD_SIZE)
    depth_bn = F.resize(depth_bn, config.LOAD_SIZE)

    rgb_bn = rgb_bn.swapaxes(0, 2).swapaxes(0, 1).numpy()
    depth_bn = depth_bn.squeeze(0).numpy()

    rgb_bn_path = save_rgb_image(rgb_bn, "bn", image_directory)
    error = np.inf
    counter = 0
    num_patches, descriptor_vectors, points1_2d = None, None, None
    while error > config.ERR_THRESHOLD:
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
            )
            points1_2d, points2_2d, time_taken, num_patches, descriptor_vectors = (
                results["points1_2d"],
                results["points2_2d"],
                results["time_taken"],
                results["num_patches"],
                results["descriptor_vectors"],
            )
        else:
            results = get_correspondences(
                config, counter, rgb_bn_path, rgb_live_path
            )
            points1_2d, points2_2d, time_taken = (
                results["points1_2d"],
                results["points2_2d"],
                results["time_taken"],
            )

        # Given the pixel coordinates of the correspondences, add the depth channel.
        points1 = env.project_to_3d(points1_2d, depth_bn)
        points2 = env.project_to_3d(points2_2d, depth_live)

        error = compute_error(points1, points2)
        if config.VERBOSITY > 0:
            print(f"Error: {error}, time taken: {time_taken}")

        if error < config.ERR_THRESHOLD:
            break

        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2, config)

        # Move robot
        env.move_in_camera_frame(t, R)
        counter += 1

        if counter > config.TRIES_LIMIT:
            print("Reached tries limit")
            return False

    # Once error is small enough, replay demo.
    return env.replay_demo(demo_vels)


def get_embeddings(image_path, url):
    """
    Get the embeddings of an image from the DINO server using the embeddings endpoint
    :param image_path: The path to the image
    :param url: The url of the DINO server
    :return: The embeddings of the image
    """
    url += "embeddings"
    with open(image_path, "rb") as f:
        files = {"image": f.read()}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        return parsed_response["embeddings"]
    else:
        print(response.json())


def calculate_object_similarities(config):
    """
    Calculate the similarities between all images in a directory using the embeddings from the DINO server and the cosine
    similarity metric.
    :param config: The configuration object
    :return: A dictionary containing the similarities between all pairs of images
    """
    image_embeddings = {}
    for filename in os.listdir(config.IMAGE_DIR):
        file_path = os.path.join(config.IMAGE_DIR, filename)
        img_emb = get_embeddings(file_path, config.BASE_URL)
        image_embeddings[filename.split(".")[0]] = img_emb

    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    similarities = {}
    for a in image_embeddings:
        for b in image_embeddings:
            if a != b and (a, b) not in similarities and (b, a) not in similarities:
                similarities[(a, b)] = cos_sim(image_embeddings[a], image_embeddings[b])

    return similarities


def run_dino_once(config):
    # Remove all images from the working directory
    image_directory = set_up_images_directory(config)

    # RECORD DEMO:
    env = DemoSim(config)
    # for now, just load the latest demo
    data = env.load_demonstration("demonstrations/demonstration_000.json")

    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    env.reset()
    # load a new object
    success = deploy_dinobot(env, data, config, image_directory)
    env.disconnect()
    clear_images(image_directory)
    return success


def run_fast_and_slow(config):
    config.VERBOSITY = 1
    config.USE_GUI = False
    config.USE_FAST_CORRESPONDENCES = True
    num_of_runs = 10
    successes_1 = 0
    for i in range(num_of_runs):
        config.SEED = i
        success = run_dino_once(config)
        if success:
            successes_1 += 1

    config.USE_FAST_CORRESPONDENCES = False
    successes_2 = 0
    for i in range(num_of_runs):
        config.SEED = i
        success = run_dino_once(config)
        if success:
            successes_2 += 1
    return successes_1 + successes_2


def run_hyperparam_search():
    wandb.login(key="8d9dd70311672d46669adf913d75468f2ba2095b")

    sweep_config = {
        "name": "dinobot",
        "method": "bayes",
        "metric": {"name": "successes", "goal": "maximize"},
        "parameters": {
            "load_size": {"values": [224, 240, 320, 360, 400]},
            "stride": {"values": [2, 4, 8]},
            "thresh": {"min": 0.1, "max": 0.25},
            "err_threshold": {"min": 0.01, "max": 0.05},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="dinobot")

    def train(w_config=None):
        with wandb.init(config=w_config):
            w_config = wandb.config
            c = Config()

            c.LOAD_SIZE = w_config["load_size"]
            c.STRIDE = w_config["stride"]
            c.THRESH = w_config["thresh"]
            c.ERR_THRESHOLD = w_config["err_threshold"]

            wandb.log({"config": c.__dict__})
            print(f"Config: {c.__dict__}")

            start_time = time.time()

            successes = run_fast_and_slow(c)

            wandb.log({"successes": successes, "time_taken": time.time() - start_time})
            return successes

    wandb.agent(sweep_id, train, count=1)

    wandb.finish()


if __name__ == "__main__":
    run_hyperparam_search()
