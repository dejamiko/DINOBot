"""
This is a reimplementation of the DINOBot algorithm. It allows for (hopefully) effortless swapping in different
environments. A server running the DINO model is used to offload the feature extraction and correspondence finding
processes.
"""

import json
import os
import shutil
import time

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms.functional as F
import wandb
from PIL import Image

from config import Config
from sim_keyboard_demo_capturing import DemoSim


def find_correspondences(image_path1, image_path2, url, config):
    url += "correspondences"
    with open(image_path1, "rb") as f:
        files = {"image1": f.read()}
    with open(image_path2, "rb") as f:
        files["image2"] = f.read()
    files["config"] = json.dumps(config)
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        if config["draw"]:
            image1_correspondences = Image.fromarray(
                np.array(parsed_response["image1_correspondences"], dtype="uint8")
            )
            image2_correspondences = Image.fromarray(
                np.array(parsed_response["image2_correspondences"], dtype="uint8")
            )
            return (
                parsed_response["points1"],
                parsed_response["points2"],
                image1_correspondences,
                image2_correspondences,
                parsed_response["time_taken"],
            )
        return (
            parsed_response["points1"],
            parsed_response["points2"],
            parsed_response["time_taken"],
        )
    else:
        print(response.json())


def find_correspondeces_fast(
        image_path1,
        image_path2,
        url,
        config,
        num_patches=None,
        descriptor_vectors=None,
        points1=None,
):
    url += "correspondences_fast"
    with open(image_path1, "rb") as f:
        files = {"image1": f.read()}
    with open(image_path2, "rb") as f:
        files["image2"] = f.read()
    args = {
        "num_patches": num_patches,
        "descriptor_vectors": descriptor_vectors,
        "points1_2d": points1,
        "config": config,
    }

    # save the arguments to a file as a json object and send this file
    files["args"] = json.dumps(args)

    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        if config["draw"]:
            image1_correspondences = Image.fromarray(
                np.array(parsed_response["image1_correspondences"], dtype="uint8")
            )
            image2_correspondences = Image.fromarray(
                np.array(parsed_response["image2_correspondences"], dtype="uint8")
            )
            return (
                parsed_response["points1"],
                parsed_response["points2"],
                image1_correspondences,
                image2_correspondences,
                parsed_response["time_taken"],
                parsed_response["num_patches"],
                parsed_response["descriptor_vectors"],
            )
        return (
            parsed_response["points1"],
            parsed_response["points2"],
            parsed_response["time_taken"],
            parsed_response["num_patches"],
            parsed_response["descriptor_vectors"],
        )
    else:
        print(response.json())


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


def save_rgb_image(image, filename):
    """
    Take a picture with the wrist camera and save it to disk.
    :param image: The image to save
    :param filename: The filename to save the image as
    :return: The path to the saved image
    """
    # first see if there already is an image with the same name
    im_name = f"images/rgb_image_{filename}_0.png"
    i = 0
    while os.path.exists(im_name):
        i += 1
        im_name = f"images/rgb_image_{filename}_{i}.png"

    cv2.imwrite(im_name, image)

    return im_name


def clear_images(config):
    """
    Clear all images from the working directory
    """
    if not os.path.exists(config.IMAGE_DIR):
        os.makedirs(config.IMAGE_DIR)
    folder = config.IMAGE_DIR
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def deploy_dinobot(env, data, config):
    """
    Run the main dinobot loop with a single object
    :param env: The environment in which the robot is operating
    :param data: The data containing the bottleneck images and the demonstration velocities (after semantic retrieval)
    :param config: The configuration object
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

    rgb_bn_path = save_rgb_image(rgb_bn, "bn")
    error = np.inf
    counter = 0
    while error > config.ERR_THRESHOLD:
        # Collect observations at the current pose.
        rgb_live, depth_live = env.get_rgbd_image()
        rgb_live_path = save_rgb_image(rgb_live, "live")

        # Compute pixel correspondences between new observation and bottleneck observation.
        if config.DRAW_CORRESPONDENCES:
            if config.USE_FAST_CORRESPONDENCES:
                if counter % config.RECOMPUTE_EVERY == 0:
                    (
                        points1_2d,
                        points2_2d,
                        im_1_c,
                        im_2_c,
                        time_taken,
                        num_patches,
                        descriptor_vectors,
                    ) = find_correspondeces_fast(
                        rgb_bn_path,
                        rgb_live_path,
                        config.BASE_URL,
                        config.get_dino_config(),
                    )
                else:
                    (
                        points1_2d,
                        points2_2d,
                        im_1_c,
                        im_2_c,
                        time_taken,
                        num_patches,
                        descriptor_vectors,
                    ) = find_correspondeces_fast(
                        rgb_bn_path,
                        rgb_live_path,
                        config.BASE_URL,
                        config.get_dino_config(),
                        num_patches,
                        descriptor_vectors,
                        points1_2d,
                    )
            else:
                points1_2d, points2_2d, im_1_c, im_2_c, time_taken = (
                    find_correspondences(
                        rgb_bn_path,
                        rgb_live_path,
                        config.BASE_URL,
                        config.get_dino_config(),
                    )
                )

            im_1_c.save(f"images/image1_correspondences_{counter}.png")
            im_2_c.save(f"images/image2_correspondences_{counter}.png")
        else:
            if config.USE_FAST_CORRESPONDENCES:
                if counter % config.RECOMPUTE_EVERY == 0:
                    (
                        points1_2d,
                        points2_2d,
                        time_taken,
                        num_patches,
                        descriptor_vectors,
                    ) = find_correspondeces_fast(
                        rgb_bn_path,
                        rgb_live_path,
                        config.BASE_URL,
                        config.get_dino_config(),
                    )
                else:
                    (
                        points1_2d,
                        points2_2d,
                        time_taken,
                        num_patches,
                        descriptor_vectors,
                    ) = find_correspondeces_fast(
                        rgb_bn_path,
                        rgb_live_path,
                        config.BASE_URL,
                        config.get_dino_config(),
                        num_patches,
                        descriptor_vectors,
                        points1_2d,
                    )
            else:
                points1_2d, points2_2d, time_taken = find_correspondences(
                    rgb_bn_path,
                    rgb_live_path,
                    config.BASE_URL,
                    config.get_dino_config(),
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
    clear_images(config)

    # RECORD DEMO:
    env = DemoSim(config)
    data = env.load_demonstration("demonstrations/demonstration_000.json")

    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    env.reset()
    # load a new object
    success = deploy_dinobot(env, data, config)
    env.disconnect()
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
            "err_threshold": {"min": 0.01, "max": 0.05}
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
