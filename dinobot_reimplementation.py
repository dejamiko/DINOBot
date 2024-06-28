"""
This script is my attempt at reimplementing the DINOBot algorithm with the DINO-ViT features and pybullet simulation.

Most functions were moved to the sim.py file and were implemented there.
"""

import os
import shutil
import warnings

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from dino_vit_features.correspondences import draw_correspondences

from config import Config
from sim import ArmEnv

warnings.filterwarnings("ignore")


def find_correspondences(image_path1, image_path2, url):
    """
    Find correspondences between two images using the DINO-ViT features. This uses the DINO server to extract the
    features and find the correspondences between the two images.
    :param image_path1: A path to the first image
    :param image_path2: A path to the second image
    :param url: The url of the DINO server
    :return: The correspondences between the two images, the images with the correspondences drawn on them, and the time
    taken to find the correspondences.
    """
    url += "correspondences"
    with open(image_path1, "rb") as f:
        files = {"image1": f.read()}
    with open(image_path2, "rb") as f:
        files["image2"] = f.read()
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        image1_pil = Image.fromarray(
            np.array(parsed_response["image1_pil"], dtype="uint8")
        )
        image2_pil = Image.fromarray(
            np.array(parsed_response["image2_pil"], dtype="uint8")
        )
        return (
            parsed_response["points1"],
            parsed_response["points2"],
            image1_pil,
            image2_pil,
            parsed_response["time_taken"],
        )
    else:
        print(response.json())


def find_transformation(X, Y):
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
        print("det(R) < 0, reflection detected!, correcting for it ...")
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
        data["rgb_bn_path"],
        data["depth_bn"],
        data["demo_vels"],
    )
    rgb_bn_path = save_rgb_image(rgb_bn, "bn")
    error = np.inf
    while error > config.ERR_THRESHOLD:
        # Collect observations at the current pose.
        rgb_live, depth_live = env.get_rgbd_image()
        rgb_live_path = save_rgb_image(rgb_live, "live")

        # Compute pixel correspondences between new observation and bottleneck observation.
        points1, points2, image1_pil, image2_pil, time = find_correspondences(
            rgb_bn_path, rgb_live_path, config.BASE_URL
        )

        # save the images
        fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
        fig1.savefig(f"images/image1_correspondences.png")
        fig2.savefig(f"images/image2_correspondences.png")

        # Given the pixel coordinates of the correspondences, add the depth channel.
        points1 = env.project_to_3d(points1, depth_bn)
        points2 = env.project_to_3d(points2, depth_live)

        error = compute_error(points1, points2)
        print("error", error)

        if error < config.ERR_THRESHOLD:
            break

        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)

        # Move robot
        env.move_in_camera_frame(t, R)

    # Once error is small enough, replay demo.
    env.replay_demo(demo_vels)


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


if __name__ == "__main__":
    config = Config()
    config.VERBOSITY = 0
    # Remove all images from the working directory
    clear_images(config)

    # RECORD DEMO:
    env = ArmEnv(config)
    env.setup()
    data = env.record_demo()

    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    env.reset()
    # load a new object
    env.setup()
    deploy_dinobot(env, data, config)
