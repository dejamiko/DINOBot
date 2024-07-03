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

from config import Config
from dinobot_utils import extract_descriptors, extract_descriptor_nn, extract_desc_maps
from dino_vit_features.correspondences import draw_correspondences
from sim import ArmEnv


# def find_correspondences(image_path1, image_path2, url, draw=False):
#     """
#     Find correspondences between two images using the DINO-ViT features. This uses the DINO server to extract the
#     features and find the correspondences between the two images.
#     :param image_path1: A path to the first image
#     :param image_path2: A path to the second image
#     :param url: The url of the DINO server
#     :param draw: A flag to determine whether to draw the correspondences on the images
#     :return: The correspondences between the two images, the two images, the images with the correspondences drawn on
#     them, and the time taken to find the correspondences. If draw is False, the images with the correspondences are not
#     returned.
#     """
#     url += 'correspondences'
#     with open(image_path1, 'rb') as f:
#         files = {'image1': f.read()}
#     with open(image_path2, 'rb') as f:
#         files['image2'] = f.read()
#     args = {'draw': draw}
#     response = requests.post(url, files=files, data=args)
#     if response.status_code == 200:
#         parsed_response = response.json()
#         if draw:
#             image1_correspondences = Image.fromarray(np.array(parsed_response["image1_correspondences"], dtype='uint8'))
#             image2_correspondences = Image.fromarray(np.array(parsed_response["image2_correspondences"], dtype='uint8'))
#             return (
#                 parsed_response["points1"],
#                 parsed_response["points2"],
#                 image1_correspondences,
#                 image2_correspondences,
#                 parsed_response["time_taken"]
#             )
#         else:
#             return (
#                 parsed_response["points1"],
#                 parsed_response["points2"],
#                 parsed_response["time_taken"]
#             )
#     else:
#         print(response.json())


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
        print("det(R) < 0, reflection detected")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

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
    rgb_bn_path = save_rgb_image(rgb_bn, "bn")
    error = np.inf
    counter = 0
    while error > config.ERR_THRESHOLD:
        # Collect observations at the current pose.
        rgb_live, depth_live = env.get_rgbd_image()
        rgb_live_path = save_rgb_image(rgb_live, "live")

        # Compute pixel correspondences between new observation and bottleneck observation.
        start_time = time.time()
        if counter % 10 == 0:
            # print("Extracting descriptors for the bottleneck image")
            patches_xy, desc1, descriptor_vectors, num_patches = extract_descriptors(rgb_bn_path, rgb_live_path)
            descriptor_list, org_images_list = extract_desc_maps([rgb_bn_path])
            key_y, key_x = extract_descriptor_nn(descriptor_vectors, descriptor_list[0], num_patches)
            points1_2d = [(y, x) for y, x in zip(np.array(key_y) * config.stride, np.array(key_x) * config.stride)]

        # print("Extracting descriptors for the live image")
        descriptor_list, org_images_list = extract_desc_maps([rgb_live_path])
        key_y, key_x = extract_descriptor_nn(descriptor_vectors, descriptor_list[0], num_patches)
        points2_2d = [(y, x) for y, x in zip(np.array(key_y) * config.stride, np.array(key_x) * config.stride)]

        # fig1, fig2 = draw_correspondences(points1_2d, points2_2d, rgb_bn, rgb_live)

        # print(f"Points1: {points1_2d}, Points2: {points2_2d}")

        # save the images
        # img1_c.save('images/image1.png')
        # img2_c.save('images/image2.png')

        # Given the pixel coordinates of the correspondences, add the depth channel.
        points1 = env.project_to_3d(points1_2d, depth_bn)
        points2 = env.project_to_3d(points2_2d, depth_live)

        error = compute_error(points1, points2)
        print(f"Error: {error}, time taken: {time.time() - start_time}")

        # print(f"Points1: {points1}, Points2: {points2}")

        if error < config.ERR_THRESHOLD:
            break

        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)

        # try using the other method
        points1_t = np.transpose(points1)
        points2_t = np.transpose(points2)
        R_2, t_2 = rigid_transform_3D(points1_t, points2_t)

        # print(f"R: {R}, t: {t}, R_2: {R_2}, t_2: {t_2}")

        # Move robot
        env.move_in_camera_frame(t, R)

        counter += 1

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
    np.random.seed(0)
    torch.manual_seed(0)
    env = ArmEnv(config)
    data = env.record_demo()

    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    env.reset()
    # load a new object
    deploy_dinobot(env, data, config)
