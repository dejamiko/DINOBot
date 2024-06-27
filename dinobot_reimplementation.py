"""
This script is my attempt at reimplementing the DINOBot algorithm with the DINO-ViT features and pybullet simulation.

Most functions were moved to the sim.py file and were implemented there.
"""
import os
import shutil
import warnings

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from dino_vit_features.correspondences import draw_correspondences

from sim import ArmEnv

warnings.filterwarnings("ignore")

load_size = 224
url = 'http://146.169.1.68:8000/infer'
# Deployment hyperparameters
ERR_THRESHOLD = 0.05  # A generic error between the two sets of points


def find_correspondences(image_path1, image_path2, url):
    with open(image_path1, 'rb') as f:
        files = {'image1': f.read()}
    with open(image_path2, 'rb') as f:
        files['image2'] = f.read()
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        image1_pil = Image.fromarray(np.array(parsed_response["image1_pil"], dtype='uint8'))
        image2_pil = Image.fromarray(np.array(parsed_response["image2_pil"], dtype='uint8'))
        return (parsed_response["points1"], parsed_response["points2"], image1_pil, image2_pil,
                parsed_response["time_taken"])
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
    return np.linalg.norm(np.array(points1) - np.array(points2))


def filter_points(points1, points2):
    """
    Filter out points that are very close to each other (as they are likely to be the background)
    """
    new_points1 = []
    new_points2 = []
    for i in range(len(points1)):
        if np.linalg.norm(np.array(points1[i]) - np.array(points2[i])) > 1:
            new_points1.append(points1[i])
            new_points2.append(points2[i])
    return new_points1, new_points2


def clear_images():
    # empty the working image directory
    folder = 'images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def record_demo_data(env):
    # RECORD DEMO:
    # Move the end-effector to the bottleneck pose and store observation.
    # Get rgbd from wrist camera.
    rgb_bn, depth_bn = env.take_picture_and_save("bn")

    # Record demonstration.
    demo_vels = env.record_demo()

    return {"rgb_bn": rgb_bn, "depth_bn": depth_bn, "demo_vels": demo_vels}


def plot_points(image, points1, points2):
    print("Plotting the points...")
    # on image, plot points1 and points2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(image)
    for i in range(len(points1)):
        ax.plot(points1[i][1], points1[i][0], 'ro')
        ax.plot(points2[i][1], points2[i][0], 'bo')
        ax.plot([points1[i][1], points2[i][1]], [points1[i][0], points2[i][0]], 'g-')
    plt.show()


def deploy_dinobot(env, data):
    rgb_bn, depth_bn, demo_vels = data["rgb_bn"], data["depth_bn"], data["demo_vels"]
    error = np.inf
    while error > ERR_THRESHOLD:
        # Collect observations at the current pose.
        rgb_live, depth_live = env.take_picture_and_save("lv")

        # Compute pixel correspondences between new observation and bottleneck observation.
        points1, points2, image1_pil, image2_pil, time = find_correspondences(rgb_bn, rgb_live, url)

        # filter out points that are very close to each other (as they are likely to be the background)
        # points1, points2 = filter_points(points1, points2)

        # save the images
        fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
        fig1.savefig(f'images/image1_correspondences.png')
        fig2.savefig(f'images/image2_correspondences.png')

        # Given the pixel coordinates of the correspondences, add the depth channel.
        # points1 = add_depth(points1, depth_bn)
        # points2 = add_depth(points2, depth_live)
        points1 = env.project_to_3d(points1, depth_bn)
        points2 = env.project_to_3d(points2, depth_live)

        print("points1", points1)
        print("points2", points2)

        error = compute_error(points1, points2)
        print("error", error)

        if error < ERR_THRESHOLD:
            break

        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)
        print("R", R)
        print("t", t)

        # transform the points to plot them
        # points1_cp = np.array(points1)
        # points1_cp = np.dot(points1_cp, R.T) + t
        # # plot points1 and points2 to see if they are aligned
        # plot_points(image2_pil, points1_cp, points2)

        # A function to convert pixel distance into meters based on calibration of camera.
        # t_meters = env.convert_pixels_to_meters(t)
        # print("t_meters", t, "R", R)

        # Move robot
        env.move_in_camera_frame(t, R)

    # Once error is small enough, replay demo.
    env.replay_demo(demo_vels)


def get_all_object_images(env):
    return "images/"


def calculate_object_similarities(img_directory):
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    # dino = dino.cuda().float()

    image_transforms = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    image_embeddings = {}
    for filename in os.listdir(img_directory):
        file_path = os.path.join(img_directory, filename)
        img = Image.open(file_path)
        img = image_transforms(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            pan_emb = dino(img)
        image_embeddings[filename.split(".")[0]] = pan_emb

    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    similarities = {}
    for a in image_embeddings:
        for b in image_embeddings:
            if a != b and (a, b) not in similarities and (b, a) not in similarities:
                similarities[(a, b)] = cos_sim(image_embeddings[a], image_embeddings[b])

    return similarities


if __name__ == "__main__":
    clear_images()

    env = ArmEnv(load_size)
    env.verbosity = 0
    env.load_object()

    # this can be one demo or multiple demos
    data = record_demo_data(env)

    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    env.reset()
    # load a new object
    env.load_object()

    img_directory = get_all_object_images(env)
    # img_directory = "images/"
    # similarities = calculate_object_similarities(img_directory)
    #
    # print(similarities)

    deploy_dinobot(env, data)
