import json
import os
import time

import numpy as np
import requests
from PIL import Image


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


def get_embeddings(image_path, url):
    url += "embeddings"
    with open(image_path, "rb") as f:
        files = {"image": f.read()}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        return parsed_response["embeddings"]
    else:
        print(response.json())


def get_embeddings_old(image_path, url):
    url += "embeddings_old"
    with open(image_path, "rb") as f:
        files = {"image": f.read()}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        return parsed_response["embeddings"]
    else:
        print(response.json())


if __name__ == "__main__":
    # before running, establish an ssh tunnel
    # in a terminal run the following command:
    # ssh -L 8080:linnet.doc.ic.ac.uk:8000 md1823@shell5.doc.ic.ac.uk
    # and log in with the college password
    try:
        os.remove("images/image1.png")
        os.remove("images/image2.png")
    except FileNotFoundError:
        pass
    image1 = "images/rgb_image_bn_0.png"
    image2 = "images/rgb_image_lv_0.png"
    url = "http://localhost:8080/"

    config = {
        "num_pairs": 8,
        "load_size": 224,
        "layer": 9,
        "facet": "key",
        "bin": True,
        "thresh": 0.05,
        "model_type": "dino_vits8",
        "stride": 4,
        "device": "cpu",
        "draw": True,
    }

    # Get correspondences
    points1, points2, img1_c, img2_c, time_taken = find_correspondences(
        image1, image2, url, config
    )
    print("Time taken:", time_taken)

    # save the images
    img1_c.save("images/image1.png")
    img2_c.save("images/image2.png")

    config["draw"] = False

    points1_a, points2_a, time_taken = find_correspondences(image1, image2, url, config)
    print("Time taken:", time_taken)

    print(np.allclose(points1, points1_a))
    print(np.allclose(points2, points2_a))

    config["draw"] = True

    # Get correspondences fast
    points1, points2, img1_c, img2_c, time_taken, num_patches, descriptor_vectors = (
        find_correspondeces_fast(image1, image2, url, config)
    )
    print("Time taken:", time_taken)

    (
        points1_2,
        points2_2,
        img1_c_2,
        img2_c_2,
        time_taken,
        num_patches,
        descriptor_vectors,
    ) = find_correspondeces_fast(
        image1, image2, url, config, num_patches, descriptor_vectors, points1
    )
    print("Time taken:", time_taken)

    # save the images
    img1_c.save("images/image1_fast.png")
    img2_c.save("images/image2_fast.png")
    img1_c_2.save("images/image1_2_fast.png")
    img2_c_2.save("images/image2_2_fast.png")

    config["draw"] = False

    points1_a, points2_a, time_taken, num_patches_a, descriptor_vectors_a = (
        find_correspondeces_fast(image1, image2, url, config)
    )
    print("Time taken:", time_taken)

    points1_2_a, points2_2_a, time_taken, num_patches_a, descriptor_vectors_a = (
        find_correspondeces_fast(
            image1, image2, url, config, num_patches_a, descriptor_vectors_a, points1_a
        )
    )
    print("Time taken:", time_taken)

    print(np.allclose(points1, points1_a))
    print(np.allclose(points2, points2_a))
    print(np.allclose(points1_2, points1_2_a))
    print(np.allclose(points2_2, points2_2_a))
    print(np.allclose(num_patches, num_patches_a))
    print(np.allclose(descriptor_vectors, descriptor_vectors_a))

    # Get embeddings
    start_time = time.time()
    embeddings = get_embeddings(image1, url)
    print("Time taken:", time.time() - start_time)
    start_time = time.time()
    embeddings2 = get_embeddings(image2, url)
    print("Time taken:", time.time() - start_time)

    # get embeddings using the other method
    start_time = time.time()
    embeddings_old = get_embeddings_old(image1, url)
    print("Time taken:", time.time() - start_time)

    start_time = time.time()
    embeddings2_old = get_embeddings_old(image2, url)
    print("Time taken:", time.time() - start_time)

    print(np.allclose(embeddings, embeddings_old))
    print(np.allclose(embeddings2, embeddings2_old))
