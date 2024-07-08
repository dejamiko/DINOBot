import json

import numpy as np
import requests
from PIL import Image

from DINOserver.server import correspondences_backend, correspondences_fast_backend


def find_correspondences(image_path1, image_path2, url, dino_config):
    if dino_config["run_locally"]:
        return find_correspondences_locally(
            image_path1, image_path2, dino_config
        )
    else:
        return find_correspondences_server(
            image_path1, image_path2, url, dino_config
        )


def find_correspondences_locally(image_path1, image_path2, dino_config):
    results = correspondences_backend(dino_config, image_path1, image_path2)
    if dino_config.DRAW_CORRESPONDENCES:
        return (
            results["points1"],
            results["points2"],
            results["time_taken"],
            results["image1_correspondences"],
            results["image2_correspondences"],
        )
    return (
        results["points1"],
        results["points2"],
        results["time_taken"],
    )


def find_correspondences_server(image_path1, image_path2, url, dino_config):
    url += "correspondences"
    with open(image_path1, "rb") as f:
        files = {"image1": f.read()}
    with open(image_path2, "rb") as f:
        files["image2"] = f.read()
    files["config"] = json.dumps(dino_config)
    response = requests.post(url, files=files)
    if response.status_code == 200:
        parsed_response = response.json()
        if dino_config["draw"]:
            image1_correspondences = Image.fromarray(
                np.array(parsed_response["image1_correspondences"], dtype="uint8")
            )
            image2_correspondences = Image.fromarray(
                np.array(parsed_response["image2_correspondences"], dtype="uint8")
            )
            return (
                parsed_response["points1"],
                parsed_response["points2"],
                parsed_response["time_taken"],
                image1_correspondences,
                image2_correspondences,
            )
        return (
            parsed_response["points1"],
            parsed_response["points2"],
            parsed_response["time_taken"],
        )
    else:
        print(response.json())


def find_correspondences_fast(
    image_path1,
    image_path2,
    url,
    dino_config,
    num_patches=None,
    descriptor_vectors=None,
    points1=None,
):
    if dino_config["run_locally"]:
        return find_correspondences_fast_locally(
            image_path1,
            image_path2,
            dino_config,
            num_patches,
            descriptor_vectors,
            points1,
        )
    else:
        return find_correspondences_fast_server(
            image_path1,
            image_path2,
            url,
            dino_config,
            num_patches,
            descriptor_vectors,
            points1,
        )


def find_correspondences_fast_locally(
    image_path1,
    image_path2,
    dino_config,
    num_patches=None,
    descriptor_vectors=None,
    points1=None,
):
    results = correspondences_fast_backend(
        dino_config, image_path1, image_path2, num_patches, descriptor_vectors, points1
    )
    if dino_config.DRAW_CORRESPONDENCES:
        return (
            results["points1"],
            results["points2"],
            results["time_taken"],
            results["image1_correspondences"],
            results["image2_correspondences"],
        )
    return (
        results["points1"],
        results["points2"],
        results["time_taken"],
    )


def find_correspondences_fast_server(
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
                parsed_response["time_taken"],
                parsed_response["num_patches"],
                parsed_response["descriptor_vectors"],
                image1_correspondences,
                image2_correspondences,
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


def get_correspondences(config, counter, rgb_bn_path, rgb_live_path, **kwargs):
    results = {}
    arguments = {
        "image_path1": rgb_bn_path,
        "image_path2": rgb_live_path,
        "url": config.BASE_URL,
        "dino_config": config.get_dino_config(),
    }
    if config.USE_FAST_CORRESPONDENCES and counter % config.RECOMPUTE_EVERY == 0:
        arguments["num_patches"] = kwargs["num_patches"]
        arguments["descriptor_vectors"] = (kwargs["descriptor_vectors"],)
        arguments["points1"] = kwargs["points1_2d"]

    if config.USE_FAST_CORRESPONDENCES:
        res = find_correspondences_fast(**arguments)
        results["num_patches"] = res[3]
        results["descriptor_vectors"] = res[4]
    else:
        res = find_correspondences(**arguments)
        results["time_taken"] = res[2]
    results["points1_2d"] = res[0]
    results["points2_2d"] = res[1]
    results["time_taken"] = res[2]

    if config.DRAW_CORRESPONDENCES:
        im_1_c = results[-2]
        im_2_c = results[-1]

        im_1_c.save(f"images/image1_correspondences_{counter}.png")
        im_2_c.save(f"images/image2_correspondences_{counter}.png")

    return results
