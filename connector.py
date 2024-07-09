from DINOserver.dino_vit_features.extractor import ViTExtractor

from DINOserver.client import find_correspondences as find_correspondences_server
from DINOserver.client import (
    find_correspondences_fast as find_correspondences_fast_server,
)
from DINOserver.server import correspondences_backend, correspondences_fast_backend
from config import Config


def find_correspondences(image_path1, image_path2, url, dino_config):
    if dino_config["run_locally"]:
        return find_correspondences_locally(image_path1, image_path2, dino_config)
    else:
        return find_correspondences_server(image_path1, image_path2, url, dino_config)


def find_correspondences_locally(image_path1, image_path2, dino_config):
    results = correspondences_backend(dino_config, image_path1, image_path2)
    if dino_config["draw"]:
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
    extractor = ViTExtractor(
        dino_config["model_type"], dino_config["stride"], device=dino_config["device"]
    )
    results = correspondences_fast_backend(
        dino_config,
        image_path1,
        image_path2,
        num_patches,
        descriptor_vectors,
        points1,
        extractor,
    )
    if dino_config["draw"]:
        return (
            results["points1"],
            results["points2"],
            results["time_taken"],
            results["num_patches"],
            results["descriptor_vectors"],
            results["image1_correspondences"],
            results["image2_correspondences"],
        )
    return (
        results["points1"],
        results["points2"],
        results["time_taken"],
        results["num_patches"],
        results["descriptor_vectors"],
    )


def get_correspondences(config, counter, rgb_bn_path, rgb_live_path, **kwargs):
    results = {}
    arguments = {
        "image_path1": rgb_bn_path,
        "image_path2": rgb_live_path,
        "url": config.BASE_URL,
        "dino_config": config.get_dino_config(),
    }
    if config.USE_FAST_CORRESPONDENCES:
        arguments["num_patches"] = kwargs.get("num_patches", None)
        arguments["descriptor_vectors"] = kwargs.get("descriptor_vectors", None)
        arguments["points1"] = kwargs.get("points1_2d", None)
        if counter % config.RECOMPUTE_EVERY == 0:
            arguments["num_patches"] = None
            arguments["descriptor_vectors"] = None
            arguments["points1"] = None

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
        im_1_c = res[-2]
        im_2_c = res[-1]

        im_1_c.save(f"images/image1_correspondences_{counter}.png")
        im_2_c.save(f"images/image2_correspondences_{counter}.png")

    return results


if __name__ == "__main__":
    config = Config()
    config.RUN_LOCALLY = False
    config.USE_FAST_CORRESPONDENCES = False
    image1 = "images/rgb_image_bn_0.png"
    image2 = "images/rgb_image_live_0.png"

    results = get_correspondences(config, 0, image1, image2)
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert len(results) == 3
    print("Time taken", results["time_taken"])

    config.DRAW_CORRESPONDENCES = True
    print("we're before")
    results = get_correspondences(config, 0, image1, image2)
    print("we're after")
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert len(results) == 3
    print("Time taken", results["time_taken"])

    config.USE_FAST_CORRESPONDENCES = True
    config.DRAW_CORRESPONDENCES = False

    results = get_correspondences(config, 0, image1, image2)
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "num_patches" in results
    assert "descriptor_vectors" in results
    assert len(results) == 5
    print("Time taken", results["time_taken"])

    config.DRAW_CORRESPONDENCES = True
    print("we're before")
    results = get_correspondences(config, 0, image1, image2)
    print("we're after")
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "num_patches" in results
    assert "descriptor_vectors" in results
    assert len(results) == 5
    print("Time taken", results["time_taken"])

    config.DRAW_CORRESPONDENCES = False
    results = get_correspondences(
        config,
        2,
        image1,
        image2,
        points1_2d=results["points1_2d"],
        num_patches=results["num_patches"],
        descriptor_vectors=results["descriptor_vectors"],
    )
    assert "points1_2d" in results
    assert "points2_2d" in results
    assert "time_taken" in results
    assert "num_patches" in results
    assert "descriptor_vectors" in results
    assert len(results) == 5
    print("Time taken", results["time_taken"])
