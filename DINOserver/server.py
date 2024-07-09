import io
import json
import os
import time
import warnings

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from .dino_vit_features.correspondences import find_correspondences, draw_correspondences
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoImageProcessor

from .dinobot_utils import extract_desc_maps, extract_descriptor_nn, extract_descriptors

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route("/correspondences_fast", methods=["POST"])
def correspondences_fast():
    if "image1" not in request.files:
        return jsonify({"error": "No image1 part"}), 400
    image1 = request.files["image1"]
    if "image2" not in request.files:
        return jsonify({"error": "No image2 part"}), 400
    image2 = request.files["image2"]
    if "args" not in request.files:
        return jsonify({"error": "No args part"}), 400
    args = json.loads(request.files["args"].read())
    num_patches = args.get("num_patches")
    descriptor_vectors = args.get("descriptor_vectors")
    points1_2d = args.get("points1_2d")
    config = args.get("config")

    try:
        # Save the images to disk
        img1 = "tmp_images/img1.png"
        img2 = "tmp_images/img2.png"
        # first make sure the directory exists
        os.makedirs(os.path.dirname(img1), exist_ok=True)
        with open(img1, "wb") as f:
            f.write(image1.read())
        with open(img2, "wb") as f:
            f.write(image2.read())

        results = correspondences_fast_backend(
            config, img1, img2, num_patches, descriptor_vectors, points1_2d
        )
        if config.get("draw", False):
            results["image1_correspondences"] = np.array(
                results["image1_correspondences"]
            ).tolist()
            results["image2_correspondences"] = np.array(
                results["image2_correspondences"]
            ).tolist()

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def correspondences_fast_backend(
    config, img1, img2, num_patches, descriptor_vectors, points1_2d, extractor=None
):
    start_time = time.time()
    if num_patches is None or descriptor_vectors is None or points1_2d is None:
        descriptor_vectors, num_patches = extract_descriptors(img1, img2, config)
        descriptor_list = extract_desc_maps([img1], config, extractor)
        key_y, key_x = extract_descriptor_nn(
            descriptor_vectors, descriptor_list[0], num_patches, config["device"]
        )
        points1_2d = [
            (int(y), int(x))
            for y, x in zip(
                np.array(key_y) * config["stride"],
                np.array(key_x) * config["stride"],
            )
        ]
    else:
        descriptor_vectors = torch.tensor(descriptor_vectors)

    descriptor_list = extract_desc_maps([img2], config, extractor)
    key_y, key_x = extract_descriptor_nn(
        descriptor_vectors, descriptor_list[0], num_patches, config["device"]
    )
    points2_2d = [
        (int(y), int(x))
        for y, x in zip(
            np.array(key_y) * config["stride"], np.array(key_x) * config["stride"]
        )
    ]

    if config.get("draw", False):
        # draw the correspondences
        image1_correspondences, image2_correspondences = draw_correspondences(
            points1_2d, points2_2d, Image.open(img1), Image.open(img2)
        )
        buffer1 = io.BytesIO()
        image1_correspondences.savefig(buffer1)
        buffer1.seek(0)
        image1_correspondences = Image.open(buffer1)
        buffer2 = io.BytesIO()
        image2_correspondences.savefig(buffer2)
        buffer2.seek(0)
        image2_correspondences = Image.open(buffer2)

        return {
            "points1": points1_2d,
            "points2": points2_2d,
            "time_taken": time.time() - start_time,
            "num_patches": num_patches,
            "descriptor_vectors": descriptor_vectors.tolist(),
            "image1_correspondences": image1_correspondences,
            "image2_correspondences": image2_correspondences,
        }

    return {
        "points1": points1_2d,
        "points2": points2_2d,
        "time_taken": time.time() - start_time,
        "num_patches": num_patches,
        "descriptor_vectors": descriptor_vectors.tolist(),
    }


@app.route("/correspondences", methods=["POST"])
def correspondences():
    # Get the image from the request
    if "image1" not in request.files:
        return jsonify({"error": "No image1 part"}), 400
    image1 = request.files["image1"]
    if "image2" not in request.files:
        return jsonify({"error": "No image2 part"}), 400
    # Get the image from the request
    image2 = request.files["image2"]

    config = json.loads(request.files["config"].read())

    try:
        # Save the images to disk
        img1 = "tmp_images/img1.png"
        img2 = "tmp_images/img2.png"
        # first make sure the directory exists
        os.makedirs(os.path.dirname(img1), exist_ok=True)
        with open(img1, "wb") as f:
            f.write(image1.read())
        with open(img2, "wb") as f:
            f.write(image2.read())

        results = correspondences_backend(config, img1, img2)
        if config.get("draw", False):
            results["image1_correspondences"] = np.array(
                results["image1_correspondences"]
            ).tolist()
            results["image2_correspondences"] = np.array(
                results["image2_correspondences"]
            ).tolist()

        return jsonify(results)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def correspondences_backend(config, img1, img2):
    start_time = time.time()
    with torch.no_grad():
        points1, points2, image1_pil, image2_pil = find_correspondences(
            img1,
            img2,
            config["num_pairs"],
            config["load_size"],
            config["layer"],
            config["facet"],
            config["bin"],
            config["thresh"],
            config["model_type"],
            config["stride"],
        )
    if config.get("draw", False):
        # draw the correspondences
        image1_correspondences, image2_correspondences = draw_correspondences(
            points1, points2, image1_pil, image2_pil
        )
        buffer1 = io.BytesIO()
        image1_correspondences.savefig(buffer1)
        buffer1.seek(0)
        image1_correspondences = Image.open(buffer1)
        buffer2 = io.BytesIO()
        image2_correspondences.savefig(buffer2)
        buffer2.seek(0)
        image2_correspondences = Image.open(buffer2)

        return {
            "points1": points1,
            "points2": points2,
            "time_taken": time.time() - start_time,
            "image1_correspondences": image1_correspondences,
            "image2_correspondences": image2_correspondences,
        }

    return {
        "points1": points1,
        "points2": points2,
        "time_taken": time.time() - start_time,
    }


@app.route("/embeddings", methods=["POST"])
def get_embeddings():
    # Get the image from the request
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    image_file = request.files["image"]
    # load this to a PIL image
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        cls_token = outputs.pooler_output
        return jsonify({"embeddings": cls_token.tolist()})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route("/embeddings_old", methods=["POST"])
def get_embeddings_old():
    # Get the image from the request
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    image_file = request.files["image"]
    # load this to a PIL image
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    try:
        dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")

        image_transforms = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        img = image_transforms(image)
        img = img.unsqueeze(0)
        with torch.no_grad():
            img_emb = dino(img)
        return jsonify({"embeddings": img_emb.tolist()})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
