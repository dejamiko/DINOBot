import time
import warnings

import torch
from dino_vit_features.correspondences import find_correspondences

warnings.filterwarnings("ignore")

# Hyperparameters for DINO correspondences extraction
num_pairs = 8
load_size = 224
layer = 9
facet = 'key'
bin = True
thresh = 0.05
model_type = 'dino_vits8'
stride = 4

if __name__ == "__main__":
    start_time = time.time()
    rgb_live = "sample_images/rgb_image_lv_0.png"
    rgb_bn = "sample_images/rgb_image_bn_0.png"
    with torch.no_grad():
        points1, points2, image1_pil, image2_pil = find_correspondences(
            rgb_live,
            rgb_bn,
            num_pairs,
            load_size,
            layer,
            facet,
            bin,
            thresh,
            model_type,
            stride
        )
        print(points1, points2)
    print(f"Time taken: {time.time() - start_time} seconds")