import mediapy as media
import numpy as np

from config import Config

import torch
from correspondences import find_correspondences
from dino_vit_features.extractor import ViTExtractor

config = Config()


extractor = ViTExtractor(config.model_type, config.stride, device=config.device)


def extract_descriptors(
    image_1_path, image_2_path, num_pairs=config.num_pairs, load_size=config.load_size
):
    with torch.no_grad():
        _, _, _, _, patches_xy, desc1, _, num_patches = find_correspondences(
            image_1_path,
            image_2_path,
            num_pairs,
            load_size,
            config.layer,
            config.facet,
            config.bin,
            config.thresh,
            config.model_type,
            config.stride,
            return_patches_x_y=True,
        )
        desc1 = desc1.reshape((num_patches[0], num_patches[1], 6528))
        descriptor_vectors = desc1[patches_xy[0], patches_xy[1]]
        return patches_xy, desc1, descriptor_vectors, num_patches


def extract_desc_maps(image_paths, load_size=config.load_size):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    path = image_paths[0]
    if isinstance(path, str):
        pass
    else:
        paths = []
        for i in range(len(image_paths)):
            paths.append(f"image_{i}.png")
            media.write_image(f"image_{i}.png", image_paths[i])
        image_paths = paths

    descriptors_list = []
    org_images_list = []
    with torch.no_grad():
        for i in range(len(image_paths)):
            image_path = image_paths[i]

            image_batch, image_pil = extractor.preprocess(image_path, load_size)
            image_batch_transposed = np.transpose(image_batch[0], (1, 2, 0))

            descriptors = extractor.extract_descriptors(
                image_batch.to(config.device), config.layer, config.facet, config.bin
            )
            patched_shape = extractor.num_patches
            descriptors = descriptors.reshape((patched_shape[0], patched_shape[1], -1))

            descriptors_list.append(descriptors.cpu())
            image_batch_transposed = (
                image_batch_transposed - image_batch_transposed.min()
            )
            image_batch_transposed = (
                image_batch_transposed / image_batch_transposed.max()
            )
            image_batch_transposed = np.array(
                image_batch_transposed * 255, dtype=np.uint8
            )
            org_images_list.append(
                media.resize_image(
                    image_batch_transposed,
                    (
                        image_batch_transposed.shape[0] // config.patch_size,
                        image_batch_transposed.shape[1] // config.patch_size,
                    ),
                )
            )
    return descriptors_list, org_images_list


def extract_descriptor_nn(descriptors, emb_im, patched_shape, return_heatmaps=False):
    cs_ys_list = []
    cs_xs_list = []
    cs_list = []
    cs = torch.nn.CosineSimilarity(dim=-1)
    for i in range(len(descriptors)):
        cs_i = cs(descriptors[i].cuda(), emb_im.cuda())
        cs_i = cs_i.reshape((-1))
        cs_i_y = cs_i.argmax().cpu() // patched_shape[1]
        cs_i_x = cs_i.argmax().cpu() % patched_shape[1]

        cs_ys_list.append(int(cs_i_y))
        cs_xs_list.append(int(cs_i_x))

        cs_list.append(np.array(cs_i.cpu()).reshape(patched_shape))
    if return_heatmaps:
        return cs_ys_list, cs_xs_list, cs_list
    return cs_ys_list, cs_xs_list
