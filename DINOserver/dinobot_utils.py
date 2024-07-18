import torch

from .correspondences import find_correspondences
from .dino_vit_features.extractor import ViTExtractor


def extract_descriptors(image_1_path, image_2_path, config):
    with torch.no_grad():
        _, _, _, _, patches_xy, desc1, _, num_patches = find_correspondences(
            image_1_path,
            image_2_path,
            config["num_pairs"],
            config["load_size"],
            config["layer"],
            config["facet"],
            config["bin"],
            config["thresh"],
            config["model_type"],
            config["stride"],
            return_patches_x_y=True,
        )
        desc1 = desc1.reshape((num_patches[0], num_patches[1], 6528))
        descriptor_vectors = desc1[patches_xy[0], patches_xy[1]]
        return descriptor_vectors, num_patches


def extract_desc_maps(image_paths, config, extractor=None):
    if extractor is None:
        extractor = ViTExtractor(
            config["model_type"], config["stride"], device=config["device"]
        )
    descriptors_list = []
    with torch.no_grad():
        for i in range(len(image_paths)):
            image_path = image_paths[i]

            image_batch, _ = extractor.preprocess(image_path, config["load_size"])

            descriptors = extractor.extract_descriptors(
                image_batch.to(config["device"]),
                config["layer"],
                config["facet"],
                config["bin"],
            )
            patched_shape = extractor.num_patches
            descriptors = descriptors.reshape((patched_shape[0], patched_shape[1], -1))

            descriptors_list.append(descriptors.cpu())
    torch.cuda.empty_cache()
    return descriptors_list


def extract_descriptor_nn(descriptors, emb_im, patched_shape, device):
    cs_ys_list = []
    cs_xs_list = []
    cs = torch.nn.CosineSimilarity(dim=-1)
    for i in range(len(descriptors)):
        cs_i = cs(descriptors[i].to(device), emb_im.to(device))
        cs_i = cs_i.reshape((-1))
        cs_i_y = cs_i.argmax().cpu() // patched_shape[1]
        cs_i_x = cs_i.argmax().cpu() % patched_shape[1]

        cs_ys_list.append(int(cs_i_y))
        cs_xs_list.append(int(cs_i_x))
    return cs_ys_list, cs_xs_list
