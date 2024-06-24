import mediapy as media
import numpy as np
import cv2
import open3d as o3d

import sys
sys.path.append('/home/norman/dino-vit-features')

import matplotlib.pyplot as plt
import torch
from correspondences import find_correspondences, draw_correspondences
from extractor import ViTExtractor
from PIL import Image

num_pairs = 8 #@param
load_size = 480 #@param
layer = 9 #@param
facet = 'key' #@param
bin=True #@param
thresh=0.2 #@param
model_type='dino_vits8' #@param
stride = 8 #@param
patch_size = 8 #not changeable
device = "cuda"

extractor = ViTExtractor(model_type, stride, device=device)

def extract_descriptors(image_1_path, image_2_path, num_pairs = num_pairs, load_size = load_size):
    with torch.no_grad():
        points1, points2, image1_pil, image2_pil, \
        patches_xy, desc1, desc2, num_patches = find_correspondences(image_1_path, image_2_path, num_pairs, load_size, layer,
                                                                       facet, bin, thresh, model_type, stride,
                                                                       return_patches_x_y = True)
        desc1 = desc1.reshape((num_patches[0],num_patches[1],6528))
        descriptor_vectors = desc1[patches_xy[0], patches_xy[1]]
        print("num patches", num_patches)
        return patches_xy, desc1, descriptor_vectors, num_patches



def extract_desc_maps(image_paths, load_size = load_size):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    path = image_paths[0]
    if isinstance(path, str):
        pass
    else:
        paths = []
        for i in range(len(image_paths)):
            paths.append(f"image_{i}.png")
            media.write_image( f"image_{i}.png", image_paths[i])
        image_paths = paths

    descriptors_list = []
    org_images_list = []
    with torch.no_grad():
        for i in range(len(image_paths)):
            image_path = image_paths[i]

            image_batch, image_pil = extractor.preprocess(image_path, load_size)
            image_batch_transposed = np.transpose(image_batch[0], (1,2,0))

            print("image1_batch.size", image_batch.size())
            descriptors = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin)
            patched_shape = extractor.num_patches
            descriptors = descriptors.reshape((patched_shape[0],
                                patched_shape[1],
                                -1))

            descriptors_list.append(descriptors.cpu())
            image_batch_transposed = image_batch_transposed - image_batch_transposed.min()
            image_batch_transposed = image_batch_transposed/image_batch_transposed.max()
            image_batch_transposed = np.array(image_batch_transposed*255, dtype = np.uint8)
            org_images_list.append(media.resize_image(image_batch_transposed, (image_batch_transposed.shape[0]//patch_size,
                                                                   image_batch_transposed.shape[1]//patch_size)))
    return descriptors_list, org_images_list


def extract_descriptor_nn(descriptors, emb_im, patched_shape, return_heatmaps = False):
    cs_ys_list = []
    cs_xs_list = []
    cs_list = []
    cs = torch.nn.CosineSimilarity(dim=-1)
    print("emb_im shape", emb_im.shape)
    for i in range(len(descriptors)):
        cs_i = cs(descriptors[i].cuda(), emb_im.cuda())
        print("cs_i.shape", cs_i.shape)
        cs_i = cs_i.reshape((-1))
        cs_i_y = cs_i.argmax().cpu()//patched_shape[1]
        cs_i_x = cs_i.argmax().cpu()%patched_shape[1]

        cs_ys_list.append(int(cs_i_y))
        cs_xs_list.append(int(cs_i_x))

        cs_list.append(np.array(cs_i.cpu()).reshape(patched_shape))

        cs_i = cs_i.reshape(patched_shape)
    if return_heatmaps:
        return cs_ys_list, cs_xs_list, cs_list
    return cs_ys_list, cs_xs_list


def draw_keypoints(image, key_y, key_x, colors):
    #assert len(key_y) == len(key_x)
    canvas = np.zeros((image.shape[0],image.shape[1],3))
    if len(image.shape) < 3 or image.shape[-1] == 1:
        canvas[:,:,0] = image
    else:
        canvas = np.array(image)
    for i in range(len(key_y)):
        color = colors[i]
        canvas = canvas.astype(np.uint8)
        canvas[key_y[i],key_x[i],:] = np.array(color)
    return canvas




def project_in_3d(points, depth, intrinsics_mat):
  x3d, y3d, z3d = [], [], []

  for p in points:

    u, v = p

    #v*=1.285
    #u*=1.285

    v = int(v)
    u = int(u)

    #u+=120

    #print(u,v)

    fx = intrinsics_mat[0,0]
    fy = intrinsics_mat[1,1]

    cx = intrinsics_mat[0,2]
    cy = intrinsics_mat[1,2]

    d = depth[u,v]

    z = d / 1000

    x = (v - cx) * z / fx

    y = (u - cy) * z / fy

    x3d.append(x)
    y3d.append(y)
    z3d.append(z)
  return x3d, y3d, z3d


def clean_zero_kps(k):
    k = np.array(k)
    print("shape k", k.shape)
    non_zero_ids = np.arange(0, k.shape[2])
    for k_i in k:
        k_i_x = k_i[0]
        k_i_z = k_i[2]
        
        zero_ids = np.argwhere(np.array(k_i_x) ==  0)
        for z_i in zero_ids:
            non_zero_ids = np.delete(non_zero_ids, np.argwhere(non_zero_ids == z_i[0]))

          
        far_ids = np.argwhere(np.array(k_i_z) >  1.5)
        for f_i in far_ids:
            non_zero_ids = np.delete(non_zero_ids, np.argwhere(non_zero_ids == f_i[0]))
    k = k[:,:,non_zero_ids]
    return k, non_zero_ids


def extract_data_for_training(paths, intrinsics_mat, num_pairs = 15, clean = True, show = True, matches_between_same_image = False):
    assert len(paths) > 2

    #Find which keypoints to track
    video1 = np.load(f"{paths[0]}/rgbs_subsampled_30.npy")
    video2 = np.load(f"{paths[1]}/rgbs_subsampled_30.npy")
    if matches_between_same_image:
      video1 = np.load(f"{paths[1]}/rgbs_subsampled_30.npy")
        

    media.write_image("a.png", video1[0])
    media.write_image("b.png", video2[0])

    patches_xy, desc_map, descriptor_vectors, num_patches = extract_descriptors("a.png",
                                                                            "b.png", 
                                                                            load_size = load_size, 
                                                                            num_pairs = num_pairs)
    
    kps, jts = [], []
    colors = [np.random.randint(0,255,size=3) for i in range(num_pairs)]

    for path in paths:
        video = np.load(f"{path}/rgbs_subsampled_30.npy")
        descriptors_list, org_images_list = extract_desc_maps(video[0])
        key_y, key_x = extract_descriptor_nn(descriptor_vectors, emb_im=descriptors_list[0], patched_shape=num_patches, return_heatmaps = False)
        points = [(y,x) for y, x in zip(np.array(key_y)*stride, np.array(key_x)*stride)]
        depth = np.load(f"{path}/depths_subsampled_30.npy")[0]
        x3d, y3d, z3d = project_in_3d(points, depth, intrinsics_mat)


        #Display

        if show:
          i=0
          image_w_keypoints = draw_keypoints(np.array(org_images_list[i], dtype = np.uint8), key_y, key_x,  colors)
          media.show_image(media.resize_image(image_w_keypoints, (image_w_keypoints.shape[0]*4,image_w_keypoints.shape[1]*4)))


        kps.append([x3d, y3d, z3d])
        jts.append(np.load(f"{path}/hand_joints_kpts_3d.npy"))

    if clean:
      kps, non_zero_ids = clean_zero_kps(kps)
    else:
      non_zero_ids = np.arange(0, num_pairs)
    
    return kps, jts, descriptor_vectors, num_patches, non_zero_ids, colors

"""Example use of the above:
k, j, descriptors, patches, non_zero_ids, colors = extract_data_for_training(paths = [f"/home/norman/Downloads/fuck_kinestheticc_teaching/blue_cup_on_plate_{i}/" for i in range(0,4)],
                                intrinsics_mat=intrinsics_mat, num_pairs=40)"""

def keypoints_using_descriptors(image, depth, descriptors, patches, non_zero_ids, stride, intrinsics_mat):
    descriptors_list, org_images_list = extract_desc_maps([image])
    key_y, key_x = extract_descriptor_nn(descriptors, emb_im=descriptors_list[0], patched_shape=patches, return_heatmaps = False)
    key_y = np.array(key_y)[non_zero_ids]
    key_x = np.array(key_x)[non_zero_ids]
    points = [(y,x) for y, x in zip(np.array(key_y)*stride, np.array(key_x)*stride)]
    x3d, y3d, z3d = project_in_3d(points, depth, intrinsics_mat)


def clean_reshape(k):
    clean_k_reshape = []
    for i in range(len(k)):
        new = []
        for j in range(len(k[0,0])):
            new.append([k[i,0,j], k[i,1,j], k[i,2,j]])
        clean_k_reshape.append(new)
    return clean_k_reshape



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
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t