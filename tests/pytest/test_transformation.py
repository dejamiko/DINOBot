import numpy as np

from config import Config
from dinobot import find_transformation


def test_find_transformation():
    # Adapted from https://github.com/nghiaho12/rigid_transform_3D
    config = Config()
    # Test with random data
    # Random rotation and translation
    R = np.random.rand(3, 3)
    t = np.random.rand(3, 1)

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt

    # remove reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U @ Vt

    # number of points
    n = 10

    A = np.random.rand(3, n)
    B = R @ A + t

    # Recover R and t
    ret_R, ret_t = find_transformation(A.T, B.T, config)

    # make ret_t a (3,1) array
    ret_t = ret_t.reshape(-1, 1)

    # Compare the recovered R and t with the original
    B2 = (ret_R @ A) + ret_t

    # Find the root mean squared error
    err = B2 - B
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err / n)

    assert rmse < 1e-5
