import cv2
import numpy as np
import torch
from src.utils import utils_v2v

def normals(depthmap, normalize=True, keep_dims=True):
    """Calculate depth normals as normals = gF(x,y,z) = (-dF/dx, -dF/dy, 1)

    Args:
        depthmap (np.ndarray): depth map of any dtype, single channel, len(depthmap.shape) == 3
        normalize (bool, optional): if True, normals will be normalized to have unit-magnitude
            Default: True
        keep_dims (bool, optional):
            if True, normals shape will be equals to depthmap shape,
            if False, normals shape will be smaller than depthmap shape.
            Default: True

    Returns:
        Depth normals

    """
    depthmap = np.asarray(depthmap, np.float32)

    if keep_dims is True:
        mask = depthmap != 0
    else:
        mask = depthmap[1:-1, 1:-1] != 0

    if keep_dims is True:
        normals = np.zeros((depthmap.shape[0], depthmap.shape[1], 3), dtype=np.float32)
        normals[1:-1, 1:-1, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[1:-1, 1:-1, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    else:
        normals = np.zeros((depthmap.shape[0] - 2, depthmap.shape[1] - 2, 3), dtype=np.float32)
        normals[:, :, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[:, :, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    normals[:, :, 2] = 1

    normals[~mask] = [0, 0, 0]

    if normalize:
        # div = np.sqrt(np.sum(normals[mask] ** 2, axis=-1))
        div = np.linalg.norm(normals[mask], ord=2, axis=-1, keepdims=True).repeat(3, axis=-1) + 1e-12
        normals[mask] /= div

    return normals


def uint8_depth_map(depth_map):
    """Cast depth map to np.uint8"""
    return depth_map.astype(np.uint8)


def show(depth_map, mode=0):
    """Equalize depth map for visualization"""
    if mode == 0:
        return uint8_depth_map(depth_map)
    elif mode == 1:
        return ((depth_map - depth_map.min()).astype(np.float32) / depth_map.max() * 255.).astype(np.uint8)


def save(depth_map, path, mode=0):
    """Save depth map"""
    if not path.endswith('.png'):
        raise ValueError
    if mode == 0:
        cv2.imwrite(path, uint8_depth_map(depth_map))
    elif mode == 1:
        cv2.imwrite(path, depth_map(depth_map))


def pointcloud_slow(depth_map, calib_params):
    """Generate pointcloud from depth map, slow version"""
    points = np.zeros((np.prod(depth_map.shape), 3), dtype=np.float32)
    index = 0
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            points[index] = [
                ((c + 1) - calib_params["cx"]) * img[r, c] / calib_params["fx"],
                (calib_params["cy"] - (r + 1)) * img[r, c] / calib_params["fy"],
                img[r, c]
            ]
            index += 1
    return points


def pointcloud(depth_map, fx, fy, cx=None, cy=None):
    """Generate pointcloud from depth map"""
    points = utils_v2v.depthmap2points(depth_map, fx, fy, cx, cy)
    points = points.reshape((-1, 3))
    return points


def pointcloud_normalization(ptc1):
    """Normalize pointcloud as out=(input-input.mean())/input.std()"""
    if ptc1[:, 0].std() == 0:
        ptc1[:, 0] = ptc1[:, 0] - ptc1[:, 0].mean()
    else:
        ptc1[:, 0] = (ptc1[:, 0] - ptc1[:, 0].mean()) / ptc1[:, 0].std()
    if ptc1[:, 1].std() == 0:
        ptc1[:, 1] = ptc1[:, 1] - ptc1[:, 1].mean()
    else:
        ptc1[:, 1] = (ptc1[:, 1] - ptc1[:, 1].mean()) / ptc1[:, 1].std()
    if ptc1[:, 2].std() == 0:
        ptc1[:, 2] = ptc1[:, 2] - ptc1[:, 2].mean()
    else:
        ptc1[:, 2] = (ptc1[:, 2] - ptc1[:, 2].mean()) / ptc1[:, 2].std()
    return ptc1


def depth_to_world(kpt, depth_map):
    """Transform kpt from 2D to 3D Real World Coordinates (RWC) for ITOP Dataset

    Args:
        kpt (np.ndarray): Array containing keypoints to transform
        depthmap (np.ndarray): single channel depth map, len(depthmap.shape) == 2

    Returns:
        np.ndarray: Converted keypoints

    """
    new_kpt = np.zeros((kpt.shape[0], 3))
    z = np.zeros(len(kpt), dtype=np.float32)
    n = 4       # Patch size
    for i, k in enumerate(kpt.astype(np.int)):
        if k[0] <= 0 and k[1] <= 0:
            z[i] = 0
        else:
            patch = depth_map[min(max(0, int(k[1]) - n), depth_map.shape[0] - n):
                              min(max(0, int(k[1])) + n + 1, depth_map.shape[0]),
                              min(max(0, int(k[0]) - n), depth_map.shape[1] - n):
                              min(max(0, int(k[0])) + n + 1, depth_map.shape[1])]
            z[i] = np.median(patch[patch != 0]) if np.mean(patch) != 0 else 0
    new_kpt[:, 0] = (kpt[:, 0] - 160) * 0.0035 * z
    new_kpt[:, 1] = -(kpt[:, 1] - 120) * 0.0035 * z
    new_kpt[:, 2] = z
    return new_kpt

def depth_to_world_single(kpt, depth):
    new_kpt = np.zeros((kpt.shape[0], 3))
    new_kpt[..., 0] = (kpt[..., 0] - 160) * 0.0035 * depth
    new_kpt[..., 1] = -(kpt[..., 1] - 120) * 0.0035 * depth
    new_kpt[..., 2] = depth
    return new_kpt

def zaxis_to_world(kpt: torch.Tensor):
    """Transform kpt from 2D+Z to 3D Real World Coordinates (RWC) for ITOP Dataset

    Args:
        kpt (np.ndarray): Array containing keypoints to transform

    Returns:
        np.ndarray: Converted keypoints

    """
    tmp = kpt.clone()
    tmp[..., 0] = (tmp[..., 0].clone() - 160) * 0.0035 * tmp[..., 2].clone()
    tmp[..., 1] = -(tmp[..., 0].clone() - 120) * 0.0035 * tmp[..., 2].clone()
    return tmp

def zaxis_to_world_np(kpt: np.ndarray):
    tmp = kpt.copy()
    tmp[..., 0] = (tmp[..., 0].copy() - 160) * 0.0035 * tmp[..., 2].copy()
    tmp[..., 1] = -(tmp[..., 1].copy() - 120) * 0.0035 * tmp[..., 2].copy()
    return tmp

def world_to_depth(kpt_3d):
    """Transform kpt from 3D Real World Coordinates (RWC) to 2D for ITOP Dataset

    Args:
        kpt (np.ndarray): Array containing keypoints to transform

    Returns:
        np.ndarray: Converted keypoints

    """
    assert kpt_3d.shape[-1] == 3
    # Working for bot full patch or single sample
    if len(kpt_3d.shape) == 3:
        tmp = np.zeros((kpt_3d.shape[0], kpt_3d.shape[1], 2))
        tmp[..., 0] = kpt_3d[..., 0] / (0.0035 * kpt_3d[..., 2] + 1e-9) + 160
        tmp[..., 1] = -kpt_3d[..., 1] / (0.0035 * kpt_3d[..., 2] + 1e-9) + 120
    elif len(kpt_3d.shape) == 2:
        tmp = np.zeros((kpt_3d.shape[0], 2))
        tmp[:, 0] = kpt_3d[:, 0] / (0.0035 * kpt_3d[:, 2] + 1e-9) + 160
        tmp[:, 1] = -kpt_3d[:, 1] / (0.0035 * kpt_3d[:, 2] + 1e-9) + 120
    else:
        raise NotImplementedError(
            "Conversion world to depth coordinates not implemented yet, shape len: {}".format(len(kpt_3d.shape)))
    return tmp
