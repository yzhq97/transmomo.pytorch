from scipy.ndimage import gaussian_filter1d
import numpy as np
import json
import os
import torch


def preprocess_test(motion, meanpose, stdpose, unit=128):

    motion = motion * unit

    motion[1, :, :] = (motion[2, :, :] + motion[5, :, :]) / 2
    motion[8, :, :] = (motion[9, :, :] + motion[12, :, :]) / 2

    start = motion[8, :, 0]

    motion = localize_motion(motion)
    motion = normalize_motion(motion, meanpose, stdpose)

    return motion, start


def postprocess(motion, meanpose, stdpose, unit=128, start=None):

    motion = motion.detach().cpu().numpy()[0].reshape(-1, 2, motion.shape[-1])
    motion = normalize_motion_inv(motion, meanpose, stdpose)
    motion = globalize_motion(motion, start=start)
    motion = motion / unit

    return motion


def preprocess_mixamo(motion, unit=128):

    _, D, _ = motion.shape
    horizontal_dim = 0
    vertical_dim = D - 1

    motion[1, :, :] = (motion[2, :, :] + motion[5, :, :]) / 2
    motion[8, :, :] = (motion[9, :, :] + motion[12, :, :]) / 2

    # rotate 180
    motion[:, horizontal_dim, :] = - motion[:, horizontal_dim, :]
    motion[:, vertical_dim, :] = - motion[:, vertical_dim, :]

    motion = motion * unit

    return motion


def rotate_motion_3d(motion3d, change_of_basis):

    if change_of_basis is not None: motion3d = change_of_basis @ motion3d

    return motion3d


def limb_scale_motion_2d(motion2d, global_range, local_range):

    global_scale = global_range[0] + np.random.random() * (global_range[1] - global_range[0])
    local_scales = local_range[0] + np.random.random([8]) * (local_range[1] - local_range[0])
    motion_scale = scale_limbs(motion2d, global_scale, local_scales)

    return motion_scale


def localize_motion(motion):
    """
    Motion fed into our network is the local motion, i.e. coordinates relative to the hip joint.
    This function removes global motion of the hip joint, and instead represents global motion with velocity
    """

    D = motion.shape[1]

    # subtract centers to local coordinates
    centers = motion[8, :, :] # N_dim x T
    motion = motion - centers

    # adding velocity
    translation = centers[:, 1:] - centers[:, :-1]
    velocity = np.c_[np.zeros((D, 1)), translation]
    velocity = velocity.reshape(1, D, -1)
    motion = np.r_[motion[:8], motion[9:], velocity]
    # motion_proj = np.r_[motion_proj[:8], motion_proj[9:]]

    return motion


def globalize_motion(motion, start=None, velocity=None):
    """
    inverse process of localize_motion
    """

    if velocity is None: velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += start.reshape([2, 1])

    return motion_inv + centers.reshape((1, 2, -1))


def normalize_motion(motion, meanpose, stdpose):
    """
    :param motion: (J, 2, T)
    :param meanpose: (J, 2)
    :param stdpose: (J, 2)
    :return:
    """
    if motion.shape[1] == 2 and meanpose.shape[1] == 3:
        meanpose = meanpose[:, [0, 2]]
    if motion.shape[1] == 2 and stdpose.shape[1] == 3:
        stdpose = stdpose[:, [0, 2]]
    return (motion - meanpose[:, :, np.newaxis]) / stdpose[:, :, np.newaxis]


def normalize_motion_inv(motion, meanpose, stdpose):
    if motion.shape[1] == 2 and meanpose.shape[1] == 3:
        meanpose = meanpose[:, [0, 2]]
    if motion.shape[1] == 2 and stdpose.shape[1] == 3:
        stdpose = stdpose[:, [0, 2]]
    return motion * stdpose[:, :, np.newaxis] + meanpose[:, :, np.newaxis]


def get_change_of_basis(motion3d, angles=None):
    """
    Get the unit vectors for local rectangular coordinates for given 3D motion
    :param motion3d: numpy array. 3D motion from 3D joints positions, shape (nr_joints, 3, nr_frames).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
    """
    # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
    horizontal = (motion3d[2] - motion3d[5] + motion3d[9] - motion3d[12]) / 2
    horizontal = np.mean(horizontal, axis=1)
    horizontal = horizontal / np.linalg.norm(horizontal)
    local_z = np.array([0, 0, 1])
    local_y = np.cross(horizontal, local_z)  # bugs!!!, horizontal and local_Z may not be perpendicular
    local_y = local_y / np.linalg.norm(local_y)
    local_x = np.cross(local_y, local_z)
    local = np.stack([local_x, local_y, local_z], axis=0)

    if angles is not None:
        local = rotate_basis(local, angles)

    return local


def rotate_basis(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.

    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return:
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def get_foot_vel(batch_motion, foot_idx):
    return batch_motion[:, foot_idx, 1:] - batch_motion[:, foot_idx, :-1] + batch_motion[:, -2:, 1:].repeat(1, 2, 1)


def get_limbs(motion):
    J, D, T = motion.shape
    limbs = np.zeros([14, D, T])
    limbs[0] = motion[0] - motion[1] # neck
    limbs[1] = motion[2] - motion[1] # r_shoulder
    limbs[2] = motion[3] - motion[2] # r_arm
    limbs[3] = motion[4] - motion[3] # r_forearm
    limbs[4] = motion[5] - motion[1] # l_shoulder
    limbs[5] = motion[6] - motion[5] # l_arm
    limbs[6] = motion[7] - motion[6] # l_forearm
    limbs[7] = motion[1] - motion[8] # spine
    limbs[8] = motion[9] - motion[8] # r_pelvis
    limbs[9] = motion[10] - motion[9] # r_thigh
    limbs[10] = motion[11] - motion[10] # r_shin
    limbs[11] = motion[12] - motion[8] # l_pelvis
    limbs[12] = motion[13] - motion[12] # l_thigh
    limbs[13] = motion[14] - motion[13] # l_shin
    return limbs


def scale_limbs(motion, global_scale, local_scales):
    """
    :param motion: joint sequence [J, 2, T]
    :param local_scales: 8 numbers of scales
    :return: scaled joint sequence
    """

    limb_dependents = [
        [0],
        [2, 3, 4],
        [3, 4],
        [4],
        [5, 6, 7],
        [6, 7],
        [7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [9, 10, 11],
        [10, 11],
        [11],
        [12, 13, 14],
        [13, 14],
        [14]
    ]

    limbs = get_limbs(motion)
    scaled_limbs = limbs.copy() * global_scale
    scaled_limbs[0] *= local_scales[0]
    scaled_limbs[1] *= local_scales[1]
    scaled_limbs[2] *= local_scales[2]
    scaled_limbs[3] *= local_scales[3]
    scaled_limbs[4] *= local_scales[1]
    scaled_limbs[5] *= local_scales[2]
    scaled_limbs[6] *= local_scales[3]
    scaled_limbs[7] *= local_scales[4]
    scaled_limbs[8] *= local_scales[5]
    scaled_limbs[9] *= local_scales[6]
    scaled_limbs[10] *= local_scales[7]
    scaled_limbs[11] *= local_scales[5]
    scaled_limbs[12] *= local_scales[6]
    scaled_limbs[13] *= local_scales[7]

    delta = scaled_limbs - limbs

    scaled_motion = motion.copy()
    scaled_motion[limb_dependents[7]] += delta[7] # spine
    scaled_motion[limb_dependents[1]] += delta[1] # r_shoulder
    scaled_motion[limb_dependents[4]] += delta[4] # l_shoulder
    scaled_motion[limb_dependents[2]] += delta[2] # r_arm
    scaled_motion[limb_dependents[5]] += delta[5] # l_arm
    scaled_motion[limb_dependents[3]] += delta[3] # r_forearm
    scaled_motion[limb_dependents[6]] += delta[6] # l_forearm
    scaled_motion[limb_dependents[0]] += delta[0] # neck
    scaled_motion[limb_dependents[8]] += delta[8] # r_pelvis
    scaled_motion[limb_dependents[11]] += delta[11] # l_pelvis
    scaled_motion[limb_dependents[9]] += delta[9]  # r_thigh
    scaled_motion[limb_dependents[12]] += delta[12]  # l_thigh
    scaled_motion[limb_dependents[10]] += delta[10]  # r_shin
    scaled_motion[limb_dependents[13]] += delta[13]  # l_shin


    return scaled_motion


def get_limb_lengths(x):
    _, dims, _ = x.shape
    if dims == 2:
        limbs = np.max(np.linalg.norm(get_limbs(x), axis=1), axis=-1)
        limb_lengths = np.array([
            limbs[0],                  # neck
            max(limbs[1], limbs[4]),   # shoulders
            max(limbs[2], limbs[5]),   # arms
            max(limbs[3], limbs[6]),   # forearms
            limbs[7],                  # spine
            max(limbs[8], limbs[11]),  # pelvis
            max(limbs[9], limbs[12]),  # thighs
            max(limbs[10], limbs[13])  # shins
        ])
    else:
        limbs = np.mean(np.linalg.norm(get_limbs(x), axis=1), axis=-1)
        limb_lengths = np.array([
            limbs[0],                     # neck
            (limbs[1] + limbs[4]) / 2.,   # shoulders
            (limbs[2] + limbs[5]) / 2.,   # arms
            (limbs[3] + limbs[6]) / 2.,   # forearms
            limbs[7],                     # spine
            (limbs[8] + limbs[11]) / 2.,  # pelvis
            (limbs[9] + limbs[12]) / 2.,  # thighs
            (limbs[10] + limbs[13]) / 2.  # shins
        ])
    return limb_lengths


def limb_norm(x_a, x_b):

    limb_lengths_a = get_limb_lengths(x_a)
    limb_lengths_b = get_limb_lengths(x_b)

    limb_lengths_a[limb_lengths_a < 1e-3] = 1e-3
    local_scales = limb_lengths_b / limb_lengths_a

    x_ab = scale_limbs(x_a, global_scale=1.0, local_scales=local_scales)

    return x_ab
