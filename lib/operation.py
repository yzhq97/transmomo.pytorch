import torch
import torch.nn.functional as F
import numpy as np
import imageio
from math import pi
from tqdm import tqdm
from lib.data import get_dataloader, get_meanpose
from lib.util.general import get_config
from lib.util.visualization import motion2video_np, hex2rgb
import os

eps = 1e-16


def localize_motion_torch(motion):
    """
    :param motion: B x J x D x T
    :return:
    """
    B, J, D, T = motion.size()

    # subtract centers to local coordinates
    centers = motion[:, 8:9, :, :] # B x 1 x D x (T-1)
    motion = motion - centers

    # adding velocity
    translation = centers[:, :, :, 1:] - centers[:, :, :, :-1] # B x 1 x D x (T-1)
    velocity = F.pad(translation, [1, 0], "constant", 0.) # B x 1 x D x T
    motion = torch.cat([motion[:, :8], motion[:, 9:], velocity], dim=1)

    return motion


def normalize_motion_torch(motion, meanpose, stdpose):
    """
    :param motion: (B, J, D, T)
    :param meanpose: (J, D)
    :param stdpose: (J, D)
    :return:
    """
    B, J, D, T = motion.size()
    if D == 2 and meanpose.size(1) == 3:
        meanpose = meanpose[:, [0, 2]]
    if D == 2 and stdpose.size(1) == 3:
        stdpose = stdpose[:, [0, 2]]
    return (motion - meanpose.view(1, J, D, 1)) / stdpose.view(1, J, D, 1)


def normalize_motion_inv_torch(motion, meanpose, stdpose):
    """
    :param motion: (B, J, D, T)
    :param meanpose: (J, D)
    :param stdpose: (J, D)
    :return:
    """
    B, J, D, T = motion.size()
    if D == 2 and meanpose.size(1) == 3:
        meanpose = meanpose[:, [0, 2]]
    if D == 2 and stdpose.size(1) == 3:
        stdpose = stdpose[:, [0, 2]]
    return motion * stdpose.view(1, J, D, 1) + meanpose.view(1, J, D, 1)


def globalize_motion_torch(motion):
    """
    :param motion: B x J x D x T
    :return:
    """
    B, J, D, T = motion.size()

    motion_inv = torch.zeros_like(motion)
    motion_inv[:, :8] = motion[:, :8]
    motion_inv[:, 9:] = motion[:, 8:-1]

    velocity = motion[:, -1:, :, :]
    centers = torch.zeros_like(velocity)
    displacement = torch.zeros_like(velocity[:, :, :, 0])

    for t in range(T):
        displacement += velocity[:, :, :, t]
        centers[:, :, :, t] = displacement

    motion_inv = motion_inv + centers

    return motion_inv


def restore_world_space(motion, meanpose, stdpose, n_joints=15):
    B, C, T = motion.size()
    motion = motion.view(B, n_joints, C // n_joints, T)
    motion = normalize_motion_inv_torch(motion, meanpose, stdpose)
    motion = globalize_motion_torch(motion)
    return motion


def convert_to_learning_space(motion, meanpose, stdpose):
    B, J, D, T = motion.size()
    motion = localize_motion_torch(motion)
    motion = normalize_motion_torch(motion, meanpose, stdpose)
    motion = motion.view(B, J*D, T)
    return motion


# tensor operations for rotating and projecting 3d skeleton sequence

def get_body_basis(motion_3d):
    """
    Get the unit vectors for vector rectangular coordinates for given 3D motion
    :param motion_3d: 3D motion from 3D joints positions, shape (B, n_joints, 3, seq_len).
    :param angles: (K, 3), Rotation angles around each axis.
    :return: unit vectors for vector rectangular coordinates's , shape (B, 3, 3).
    """
    B = motion_3d.size(0)

    # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
    horizontal = (motion_3d[:, 2] - motion_3d[:, 5] + motion_3d[:, 9] - motion_3d[:, 12]) / 2 # [B, 3, seq_len]
    horizontal = horizontal.mean(dim=-1) # [B, 3]
    horizontal = horizontal / horizontal.norm(dim=-1).unsqueeze(-1) # [B, 3]

    vector_z = torch.tensor([0., 0., 1.], device=motion_3d.device, dtype=motion_3d.dtype).unsqueeze(0).repeat(B, 1) # [B, 3]
    vector_y = torch.cross(horizontal, vector_z)   # [B, 3]
    vector_y = vector_y / vector_y.norm(dim=-1).unsqueeze(-1)
    vector_x = torch.cross(vector_y, vector_z)
    vectors = torch.stack([vector_x, vector_y, vector_z], dim=2)  # [B, 3, 3]

    vectors = vectors.detach()

    return vectors


def rotate_basis_euler(basis_vectors, angles):
    """
    Rotate vector rectangular coordinates from given angles.

    :param basis_vectors: [B, 3, 3]
    :param angles: [B, K, T, 3] Rotation angles around each axis.
    :return: [B, K, T, 3, 3]
    """
    B, K, T, _ = angles.size()

    cos, sin = torch.cos(angles), torch.sin(angles)
    cx, cy, cz = cos[:, :, :, 0], cos[:, :, :, 1], cos[:, :, :, 2]  # [B, K, T]
    sx, sy, sz = sin[:, :, :, 0], sin[:, :, :, 1], sin[:, :, :, 2]  # [B, K, T]

    x = basis_vectors[:, 0, :]  # [B, 3]
    o = torch.zeros_like(x[:, 0])  # [B]

    x_cpm_0 = torch.stack([o, -x[:, 2], x[:, 1]], dim=1)  # [B, 3]
    x_cpm_1 = torch.stack([x[:, 2], o, -x[:, 0]], dim=1)  # [B, 3]
    x_cpm_2 = torch.stack([-x[:, 1], x[:, 0], o], dim=1)  # [B, 3]
    x_cpm = torch.stack([x_cpm_0, x_cpm_1, x_cpm_2], dim=1)  # [B, 3, 3]
    x_cpm = x_cpm.unsqueeze(1).unsqueeze(2) # [B, 1, 1, 3, 3]

    x = x.unsqueeze(-1)  # [B, 3, 1]
    xx = torch.matmul(x, x.transpose(-1, -2)).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 3, 3]
    eye = torch.eye(n=3, dtype=basis_vectors.dtype, device=basis_vectors.device)
    eye = eye.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, 3, 3]
    mat33_x = cx.unsqueeze(-1).unsqueeze(-1) * eye \
              + sx.unsqueeze(-1).unsqueeze(-1) * x_cpm \
              + (1. - cx).unsqueeze(-1).unsqueeze(-1) * xx  # [B, K, T, 3, 3]

    o = torch.zeros_like(cz)
    i = torch.ones_like(cz)
    mat33_z_0 = torch.stack([cz, sz, o], dim=3)  # [B, K, T, 3]
    mat33_z_1 = torch.stack([-sz, cz, o], dim=3)  # [B, K, T, 3]
    mat33_z_2 = torch.stack([o, o, i], dim=3)  # [B, K, T, 3]
    mat33_z = torch.stack([mat33_z_0, mat33_z_1, mat33_z_2], dim=3)  # [B, K, T, 3, 3]

    basis_vectors = basis_vectors.unsqueeze(1).unsqueeze(2)
    basis_vectors = basis_vectors @ mat33_x.transpose(-1, -2) @ mat33_z


    return basis_vectors


def change_of_basis(motion_3d, basis_vectors=None, project_2d=False):
    # motion_3d: (B, n_joints, 3, seq_len)
    # basis_vectors: (B, K, T, 3, 3)

    if basis_vectors is None:
        motion_proj = motion_3d[:, :, [0, 2], :]  # [B, n_joints, 2, seq_len]
    else:
        if project_2d: basis_vectors = basis_vectors[:, :, :, [0, 2], :]
        _, K, seq_len, _, _ = basis_vectors.size()
        motion_3d = motion_3d.unsqueeze(1).repeat(1, K, 1, 1, 1)
        motion_3d = motion_3d.permute([0, 1, 4, 3, 2]) # [B, K, J, 3, T] -> [B, K, T, 3, J]
        motion_proj = basis_vectors @ motion_3d  # [B, K, T, 2, 3] @ [B, K, T, 3, J] -> [B, K, T, 2, J]
        motion_proj = motion_proj.permute([0, 1, 4, 3, 2]) # [B, K, T, 3, J] -> [B, K, J, 3, T]

    return motion_proj


def rotate_and_maybe_project_world(X, angles=None, body_reference=True, project_2d=False):

    out_dim = 2 if project_2d else 3
    batch_size, n_joints, _, seq_len = X.size()

    if angles is not None:
        K = angles.size(1)
        basis_vectors = get_body_basis(X) if body_reference else \
            torch.eye(3, device=X.device).unsqueeze(0).repeat(batch_size, 1, 1)
        basis_vectors = rotate_basis_euler(basis_vectors, angles)
        X_trans = change_of_basis(X, basis_vectors, project_2d=project_2d)
        X_trans = X_trans.reshape(batch_size * K, n_joints, out_dim, seq_len)
    else:
        X_trans = change_of_basis(X, project_2d=project_2d)
        X_trans = X_trans.reshape(batch_size, n_joints, out_dim, seq_len)

    return X_trans



def rotate_and_maybe_project_learning(X, meanpose, stdpose, angles=None, body_reference=True, project_2d=False):
    batch_size, channels, seq_len = X.size()
    n_joints = channels // 3
    X = restore_world_space(X, meanpose, stdpose, n_joints)
    X = rotate_and_maybe_project_world(X, angles, body_reference, project_2d)
    X = convert_to_learning_space(X, meanpose, stdpose)
    return X
