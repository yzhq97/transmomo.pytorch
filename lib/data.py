import sys, os
thismodule = sys.modules[__name__]

from lib.util.motion import preprocess_mixamo, rotate_motion_3d, limb_scale_motion_2d, normalize_motion, get_change_of_basis, localize_motion, scale_limbs

import torch
import glob
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm

view_angles = np.array([ i * np.pi / 6 for i in range(-3, 4)])

def get_dataloader(phase, config):

    config.data.batch_size = config.batch_size
    config.data.seq_len = config.seq_len
    dataset_cls_name = config.data.train_cls if phase == 'train' else config.data.eval_cls
    dataset_cls = getattr(thismodule, dataset_cls_name)
    dataset = dataset_cls(phase, config.data)

    dataloader = DataLoader(dataset, shuffle=(phase=='train'),
                            batch_size=config.batch_size,
                            num_workers=(config.data.num_workers if phase == 'train' else 1),
                            worker_init_fn=lambda _: np.random.seed(),
                            drop_last=True)

    return dataloader


class _MixamoDatasetBase(Dataset):
    def __init__(self, phase, config):
        super(_MixamoDatasetBase, self).__init__()

        assert phase in ['train', 'test']
        self.phase = phase
        self.data_root = config.train_dir if phase=='train' else config.test_dir
        self.meanpose_path = config.train_meanpose_path if phase=='train' else config.test_meanpose_path
        self.stdpose_path = config.train_stdpose_path if phase=='train' else config.test_stdpose_path
        self.unit = config.unit
        self.aug = (phase == 'train')
        self.character_names = sorted(os.listdir(self.data_root))

        items = glob.glob(os.path.join(self.data_root, self.character_names[0], '*/motions/*.npy'))
        self.motion_names = ['/'.join(x.split('/')[-3:]) for x in items]

        self.meanpose, self.stdpose = get_meanpose(phase, config)
        self.meanpose = self.meanpose.astype(np.float32)
        self.stdpose = self.stdpose.astype(np.float32)

        if 'preload' in config and config.preload:
            self.preload()
            self.cached = True
        else:
            self.cached = False

    def build_item(self, mot_name, char_name):
        """
        :param mot_name: animation_name/motions/xxx.npy
        :param char_name: character_name
        :return:
        """
        return os.path.join(self.data_root, char_name, mot_name)

    def load_item(self, item):
        if self.cached:
            data = self.cache[item]
        else:
            data = np.load(item)
        return data

    def preload(self):
        print("pre-loading into memory")
        pbar = tqdm(total=len(self))
        self.cache = {}
        for motion_name in self.motion_names:
            for character_name in self.character_names:
                item = self.build_item(motion_name, character_name)
                motion3d = np.load(item)
                self.cache[item] = motion3d
                pbar.update(1)

    @staticmethod
    def gen_aug_params(rotate=False):
        if rotate:
            params = {'ratio': np.random.uniform(0.8, 1.2),
                    'roll': np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6), (np.pi / 9, np.pi / 9, np.pi / 6))}
        else:
            params = {'ratio': np.random.uniform(0.5, 1.5)}
        return edict(params)

    @staticmethod
    def augmentation(data, params=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        if params is None:
            return data, params

        # rotate
        if 'roll' in params.keys():
            cx, cy, cz = np.cos(params.roll)
            sx, sy, sz = np.sin(params.roll)
            mat33_x = np.array([
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]
            ], dtype='float')
            mat33_y = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ], dtype='float')
            mat33_z = np.array([
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]
            ], dtype='float')
            data = mat33_x @ mat33_y @ mat33_z @ data

        # scale
        if 'ratio' in params.keys():
            data = data * params.ratio

        return data, params

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.motion_names) * len(self.character_names)


def get_meanpose(phase, config):

    meanpose_path = config.train_meanpose_path if phase == "train" else config.test_meanpose_path
    stdpose_path = config.train_stdpose_path if phase == "train" else config.test_stdpose_path

    if os.path.exists(meanpose_path) and os.path.exists(stdpose_path):
        meanpose = np.load(meanpose_path)
        stdpose = np.load(stdpose_path)
    else:
        meanpose, stdpose = gen_meanpose(phase, config)
        np.save(meanpose_path, meanpose)
        np.save(stdpose_path, stdpose)
        print("meanpose saved at {}".format(meanpose_path))
        print("stdpose saved at {}".format(stdpose_path))

    if meanpose.shape[-1] == 2:
        mean_x, mean_y = meanpose[:, 0], meanpose[:, 1]
        meanpose = np.stack([mean_x, mean_x, mean_y], axis=1)

    if stdpose.shape[-1] == 2:
        std_x, std_y = stdpose[:, 0], stdpose[:, 1]
        stdpose = np.stack([std_x, std_x, std_y], axis=1)

    return meanpose, stdpose


def gen_meanpose(phase, config, n_samp=20000):

    data_dir = config.train_dir if phase == "train" else config.test_dir
    all_paths = glob.glob(os.path.join(data_dir, '*/*/motions/*.npy'))
    random.shuffle(all_paths)
    all_paths = all_paths[:n_samp]
    all_joints = []

    print("computing meanpose and stdpose")

    for path in tqdm(all_paths):
        motion = np.load(path)
        if motion.shape[1] == 3:
            basis = None
            if sum(config.rotation_axes) > 0:
                x_angles = view_angles if config.rotation_axes[0] else np.array([0])
                z_angles = view_angles if config.rotation_axes[1] else np.array([0])
                y_angles = view_angles if config.rotation_axes[2] else np.array([0])
                x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
                angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
                i = np.random.choice(len(angles))
                basis = get_change_of_basis(motion, angles[i])
                motion = preprocess_mixamo(motion)
                motion = rotate_motion_3d(motion, basis)
                motion = localize_motion(motion)
                all_joints.append(motion)
            else:
                motion = preprocess_mixamo(motion)
                motion = rotate_motion_3d(motion, basis)
                motion = localize_motion(motion)
                all_joints.append(motion)
        else:
            motion = motion * 128
            motion_proj = localize_motion(motion)
            all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)

    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9

    return meanpose, stdpose


class MixamoDataset(_MixamoDatasetBase):

    def __init__(self, phase, config):
        super(MixamoDataset, self).__init__(phase, config)
        x_angles = view_angles if config.rotation_axes[0] else np.array([0])
        z_angles = view_angles if config.rotation_axes[1] else np.array([0])
        y_angles = view_angles if config.rotation_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
        self.view_angles = angles

    def preprocessing(self, motion3d, view_angle=None, params=None):
        """
        :param item: filename built from self.build_tiem
        :return:
        """

        if self.aug: motion3d, params = self.augmentation(motion3d, params)

        basis = None
        if view_angle is not None: basis = get_change_of_basis(motion3d, view_angle)

        motion3d = preprocess_mixamo(motion3d)
        motion3d = rotate_motion_3d(motion3d, basis)
        motion3d = localize_motion(motion3d)
        motion3d = normalize_motion(motion3d, self.meanpose, self.stdpose)

        motion2d = motion3d[:, [0, 2], :]

        motion3d = motion3d.reshape([-1, motion3d.shape[-1]])
        motion2d = motion2d.reshape([-1, motion2d.shape[-1]])

        motion3d = torch.from_numpy(motion3d).float()
        motion2d = torch.from_numpy(motion2d).float()

        return motion3d, motion2d

    def __getitem__(self, index):
        # select two motions
        idx_a, idx_b = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot_a, mot_b = self.motion_names[idx_a], self.motion_names[idx_b]
        # select two characters
        idx_a, idx_b = np.random.choice(len(self.character_names), size=2, replace=False)
        char_a, char_b = self.character_names[idx_a], self.character_names[idx_b]
        idx_a, idx_b = np.random.choice(len(self.view_angles), size=2, replace=False)
        view_a, view_b = self.view_angles[idx_a], self.view_angles[idx_b]

        if self.aug:
            param_a = self.gen_aug_params(rotate=False)
            param_b = self.gen_aug_params(rotate=False)
        else:
            param_a = param_b = None

        item_a = self.load_item(self.build_item(mot_a, char_a))
        item_b = self.load_item(self.build_item(mot_b, char_b))
        item_ab = self.load_item(self.build_item(mot_a, char_b))
        item_ba = self.load_item(self.build_item(mot_b, char_a))

        X_a, x_a = self.preprocessing(item_a, view_a, param_a)
        X_b, x_b = self.preprocessing(item_b, view_b, param_b)

        X_aab, x_aab = self.preprocessing(item_a, view_b, param_a)
        X_bba, x_bba = self.preprocessing(item_b, view_a, param_b)
        X_aba, x_aba = self.preprocessing(item_ab, view_a, param_b)
        X_bab, x_bab = self.preprocessing(item_ba, view_b, param_a)
        X_abb, x_abb = self.preprocessing(item_ab, view_b, param_b)
        X_baa, x_baa = self.preprocessing(item_ba, view_a, param_a)

        return {"X_a": X_a, "X_b": X_b,
                "X_aab": X_aab, "X_bba": X_bba,
                "X_aba": X_aba, "X_bab": X_bab,
                "X_abb": X_abb, "X_baa": X_baa,
                "x_a": x_a, "x_b": x_b,
                "x_aab": x_aab, "x_bba": x_bba,
                "x_aba": x_aba, "x_bab": x_bab,
                "x_abb": x_abb, "x_baa": x_baa,
                "mot_a": mot_a, "mot_b": mot_b,
                "char_a": char_a, "char_b": char_b,
                "view_a": view_a, "view_b": view_b,
                "meanpose": self.meanpose, "stdpose": self.stdpose}


class MixamoLimbScaleDataset(_MixamoDatasetBase):

    def __init__(self, phase, config):
        super(MixamoLimbScaleDataset, self).__init__(phase, config)
        self.global_range = config.global_range
        self.local_range = config.local_range

        x_angles = view_angles if config.rotation_axes[0] else np.array([0])
        z_angles = view_angles if config.rotation_axes[1] else np.array([0])
        y_angles = view_angles if config.rotation_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
        self.view_angles = angles

    def preprocessing(self, motion3d, view_angle=None, params=None):
        if self.aug: motion3d, params = self.augmentation(motion3d, params)

        basis = None
        if view_angle is not None: basis = get_change_of_basis(motion3d, view_angle)

        motion3d = preprocess_mixamo(motion3d)
        motion3d = rotate_motion_3d(motion3d, basis)
        motion2d = motion3d[:, [0, 2], :]
        motion2d_scale = limb_scale_motion_2d(motion2d, self.global_range, self.local_range)

        motion2d = localize_motion(motion2d)
        motion2d_scale = localize_motion(motion2d_scale)

        motion2d = normalize_motion(motion2d, self.meanpose, self.stdpose)
        motion2d_scale = normalize_motion(motion2d_scale, self.meanpose, self.stdpose)

        motion2d = motion2d.reshape([-1, motion2d.shape[-1]])
        motion2d_scale = motion2d_scale.reshape((-1, motion2d_scale.shape[-1]))
        motion2d = torch.from_numpy(motion2d).float()
        motion2d_scale = torch.from_numpy(motion2d_scale).float()

        return motion2d, motion2d_scale

    def __getitem__(self, index):
        # select two motions
        motion_idx = np.random.choice(len(self.motion_names))
        motion = self.motion_names[motion_idx]
        # select two characters
        char_idx = np.random.choice(len(self.character_names))
        character = self.character_names[char_idx]
        view_idx = np.random.choice(len(self.view_angles))
        view = self.view_angles[view_idx]

        if self.aug:
            param = self.gen_aug_params(rotate=True)
        else:
            param = None

        item = self.build_item(motion, character)

        x, x_s = self.preprocessing(self.load_item(item), view, param)

        return {"x": x, "x_s": x_s, "mot": motion, "char": character, "view": view,
                "meanpose": self.meanpose, "stdpose": self.stdpose}


class SoloDanceDataset(Dataset):

    def __init__(self, phase, config):
        super(SoloDanceDataset, self).__init__()
        self.global_range = config.global_range
        self.local_range = config.local_range

        assert phase in ['train', 'test']
        self.data_root = config.train_dir if phase=='train' else config.test_dir
        self.phase = phase
        self.unit = config.unit
        self.meanpose_path = config.train_meanpose_path if phase == 'train' else config.test_meanpose_path
        self.stdpose_path = config.train_stdpose_path if phase == 'train' else config.test_stdpose_path
        self.character_names = sorted(os.listdir(self.data_root))

        self.items = glob.glob(os.path.join(self.data_root, '*/*/motions/*.npy'))
        self.meanpose, self.stdpose = get_meanpose(phase, config)
        self.meanpose = self.meanpose.astype(np.float32)
        self.stdpose = self.stdpose.astype(np.float32)

        if 'preload' in config and config.preload:
            self.preload()
            self.cached = True
        else:
            self.cached = False

    def load_item(self, item):
        if self.cached:
            data = self.cache[item]
        else:
            data = np.load(item)
        return data

    def preload(self):
        print("pre-loading into memory")
        pbar = tqdm(total=len(self))
        self.cache = {}
        for item in self.items:
            motion = np.load(item)
            self.cache[item] = motion
            pbar.update(1)

    def preprocessing(self, motion):

        motion = motion * self.unit

        motion[1, :, :] = (motion[2, :, :] + motion[5, :, :]) / 2
        motion[8, :, :] = (motion[9, :, :] + motion[12, :, :]) / 2

        global_scale = self.global_range[0] + np.random.random() * (self.global_range[1] - self.global_range[0])
        local_scales = self.local_range[0] + np.random.random([8]) * (self.local_range[1] - self.local_range[0])
        motion_scale = scale_limbs(motion, global_scale, local_scales)

        motion = localize_motion(motion)
        motion_scale = localize_motion(motion_scale)
        motion = normalize_motion(motion, self.meanpose, self.stdpose)
        motion_scale = normalize_motion(motion_scale, self.meanpose, self.stdpose)
        motion = motion.reshape((-1, motion.shape[-1]))
        motion_scale = motion_scale.reshape((-1, motion_scale.shape[-1]))
        motion = torch.from_numpy(motion).float()
        motion_scale = torch.from_numpy(motion_scale).float()
        return motion, motion_scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        motion = self.load_item(item)
        x, x_s = self.preprocessing(motion)
        return {"x": x, "x_s": x_s, "meanpose": self.meanpose, "stdpose": self.stdpose}
