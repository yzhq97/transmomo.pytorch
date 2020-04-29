import sys, os
sys.path.insert(0, os.getcwd())
import json
import argparse
import numpy as np
from itertools import combinations
from lib.util.motion import rotate_basis, preprocess_mixamo
import imageio

def load_and_preprocess(path):

    motion3d = np.load(path)

    # length must be multiples of 8 due to the size of convolution
    _, _, T = motion3d.shape
    T = (T // 8) * 8
    motion3d = motion3d[:, :, :T]

    # project to 2d
    motion_proj = motion3d[:, [0, 2], :]

    # reformat for mixamo data
    motion_proj = preprocess_mixamo(motion_proj, unit=1.0)

    return motion_proj

def rotate_3d(motion3d, angles):

    local3d = rotate_basis(np.eye(3), angles)
    motion3d = local3d @ motion3d

    return motion3d

def relocate(motion, fix_hip):

    if fix_hip:
        # fix hip joint in all frames
        motion = motion - motion[8:9, :, :]
    else:
        # align hip joint in the first frame
        center = motion[8, :, 0]
        motion = motion - center[np.newaxis, :, np.newaxis]

    return motion

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default="data/mixamo/36_800_24/mse_description.json")
    parser.add_argument('--data_dir', type=str, default="data/mixamo/36_800_24/test_random_rotate",
                        help="path to the directory storing test data")
    parser.add_argument('--in_dir', type=str, required=True,
                        help="path to the directory storing the inferred data")
    parser.add_argument('--fix_hip', type=bool, default=True,
                        help="whether or not to fix hip joint position")
    args = parser.parse_args()

    description = json.load(open(args.description))
    chars = list(description.keys())

    sum_squared_error = 0.0
    sum_absolute_error = 0.0
    n_nums = 0
    cnt = 0

    for char1, char2 in combinations(chars, 2):

        motions1 = description[char1]
        motions2 = description[char2]

        for i, mot1 in enumerate(motions1):
            for j, mot2 in enumerate(motions2):

                gt_path_ab = os.path.join(args.data_dir, char2, mot1, "{}.npy".format(mot1))
                gt_path_ba = os.path.join(args.data_dir, char1, mot2, "{}.npy".format(mot2))

                path_ab = os.path.join(args.in_dir, "motion_{}_{}_body_{}_{}.npy".format(char1, i, char2, j))
                path_ba = os.path.join(args.in_dir, "motion_{}_{}_body_{}_{}.npy".format(char2, j, char1, i))

                gt_ab = load_and_preprocess(gt_path_ab)
                gt_ba = load_and_preprocess(gt_path_ba)
                gt_ab = relocate(gt_ab, args.fix_hip)
                gt_ba = relocate(gt_ba, args.fix_hip)

                infered_ab = np.load(path_ab)
                infered_ba = np.load(path_ba)
                infered_ab = relocate(infered_ab, args.fix_hip)
                infered_ba = relocate(infered_ba, args.fix_hip)

                N, M, T = gt_ab.shape
                sum_squared_error += np.sum((gt_ab - infered_ab) ** 2)
                sum_absolute_error += np.sum(np.abs(gt_ab - infered_ab))
                n_nums += N * M * T

                N, M, T = gt_ba.shape
                sum_squared_error += np.sum((gt_ba - infered_ba) ** 2)
                sum_absolute_error += np.sum(np.abs(gt_ba - infered_ba))
                n_nums += N * M * T

            cnt += 1
            print("loaded {} pairs".format(cnt), end="\r")

    mse = sum_squared_error / n_nums
    mae = sum_absolute_error / n_nums

    out_str = "{} MSE {} = {}\n".format(args.in_dir, "(fix_hip)" if args.fix_hip else "", mse)
    out_str += "{} MAE {} = {}\n".format(args.in_dir, "(fix_hip)" if args.fix_hip else "", mae)

    print(out_str)

if __name__ == "__main__":
    main()






