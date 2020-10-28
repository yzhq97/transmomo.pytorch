import os
import json
import torch
import argparse
import numpy as np
from lib.data import get_meanpose
from lib.network import get_autoencoder
from lib.util.motion import preprocess_mixamo, preprocess_test, postprocess
from lib.util.general import get_config
from lib.operation import rotate_and_maybe_project_world
from itertools import combinations


def load_and_preprocess(path, config, mean_pose, std_pose):

    motion3d = np.load(path)

    # length must be multiples of 8 due to the size of convolution
    _, _, T = motion3d.shape
    T = (T // 8) * 8
    motion3d = motion3d[:, :, :T]

    # project to 2d
    motion_proj = motion3d[:, [0, 2], :]

    # reformat for mixamo data
    motion_proj = preprocess_mixamo(motion_proj, unit=1.0)

    # preprocess for network input
    motion_proj, start = preprocess_test(motion_proj, mean_pose, std_pose, config.data.unit)
    motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))
    motion_proj = torch.from_numpy(motion_proj).float()

    return motion_proj, start

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='which config to use.')
    parser.add_argument('--description', type=str, default="data/mixamo/36_800_24/mse_description.json",
                        help="path to the description file which specifies how to run test")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="path to trained model weights")
    parser.add_argument('--data_dir', type=str, default="data/mixamo/36_800_24/test_random_rotate",
                        help="path to the directory storing test data")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="path to output directory")
    args = parser.parse_args()

    config = get_config(args.config)
    ae = get_autoencoder(config)
    ae.load_state_dict(torch.load(args.checkpoint))
    ae.cuda()
    ae.eval()
    mean_pose, std_pose = get_meanpose("test", config.data)
    print("loaded model")

    description = json.load(open(args.description))
    chars = list(description.keys())

    cnt = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for char1, char2 in combinations(chars, 2):

        motions1 = description[char1]
        motions2 = description[char2]

        for i, mot1 in enumerate(motions1):
            for j, mot2 in enumerate(motions2):

                path1 = os.path.join(args.data_dir, char1, mot1, "{}.npy".format(mot1))
                path2 = os.path.join(args.data_dir, char2, mot2, "{}.npy".format(mot2))

                ############
                # CROSS 2D #
                ############

                out_path1 = os.path.join(args.out_dir, "motion_{}_{}_body_{}_{}.npy".format(char1, i, char2, j))
                out_path2 = os.path.join(args.out_dir, "motion_{}_{}_body_{}_{}.npy".format(char2, j, char1, i))

                x_a, x_a_start = load_and_preprocess(path1, config, mean_pose, std_pose)
                x_b, x_b_start = load_and_preprocess(path2, config, mean_pose, std_pose)

                x_a_batch = x_a.unsqueeze(0).cuda()
                x_b_batch = x_b.unsqueeze(0).cuda()

                x_ab = ae.cross2d(x_a_batch, x_b_batch, x_a_batch)
                x_ba = ae.cross2d(x_b_batch, x_a_batch, x_b_batch)

                x_ab = postprocess(x_ab, mean_pose, std_pose, config.data.unit, start=x_a_start)
                x_ba = postprocess(x_ba, mean_pose, std_pose, config.data.unit, start=x_b_start)

                np.save(out_path1, x_ab)
                np.save(out_path2, x_ba)

                cnt += 1
                print("computed {} pairs".format(cnt), end="\r")

    print("finished" + " " * 20)


if __name__ == "__main__":
    main()


