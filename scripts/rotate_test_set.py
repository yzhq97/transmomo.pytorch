# Mixamo data is by default front-facing
# For testing, we random rotate each motion around the vertical axis

import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import argparse
from lib.util.motion import rotate_basis

np.random.seed(123)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data/mixamo/36_800_24/test")
    parser.add_argument('--out_dir', type=str, default="data/mixamo/36_800_24/test_random_rotate")
    args = parser.parse_args()

    chars = os.listdir(args.data_dir)
    chars = sorted(chars)

    mots = os.listdir(os.path.join(args.data_dir, chars[0]))
    mots = sorted(mots)

    cnt = 0

    for mot in mots:

        angles = np.array([0., 0., 2 * np.pi * np.random.random()])

        for char in chars:

            in_path = os.path.join(args.data_dir, char, mot, "{}.npy".format(mot))

            motion3d = np.load(in_path)
            basis = rotate_basis(np.eye(3), angles)
            rotated = basis @ motion3d

            out_dir = os.path.join(args.out_dir, char, mot)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, char, mot, "{}.npy".format(mot))

            np.save(out_path, rotated)

            cnt += 1
            print("computed {} seqs".format(cnt), end="\r")

    print("finished" + " " * 20)

