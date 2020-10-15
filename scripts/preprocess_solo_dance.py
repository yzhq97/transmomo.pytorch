import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from tqdm import tqdm
import argparse

view_angles = [ [0, 0, i * np.pi / 6] for i in range(-3, 4)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="data/solo_dance/loose")
    parser.add_argument("-i", "--interval", type=int, default=32)
    parser.add_argument("-s", "--seq_len", type=int, default=64)
    parser.add_argument("-p", "--padding", type=int, default=0)
    args = parser.parse_args()
    return args


def reflection_padding(seq, padding):

    flip_seq = np.flip(seq, axis=-1)
    left_pad = flip_seq[:, :, :-1]
    while left_pad.shape[-1] < padding:
        flip_seq = np.flip(flip_seq)
        left_pad = np.concatenate([flip_seq[:, :, :-1], left_pad], axis=-1)
    left_pad = left_pad[:, :, -padding:]

    flip_seq = np.flip(seq, axis=-1)
    right_pad = flip_seq[:, :, 1:]
    while right_pad.shape[-1] < padding:
        flip_seq = np.flip(flip_seq)
        right_pad = np.concatenate([right_pad, flip_seq[:, :, 1:]], axis=-1)
    right_pad = right_pad[:, :, :padding]

    padded = np.concatenate([left_pad, seq, right_pad], axis=-1)

    return padded


def process(in_dir, seq_len, padding, interval):

    categories = os.listdir(in_dir)
    categories = sorted(categories)
    n_seqs = 0

    for cat in tqdm(categories):

        cat_dir = os.path.join(in_dir, cat)
        dance_names = os.listdir(cat_dir)
        dance_names = sorted(dance_names)

        for dance_name in dance_names:

            dance_dir = os.path.join(cat_dir, dance_name)
            files = os.listdir(dance_dir)
            files = sorted(files)
            save_dir = os.path.join(dance_dir, "motions")
            os.makedirs(save_dir, exist_ok=True)
            i = 1

            for file in files:

                file_path = os.path.join(dance_dir, file)

                motion = np.load(file_path)
                if padding > 0: motion = reflection_padding(motion, padding)
                length = motion.shape[-1]

                for start in range(0, length - seq_len, interval):

                    motion_seq = motion[:, :, start:start+seq_len]
                    assert motion_seq.shape[-1] == seq_len
                    save_path = os.path.join(save_dir, "%d.npy" % i)
                    np.save(save_path, motion_seq)
                    i += 1
                    n_seqs += 1

    return n_seqs

if __name__ == "__main__":

    args = parse_args()
    print("processing")
    n_seqs = process(args.data_dir, args.seq_len, args.padding, args.interval)
    print("%d seqs" % n_seqs)
