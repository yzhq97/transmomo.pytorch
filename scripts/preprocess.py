import sys, os
sys.path.insert(0, os.getcwd())
import shutil
import numpy as np
from tqdm import tqdm
import argparse
import glob
import random

view_angles = [ [0, 0, i * np.pi / 6] for i in range(-3, 4)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="data/mixamo/36_937_24/train")
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

    characters = os.listdir(in_dir)
    characters = sorted(characters)
    n_seqs = 0

    for char in tqdm(characters):

        char_dir = os.path.join(in_dir, char)
        motion_names = os.listdir(char_dir)
        motion_names = sorted(motion_names)

        for motion_name in motion_names:

            motion_dir = os.path.join(char_dir, motion_name)
            file_path = os.path.join(motion_dir, "%s.npy" % motion_name)
            save_dir = os.path.join(motion_dir, "motions")
            os.makedirs(save_dir, exist_ok=True)

            motion = np.load(file_path)
            if padding > 0: motion = reflection_padding(motion, padding)
            length = motion.shape[-1]

            if length < seq_len:
                shutil.rmtree(motion_dir)
                continue

            for i, start in enumerate(range(0, length - seq_len, interval)):

                motion_seq = motion[:, :, start:start+seq_len]
                assert motion_seq.shape[-1] == seq_len
                save_path = os.path.join(save_dir, "%d.npy" % (i+1))
                np.save(save_path, motion_seq)
                n_seqs += 1

    return n_seqs

if __name__ == "__main__":

    args = parse_args()
    print("processing")
    n_seqs = process(args.data_dir, args.seq_len, args.padding, args.interval)
    print("%d seqs" % n_seqs)
