import os
import argparse
import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from lib.data import get_dataloader, get_meanpose
from lib.util.general import get_config
from lib.network import get_autoencoder
from lib.operation import change_of_basis
from lib.util.motion import preprocess_test, postprocess
from lib.util.general import pad_to_height, ensure_dir
from lib.util.visualization import motion2video, motion2video_np, hex2rgb
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, required=True, help="source npy path")
    parser.add_argument("--target", type=str, required=True, help="target npy path")

    parser.add_argument("-c", "--config", type=str, default="configs/transmomo.yaml", help="Path to the config file.")
    parser.add_argument("-cp", "--checkpoint", type=str, help="path to autoencoder checkpoint")
    parser.add_argument("-o", "--out_dir", type=str, default="out", help="output directory")

    parser.add_argument('--source_height', type=int, help="source video height")
    parser.add_argument('--source_width', type=int, help="source video width")
    parser.add_argument('--target_height', type=int, help="target video height")
    parser.add_argument('--target_width', type=int, help="target video width")

    parser.add_argument('--max_length', type=int, default=120,
                        help='maximum input video length')

    args = parser.parse_args()
    return args


def main(config, args):

    cudnn.benchmark = True

    ae = get_autoencoder(config)
    ae.load_state_dict(torch.load(args.checkpoint))
    ae.cuda()
    ae.eval()

    _, src_name = os.path.split(args.source)
    src_name, _ = os.path.splitext(src_name)
    _, tgt_name = os.path.split(args.target)
    tgt_name, _ = os.path.splitext(tgt_name)

    src_h, src_w, src_scale = pad_to_height(512, args.source_height, args.source_width)
    tgt_h, tgt_w, tgt_scale = pad_to_height(512, args.target_height, args.target_width)

    mean_pose, std_pose = get_meanpose("test", config.data)

    x_src = np.load(args.source)
    x_tgt = np.load(args.target)

    length = min(x_src.shape[-1], x_tgt.shape[-1], args.max_length)
    length = 8 * (length // 8)
    x_src = x_src[:, :, :length]
    x_tgt = x_tgt[:, :, :length]

    x_src, x_src_start = preprocess_test(x_src, mean_pose, std_pose, src_scale)
    x_tgt, _ = preprocess_test(x_tgt, mean_pose, std_pose, tgt_scale)

    x_src = torch.from_numpy(x_src.reshape((1, -1, length))).float().cuda()
    x_tgt = torch.from_numpy(x_tgt.reshape((1, -1, length))).float().cuda()

    x_cross = ae.cross2d(x_src, x_tgt, x_src)
    x_cross = postprocess(x_cross, mean_pose, std_pose, unit=1.0, start=x_src_start)

    out_path = os.path.join(args.out_dir, 'retarget_{}_{}.npy'.format(src_name, tgt_name))
    np.save(out_path, x_cross)


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config)
    config.batch_size = 1
    main(config, args)