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
    parser.add_argument("-o", "--out_dir", type=str, default="out", help="outputs path")

    parser.add_argument('--n_interp', type=int, default=4, help='number of interpolations')

    parser.add_argument('--source_height', type=int, help="source height")
    parser.add_argument('--source_width', type=int, help="source width")
    parser.add_argument('--target_height', type=int, help="target height")
    parser.add_argument('--target_width', type=int, help="target width")
    parser.add_argument('--out_height', type=int, help="height", default=512)
    parser.add_argument('--out_width', type=int, help="width", default=512)

    parser.add_argument('--color1', type=str, default='#a50b69#b73b87#db9dc3', help='color1')
    parser.add_argument('--color2', type=str, default='#4076e0#40a7e0#40d7e0', help='color2')

    parser.add_argument('--fps', type=float, default=25, help="fps of output video")
    parser.add_argument('--disable_smooth', action='store_true',
                        help="disable gaussian kernel smoothing")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")

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

    color1 = np.array(hex2rgb(args.color1))
    color2 = np.array(hex2rgb(args.color2))
    src_h, src_w, src_scale = pad_to_height(512, args.source_height, args.source_width)
    tgt_h, tgt_w, tgt_scale = pad_to_height(512, args.target_height, args.target_width)
    h = args.out_height
    w = args.out_width
    mean_pose, std_pose = get_meanpose("test", config.data)

    n = args.n_interp
    step_size = 1. / (n-1)

    x_src = np.load(args.source)
    x_tgt = np.load(args.target)

    length = min(x_src.shape[-1], x_tgt.shape[-1], args.max_length)
    length = 8 * (length // 8)
    x_src = x_src[:, :, :length]
    x_tgt = x_tgt[:, :, :length]

    x_src, _ = preprocess_test(x_src, mean_pose, std_pose, src_scale)
    x_tgt, _ = preprocess_test(x_tgt, mean_pose, std_pose, tgt_scale)

    x_src = torch.from_numpy(x_src.reshape((1, -1, length))).float().cuda()
    x_tgt = torch.from_numpy(x_tgt.reshape((1, -1, length))).float().cuda()

    x_interp = ae.interpolate(x_src, x_tgt, n)

    vid_array = np.zeros([length, n * h, n * w, 3], dtype=np.uint8)
    pbar = tqdm(total=n*n)
    pbar.set_description("rendering")

    for i in range(n):
        for j in range(n):

            motion = postprocess(x_interp[:, i, j], mean_pose, std_pose, unit=1.0, start=np.array([args.out_width // 2, args.out_height // 2]))
            if not args.disable_smooth:
                motion = gaussian_filter1d(motion, sigma=2, axis=-1)

            color_weight = j * step_size
            color = (1. - color_weight) * color1 + color_weight * color2
            vid = motion2video_np(motion, h, w, color, transparency=False, show_progress=False)
            vid_array[:, i * h: (i + 1) * h, j * w: (j + 1) * w, :] = vid
            pbar.update(1)

    out_path = os.path.join(args.out_dir, 'interp_{}_{}.mp4'.format(src_name, tgt_name))
    videowriter = imageio.get_writer(out_path, fps=args.fps)
    for i in range(length):
        videowriter.append_data(vid_array[i])
    videowriter.close()


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config)
    config.batch_size = 1
    main(config, args)