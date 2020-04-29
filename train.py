from lib.data import get_dataloader
from lib.util.general import write_loss, get_config, to_gpu
import lib.trainer
import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import tensorboardX
import shutil

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the config file.")
    parser.add_argument("-o", "--out_dir", type=str, default="out", help="outputs path")
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("--preload", action="store_true", help="load all data into memory before training")
    opts = parser.parse_args()
    return opts

def train_with_config(config, opts, logger=None):

    cudnn.benchmark = True

    # Load experiment setting
    if opts.preload: config.data.preload = True
    max_iter = config.max_iter

    # Setup model and data loader
    trainer_cls = getattr(lib.trainer, config.trainer)
    trainer = trainer_cls(config)
    trainer.cuda()

    if logger is not None: logger.log("loading data")
    train_loader = get_dataloader("train", config)
    val_loader = get_dataloader("test", config)

    # Setup logger and output folders
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_dir, config.name, "logs"))
    checkpoint_directory = os.path.join(opts.out_dir, 'checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)
    shutil.copy(opts.config, os.path.join(opts.out_dir, "config.yaml")) # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory, config=config) if opts.resume else 0

    pbar = tqdm(total=max_iter)
    pbar.set_description(config.name)
    pbar.update(iterations)
    print("%s: training started" % config.name)
    if logger is not None: logger.log("training started")

    start = time.time()

    while True:

        for it, data in enumerate(train_loader):

            data = to_gpu(data)

            # Main training code
            trainer.dis_update(data, config)
            trainer.ae_update(data, config)

            trainer.update_learning_rate()

            # Run validation
            if (iterations + 1) % config.val_iter == 0:
                val_batches = []
                for i, batch in enumerate(val_loader):
                    if i >= config.val_batches: break
                    val_batches.append(batch)
                val_data = {}
                for key in val_batches[0].keys():
                    data = [batch[key] for batch in val_batches]
                    if isinstance(data[0], torch.Tensor):
                        val_data[key] = torch.cat(data, dim=0)
                val_data = to_gpu(val_data)
                trainer.validate(val_data, config)

            # Dump training stats in log file
            if (iterations + 1) % config.log_iter == 0:
                if logger is not None:
                    elapsed = (time.time() - start) / 3600.0
                    logger.log("training %6d/%6d, elapsed: %.2f hrs" % (iterations+1, max_iter, elapsed))
                write_loss(iterations, trainer, train_writer)

            # Save network weights
            if (iterations + 1) % config.snapshot_save_iter == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            pbar.update(1)
            if iterations >= max_iter:
                print("training finished")
                return


if __name__ == "__main__":

    opts = parse_args()
    config = get_config(opts.config)

    train_with_config(config, opts)