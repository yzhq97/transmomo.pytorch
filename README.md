# TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting

### [Project Page](https://yzhq97.github.io/transmomo/) | [YouTube](https://youtu.be/akbRtnRMkMk) | [Paper](https://arxiv.org/pdf/2003.14401.pdf)

This is the official PyTorch implementation of the CVPR 2020 paper "TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting".



<p align='center'>  
  <img src='https://yzhq97.github.io/assets/transmomo/dance.gif' width='480'/>
</p>



## Environment

* Python 3.6  
* Pytorch >= 1.1.0
```
conda install pytorch torchvision cudatoolkit=<your cuda version>
conda install pyyaml scikit-image scikit-learn opencv
pip install -r requirements.txt
```

## Data

### Mixamo

1. Download mixamo data [here](https://drive.google.com/open?id=1z0kD_F4jHk2sMqgvYOPfTBsguU7uGY1x).
2. Extract under `data/mixamo`
3. run `sh scripts/preprocess.sh`

For directions for downloading Mixamo data please refer to [this link](https://github.com/ChrisWu1997/2D-Motion-Retargeting/blob/master/dataset/Guide%20For%20Downloading%20Mixamo%20Data.md).

## Pretrained model

Download mixamo pretrained model [here](https://drive.google.com/open?id=120LeeR1WjdO0Emk_6hVRERu1I6Bimi6Q).

## Inference

Here inference refers motion retargeting, i.e. transfering motion from a source skeleton to a target skeleton.
We require the input skeleton sequences (e.g. extracted using a pose estimation method such as OpenPose) be provided in the format of a numpy `.npy` file.
The file should contain an array with shape `15 x 2 x length`.
The first dimension (15) corresponds the 15 body joint defined [here](https://github.com/yzhq97/transmomo.pytorch/blob/master/docs/keypoint_format.md).
The second dimension (2) corresponds to x and y coordinates.
The third dimenstion (length) is the temporal dimension. Sample command for inference:

```shell script
python infer_pair.py 
--config configs/transmomo.yaml 
--checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt # replace with actual path
--source a.npy  # replace with actual path
--target b.npy  # replace with actual path
--source_width 1280 --source_height 720 
--target_height 1920 --target_width 1080
```

For skeleton-to-video rendering, please refer to [Everybody Dance Now](https://carolineec.github.io/everybody_dance_now/).

## Training

To train the model, run
```shell script
python train.py --config configs/transmomo.yaml
```

## Testing

For testing motion retargeting MSE, first generate the motion-retargeted motions with
```shell script
python test.py
--config configs/transmomo.yaml
--checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt
--out_dir transmomo_mixamo_36_800_24_results # replace actual path to output directory
```
And then compute MSE by
```shell script
python scripts/compute_mse.py 
--in_dir transmomo_mixamo_36_800_24_results # replace with the previous output directory
```

## Project Structure

```
motion-disentangle-research.pytorch
├── configs - configuration files
├── data - place for storing data
├── docs - documentations
├── lib
│   ├── data.py - datasets and dataLoaders
│   ├── networks - encoders, decoders, discriminators, etc.
│   ├── trainer.py - training pipeline
│   ├── loss.py - loss functions
│   ├── operation.py - operations, e.g. rotation, projection, etc.
│   └── util - utility functions
├── out - place for storing output
├── infer_pair.py - perform motion retargeting
├── render_interpolate.py - perform motion and body interpolation
├── scripts - scripts for data processing and experiments
├── test.py - test MSE
└── train.py - main entrance for training
```

## TODOs

* Detailed Documentation
* Add example files
* Release in-the-wild dancing video dataset (unannotated)
* Tool for visualizing MSE error
* Tool for converting keypoint format

## Citation

Z. Yang*, W. Zhu*, W. Wu*, C. Qian, Q. Zhou, B. Zhou, C. C. Loy. "TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. (* indicates equal contribution.)

BibTex:
```
@inproceedings{transmomo2020,
title={TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting},
author={Yang, Zhuoqian and Zhu, Wentao and Wu, Wayne and Qian, Chen and Zhou, Qiang and Zhou, Bolei and Loy, Chen Change},
booktitle={Computer Vision and Pattern Recognition},
year={2020}
}
```

## Acknowledgement

This repository is partly based on Rundi Wu's [Learning Character-Agnostic Motion for Motion Retargeting in 2D](https://motionretargeting2d.github.io/) and Xun Huang's [MUNIT: Multimodal UNsupervised Image-to-image Translation](https://github.com/NVlabs/MUNIT). The skeleton-to-rendering part is based on [Everybody Dance Now](https://carolineec.github.io/everybody_dance_now/). We sincerely thank them for their inspiration and contribution to the community.
