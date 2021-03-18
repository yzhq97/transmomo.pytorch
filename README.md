# TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting

![Python](https://img.shields.io/badge/Python->=3.6-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.1.0-Orange?logo=pytorch)

### [Project Page](https://yzhq97.github.io/transmomo/) | [YouTube](https://youtu.be/akbRtnRMkMk) | [Paper](https://arxiv.org/pdf/2003.14401.pdf)

This is the official PyTorch implementation of the CVPR 2020 paper "TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting".



<p align='center'>  
  <img src='https://yzhq97.github.io/assets/transmomo/dance.gif' width='480'/>
</p>



## Environment

```
conda install pytorch torchvision cudatoolkit=<your cuda version>
conda install pyyaml scikit-image scikit-learn opencv
pip install -r requirements.txt
```

## Data

### Mixamo

[Mixamo](https://www.mixamo.com/) is a synthesized 3D character animation dataset.

1. Download mixamo data [here](https://drive.google.com/file/d/1lMa-4Bspn2_XV4wqo_s9Bfa35-19UAkB/view?usp=sharing).
2. Extract under `data/mixamo`

For directions for downloading 3D Mixamo data please refer to [this link](https://github.com/ChrisWu1997/2D-Motion-Retargeting/blob/master/dataset/Guide%20For%20Downloading%20Mixamo%20Data.md).

### SoloDance

SoloDance is a collection of dancing videos on youtube. We use [DensePose](https://github.com/facebookresearch/DensePose) to extract skeleton sequences from these videos for training.

1. Download the extracted skeleton sequences [here](https://drive.google.com/file/d/1366FaH0W2VYVW26ZbQJUp1x5GgMyMXuo/view?usp=sharing).
2. Extract under `data/solo_dance`

The original videos can be downloaded [here](https://drive.google.com/drive/folders/1hBj2uVJGABZz2aiqVYJpJ4SqBhYT-kYz?usp=sharing).

### Preprocessing
run `sh scripts/preprocess.sh` to preprocess the two datasets above.

## Pretrained model

Download the pretrained models [here](https://drive.google.com/drive/folders/1xZ2Pw7ObrDUIH89ipH1diyFZJxeXNDd8?usp=sharing).

## Inference

1. For *Skeleton Extraction*, please consider using a pose estimation library such as [Detectron2](https://github.com/facebookresearch/detectron2). We require the input skeleton sequences to be in the format of a numpy `.npy` file:
   - The file should contain an array with shape `15 x 2 x length`.
   - The first dimension (15) corresponds the 15 body joint defined [here](https://github.com/yzhq97/transmomo.pytorch/blob/master/docs/keypoint_format.md).
   - The second dimension (2) corresponds to x and y coordinates.
   - The third dimension (length) is the temporal dimension. 

2. For *Motion Retargeting Network*, we provide the sample command for inference:

  ```shell script
  python infer_pair.py 
  --config configs/transmomo.yaml 
  --checkpoint transmomo_mixamo_36_800_24/checkpoints/autoencoder_00200000.pt # replace with actual path
  --source a.npy  # replace with actual path
  --target b.npy  # replace with actual path
  --source_width 1280 --source_height 720 
  --target_height 1920 --target_width 1080
  ```

3. For *Skeleton-to-Video Rendering*, please refer to [Everybody Dance Now](https://carolineec.github.io/everybody_dance_now/).

## Training

To train the *Motion Retargeting Network*, run
```shell script
python train.py --config configs/transmomo.yaml
```
To train on the SoloDance dataest, run
```shell script
python train.py --config configs/transmomo_solo_dance.yaml
```

## Testing

For testing motion retargeting MSE, first generate the motion-retargeted motions with
```shell script
python test.py
--config configs/transmomo.yaml # replace with the actual config used for training
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
transmomo.pytorch
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

- [x] Detailed documentation

- [ ] Add example files

- [x] Release in-the-wild dancing video dataset (unannotated)

- [ ] Tool for visualizing Mixamo test error

- [ ] Tool for converting keypoint formats

## Citation

Z. Yang*, W. Zhu*, W. Wu*, C. Qian, Q. Zhou, B. Zhou, C. C. Loy. "TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. (* indicates equal contribution.)

BibTeX:
```bibtex
@inproceedings{transmomo2020,
  title={TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting},
  author={Yang, Zhuoqian and Zhu, Wentao and Wu, Wayne and Qian, Chen and Zhou, Qiang and Zhou, Bolei and Loy, Chen Change},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Acknowledgement

This repository is partly based on Rundi Wu's [Learning Character-Agnostic Motion for Motion Retargeting in 2D](https://motionretargeting2d.github.io/) and Xun Huang's [MUNIT: Multimodal UNsupervised Image-to-image Translation](https://github.com/NVlabs/MUNIT). The skeleton-to-rendering part is based on [Everybody Dance Now](https://carolineec.github.io/everybody_dance_now/). We sincerely thank them for their inspiration and contribution to the community.
