# pytorch-fc4

A PyTorch implementation of "FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling"

The original code for the FC4 method is quite outdated (it is based on Python 2 and an old version of Tensorflow). This
an attempt of providing a clean and modern Python3-based re-implementation of that method using the PyTorch library.

## FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling

Original resources:

* [Tensorflow code (requires Python 2)](https://github.com/yuanming-hu/fc4)
* [Paper by Yuanming Hu, Baoyuan Wang and Stephen Lin (2017)](https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/)

## SqueezeNet

This implementation of the FC4 method uses SqueezeNet. The SqueezeNet implementation
is [the one offered by PyTorch](https://github.com/pytorch/vision/blob/072d8b2280569a2d13b91d3ed51546d201a57366/torchvision/models/squeezenet.py)
and features:

* **SqueezeNet 1.0**. Introduced
  in ['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360)
* **SqueezeNet 1.1** *(has 2.4x less computation and slightly fewer parameters than 1.0, without sacrificing accuracy)*.
  Introduced in this [repository](https://github.com/forresti/SqueezeNet)

## Requirements

This project has been developed and tested using Python 3.8 and Torch > 1.7. Please install the required packages
using `pip3 install -r requirements.txt`.

The device on which to run the method (either `cpu` or `cuda:x`) and the random seed for reproducibility can be set as
global variables at `auxiliary/settings.py`.

Note that this implementation allows for deactivating the confidence-weighted pooling, in which case a simpler summation
pooling will be used. The usage of the confidence-weighted pooling can be configured toggling
the `USE_CONFIDENCE_WEIGHTED_POOLING` global variable at `auxiliary/settings.py`

## Dataset

This implementation of FC4 has been tested against
the [Shi's Re-processing of Gehler's Raw Color Checker Dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/). After
downloading the data, please extract it and move the `images` and `coordinates` folders and the `folds.mat` file to
the `dataset` folder.

To preprocess the dataset, run the following commands:

```bash
cd dataset
python3 img2npy.py
```

This will mask the ground truth in the images and save the preprocessed items in `.npy` format into a new folder
called `preprocessed`.

Pretrained models on the 3 benchmark folds of this dataset are available at `trained_models`.

## Training

To train the FC4 model, run `python3 train.py`. The training procedure can be configured by editing the value of the
global variables at the beginning of the `train.py` file.

## Test

To test the FC4 model, run `python3 test.py`. The test procedure can be configured by editing the value of the global
variables at the beginning of the `test.py` file.

## Visualize confidence

To visualize the confidence weights learned by the FC4 model, run `python3 visualize.py`. The procedure can be
configured by editing the value of the global variables at the beginning of the `visualize.py` file.

This is the type of visualization produced by the script:

![stages](stages.png)