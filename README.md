# pytorch-fc4
A PyTorch implementation of FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling

The original code for the FC4 method is quite outdated (is based on Python 2 and an old version of Tensorflow). This an attempt of providing a clean and modern reimplementation of that method using the PyTorch library.

## FC4
Original resources:
* Code: https://github.com/yuanming-hu/fc4
* Paper: https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/

## SqueezeNet

This implementation of the FC4 method uses SqueezeNet. The SqueezeNet implementation is the one offered by PyTorch at:
https://github.com/pytorch/vision/blob/072d8b2280569a2d13b91d3ed51546d201a57366/torchvision/models/squeezenet.py
* **SqueezeNet 1.0**. Architecture from https://arxiv.org/abs/1602.07360, 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'
* **SqueezeNet 1.1** (has 2.4x less computation and slightly fewer parameters than 1.0, without sacrificing accuracy). Architecture from: <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`

