import torch
from torch import nn
from torch.nn.functional import normalize

from classes.fc4.squeezenet.SqueezeNetLoader import SqueezeNetLoader

"""
FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling
* Original code: https://github.com/yuanming-hu/fc4
* Paper: https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/
"""


class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.backbone(x)
        o = self.final_convs(e)
        return normalize(torch.sum(torch.sum(o, 2), 2) + 1e-10, dim=1)
