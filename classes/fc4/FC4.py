import torch
from classes.fc4.squeezenet.SqueezeNetLoader import SqueezeNetLoader
from torch import nn
from torch.nn.functional import normalize

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
            nn.Conv2d(64, 4, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate the semi-dense feature maps of shape [batch_size, 4, conv_out_width, conv_out_h]
        out = self.final_convs(self.backbone(x))

        # Multiply the per-patch color estimates (first 3 dimensions) by the their confidence (last dimension)
        rgb = normalize(out[:, :3, :, :], dim=1)
        confidence = out[:, 3:4, :, :]
        p = rgb * confidence

        # Summation and normalization
        return normalize(torch.sum(torch.sum(p, 2), 2), dim=1)
