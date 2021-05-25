import torch
from torch import nn
from torch.nn.functional import normalize

from auxiliary.settings import IMG_H, IMG_W


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3 * IMG_H * IMG_W, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        pred = self.layers(x)
        pred = normalize(pred, dim=1)
        return pred
