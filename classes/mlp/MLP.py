import torch
from torch import nn, tanh
from torch.nn.functional import normalize

from auxiliary.settings import TRAIN_IMG_H, TRAIN_IMG_W


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3 * TRAIN_IMG_H * TRAIN_IMG_W, 961)
        self.output_layer = nn.Linear(961, 3)

    def impose_weights(self, w: torch.Tensor):
        if w is None:
            raise ValueError("Cannot impose None weights to MLP output layer!")
        with torch.no_grad():
            self.output_layer.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = tanh(x)
        x = self.output_layer(x)
        x = tanh(x)
        pred = normalize(x, dim=1)
        return pred
