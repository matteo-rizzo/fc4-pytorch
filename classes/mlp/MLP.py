import torch
from torch import nn, tanh
from torch.nn.functional import normalize

from auxiliary.settings import TRAIN_IMG_H, TRAIN_IMG_W
from classes.attention.SpatialAttention import SpatialAttention


class MLP(nn.Module):

    def __init__(self, input_size: int = 961, learn_attention: bool = False):
        super().__init__()
        self.input_layer = nn.Linear(3 * TRAIN_IMG_H * TRAIN_IMG_W, input_size)
        self.output_layer = nn.Linear(input_size, 3)
        self.__learn_attention = learn_attention
        if self.__learn_attention:
            self.attention = SpatialAttention(input_size=512)

    def __impose_attention(self, x: torch.Tensor, w: torch.Tensor = None):
        if w is None and self.__learn_attention:
            x_temp = x.view(x.shape[0], int(x.shape[1] ** (1 / 2)), int(x.shape[1] ** (1 / 2)))
            x_temp = x_temp.unsqueeze(1).expand(-1, 512, -1, -1)
            w = self.attention(x_temp)
            w = w.squeeze(1)
            w = w.view(w.shape[0], w.shape[1] * w.shape[2])
        if w is not None:
            x = x * w
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        x = self.input_layer(x)
        x = tanh(x)
        x = self.__impose_attention(x, w)
        x = self.output_layer(x)
        x = tanh(x)
        pred = normalize(x, dim=1)
        return pred
