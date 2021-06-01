import torch
from torch import nn
from torch.nn.functional import normalize, relu

from auxiliary.settings import IMG_H, IMG_W


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3 * IMG_H * IMG_W, 128)
        self.hidden_layer = nn.Linear(128, 961)
        self.output_layer = nn.Linear(961, 3)

    def forward(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        if w is not None:
            w = w.squeeze(0).expand(3, -1, -1)
            w = w.view(w.shape[0], w.shape[1] * w.shape[2])
            self.output_layer.weight = nn.Parameter(w)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        pred = self.output_layer(relu(self.hidden_layer(relu(self.input_layer(x)))))
        pred = normalize(pred, dim=1)
        return pred
