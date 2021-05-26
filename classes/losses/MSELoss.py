import torch
from torch import Tensor

from classes.core.Loss import Loss
from utils import scale


class MSELoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum').to(self._device)

    def _compute(self, c1: Tensor, c2: Tensor) -> Tensor:
        return self.__mse_loss(scale(c1), scale(c2)).to(self._device)
