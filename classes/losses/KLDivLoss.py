import torch
from torch import Tensor

from classes.core.Loss import Loss
from utils import scale


class KLDivLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='sum').to(self._device)
        self.__eps = torch.Tensor([0.0000001])

    def _compute(self, c1: Tensor, c2: Tensor) -> Tensor:
        return self.__kl_loss((scale(c1) + self.__eps).log(), scale(c2) + self.__eps).to(self._device)
