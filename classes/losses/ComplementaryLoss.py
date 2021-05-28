import torch
from torch import Tensor

from auxiliary.utils import scale
from classes.core.Loss import Loss

""" https://ieeexplore.ieee.org/document/9380693 """


class ComplementaryLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__one = torch.Tensor([1]).to(self._device)

    def _compute(self, c1: Tensor, c2: Tensor) -> Tensor:
        return torch.norm(self.__one - scale(c1) + scale(c2), p=1).to(self._device)
