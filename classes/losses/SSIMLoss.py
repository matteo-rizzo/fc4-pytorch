import torch
from pytorch_msssim import SSIM
from torch import Tensor

from auxiliary.utils import scale
from classes.core.Loss import Loss

""" https://github.com/VainF/pytorch-msssim """


class SSIMLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__one = torch.Tensor([1]).to(self._device)
        self.__ssim_loss = SSIM(data_range=1, channel=1)

    def _compute(self, c1: Tensor, c2: Tensor) -> Tensor:
        return self.__one - self.__ssim_loss(scale(c1), scale(c2)).to(self._device)
