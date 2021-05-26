import torch
from torch import Tensor

from classes.core.Loss import Loss


class TotalVariationLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, x: Tensor) -> Tensor:
        """
        Computes the total variation regularization (anisotropic version) for regularization of the learnable attention
        masks by encouraging spatial smoothness
        -> Reference: https://www.wikiwand.com/en/Total_variation_denoising
        @param x: the [1 x H x W] tensor for which the total variation must be computed
        """
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return (diff_i + diff_j).to(self._device)
