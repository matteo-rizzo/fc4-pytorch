import torch
from torch import Tensor
from torch.linalg import norm
from torch.nn.functional import unfold

from classes.core.Loss import Loss


class BlockSparsityLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, x: Tensor, n: int = 3, s: int = 1) -> Tensor:
        """
        Computes the block-wise sparsity regularization loss based on the assumption that the pixels of relevant regions
        are not randomly distributed in spatial domain (they are likely to be located in connected regions with similar
        blob-type shape)
        @param x: the [1 x H x W] tensor for which the sparsity regularization must be computed
        @param n: the length of the side of the square block to be considered
        @param s: the stride between blocks
        """
        return norm(norm(unfold(x, kernel_size=n, stride=n + s), ord=2, dim=1), ord=1, dim=1).to(self._device)
