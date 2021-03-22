import math
import os
from typing import Union

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torch.linalg import norm
from torch.nn.functional import normalize, unfold
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE, USE_CONFIDENCE_WEIGHTED_POOLING
from auxiliary.utils import correct, rescale, scale
from classes.fc4.FC4 import FC4


class ModelFC4:

    def __init__(self):
        self.__device = DEVICE
        self.__optimizer = None
        self.__network = FC4().to(self.__device)

    def predict(self, img: Tensor, return_steps: bool = False) -> Union[Tensor, tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which a colour of the illuminant has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, rgb, confidence = self.__network(img)
            if return_steps:
                return pred, rgb, confidence
            return pred
        return self.__network(img)

    def save_vis(self, model_output: dict, path_to_plot: str):
        model_output = {k: v.clone().detach().to(DEVICE) for k, v in model_output.items()}

        img, label, pred = model_output["img"], model_output["label"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]
        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)
        masked_original = scale(F.to_tensor(original).to(DEVICE).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_angular_loss(pred, label)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')

    def optimize(self, img: Tensor, label: Tensor) -> float:
        self.__optimizer.zero_grad()

        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, _, confidence = self.predict(img, return_steps=True)
            loss = self.get_regularized_loss(pred, label, confidence)
        else:
            pred = self.predict(img)
            loss = self.get_angular_loss(pred, label)

        loss.backward()
        self.__optimizer.step()
        return loss.item()

    @staticmethod
    def get_angular_loss(pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)

    @staticmethod
    def get_sparsity_reg_loss(x: Tensor, n: int = 3, s: int = 1, alpha: float = 0.0001) -> Tensor:
        """
        Computes the block-wise sparsity regularization loss based on the assumption that the pixels of relevant regions
        are not randomly distributed in spatial domain (they are likely to be located in connected regions with similar
        blob-type shape)
        @param x: the [1 x H x W] tensor for which the sparsity regularization must be computed
        @param n: the length of the side of the square block to be considered
        @param s: the stride between blocks
        @param alpha: a weight balancing the contribution of the regularization term to the overall loss
        """
        reg = norm(norm(unfold(x, kernel_size=n, stride=n + s), ord=2, dim=1), ord=1, dim=1)
        return alpha * reg

    @staticmethod
    def get_total_variation_loss(x: Tensor, alpha: float = 0.00001) -> Tensor:
        """
        Computes the total variation regularization (anisotropic version) for regularization of the learnable attention
        masks by encouraging spatial smoothness
        -> Reference: https://www.wikiwand.com/en/Total_variation_denoising
        @param x: the [1 x H x W] tensor for which the total variation must be computed
        @param alpha: a weight balancing the contribution of the regularization term to the overall loss
        """
        diff_i = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        return alpha * (diff_i + diff_j)

    def get_regularized_loss(self, pred: Tensor, label: Tensor, attention_mask: Tensor) -> Tensor:
        angular_loss = self.get_angular_loss(pred, label)
        sparsity_loss = self.get_sparsity_reg_loss(attention_mask)
        # total_variation_loss = self.get_total_variation_loss(attention_mask)
        # return angular_loss + total_variation_loss + sparsity_loss
        return angular_loss + sparsity_loss

    def print_network(self):
        print(self.__network)

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self.__network))

    def train_mode(self):
        self.__network = self.__network.train()

    def evaluation_mode(self):
        self.__network = self.__network.eval()

    def save(self, path_to_file: str):
        torch.save(self.__network.state_dict(), path_to_file)

    def load(self, path_to_pretrained: str):
        self.__network.load_state_dict(torch.load(path_to_pretrained, map_location=self.__device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "adam"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self.__optimizer = optimizers_map[optimizer_type](self.__network.parameters(), lr=learning_rate)
