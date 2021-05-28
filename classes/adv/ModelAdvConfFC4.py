import os
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import transforms

from auxiliary.utils import rescale
from classes.core.Model import Model
from classes.fc4.FC4 import FC4
from classes.losses.ComplementaryLoss import ComplementaryLoss
from classes.losses.IoULoss import IoULoss
from classes.losses.SSIMLoss import SSIMLoss


class ModelAdvConfFC4(Model):

    def __init__(self, adv_lambda: float = 0.00005):
        super().__init__()
        self.__adv_lambda = torch.Tensor([adv_lambda]).to(self._device)
        self._network = FC4().to(self._device)
        self.__network_adv = FC4().to(self._device)
        self.__ssim_loss = SSIMLoss(self._device)
        self.__iou_loss = IoULoss(self._device)
        self.__complementary_loss = ComplementaryLoss(self._device)

    def predict(self, img: Tensor) -> Tuple:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which a colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        return self._network(img), self.__network_adv(img)

    def optimize(self, pred_base: Tensor, pred_adv: Tensor, conf_base: Tensor, conf_adv: Tensor) -> Tuple:
        self._optimizer.zero_grad()
        loss, losses = self.get_losses(conf_base, conf_adv, pred_base, pred_adv)
        loss.backward()
        self._optimizer.step()
        return loss.item(), losses

    def get_losses(self, conf_base: Tensor, conf_adv: Tensor, pred_base: Tensor, pred_adv: Tensor) -> Tuple:
        losses = {
            "angular": self._criterion(pred_base, pred_adv),
            "ssim": self.__ssim_loss(conf_base, conf_adv),
            "iou": self.__iou_loss(conf_base, conf_adv),
            "complementary": self.__complementary_loss(conf_base, conf_adv)
        }
        loss = losses["angular"] + self.__adv_lambda * (losses["ssim"] + losses["iou"] + losses["complementary"])
        return loss, losses

    def train_mode(self):
        self._network = self._network.train()
        self.__network_adv = self.__network_adv.train()

    def evaluation_mode(self):
        self._network = self._network.eval()
        self.__network_adv = self.__network_adv.eval()

    def save_adv(self, path_to_log: str):
        torch.save(self.__network_adv.state_dict(), os.path.join(path_to_log, "model_adv.pth"))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "sgd"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
        optimizer = optimizers_map[optimizer_type]
        self._optimizer = optimizer(self.__network_adv.parameters(), lr=learning_rate)

    @staticmethod
    def save_vis(img: Tensor, conf_base: Tensor, conf_adv: Tensor, path_to_save: str):
        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        size = original.size[::-1]

        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(original)
        axs[0].set_title("Original")
        axs[0].axis("off")

        conf_base = rescale(conf_base.detach().cpu(), size).squeeze(0).permute(1, 2, 0)
        axs[1].imshow(conf_base, cmap="gray")
        axs[1].set_title("Base confidence")
        axs[1].axis("off")

        conf_adv = rescale(conf_adv.detach().cpu(), size).squeeze(0).permute(1, 2, 0)
        axs[2].imshow(conf_adv, cmap="gray")
        axs[2].set_title("Adv confidence")
        axs[2].axis("off")

        plt.savefig(path_to_save, bbox_inches='tight')
