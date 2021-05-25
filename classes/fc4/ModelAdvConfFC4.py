import math
import os
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from pytorch_msssim import SSIM
from torch import Tensor
from torch.nn.functional import normalize
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import scale, rescale
from classes.fc4.FC4 import FC4


class ModelAdvConfFC4:

    def __init__(self, adv_lambda: float = 0.00005):
        self.__device = DEVICE
        self.__adv_lambda = torch.Tensor([adv_lambda]).to(self.__device)
        self.__optimizer = None
        self.__network_base = FC4().to(self.__device)
        self.__network_adv = FC4().to(self.__device)

        self.__ssim_loss = SSIM(data_range=1, channel=1)
        self.__mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum').to(self.__device)
        self.__kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='sum').to(self.__device)

        self.__eps = torch.Tensor([0.0000001])

    def predict(self, img: Tensor) -> Tuple:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which a colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        return self.__network_base(img), self.__network_adv(img)

    def optimize(self, pred_base: Tensor, pred_adv: Tensor, conf_base: Tensor, conf_adv: Tensor) -> Tuple:
        self.__optimizer.zero_grad()
        loss, losses = self.get_losses(conf_base, conf_adv, pred_base, pred_adv)
        loss.backward()
        self.__optimizer.step()
        return loss.item(), losses

    def get_losses(self, conf_base: Tensor, conf_adv: Tensor, pred_base: Tensor, pred_adv: Tensor) -> Tuple:
        losses = {
            "angular": self.get_angular_loss(pred_base, pred_adv),
            "ssim": self.get_ssim_loss(conf_base, conf_adv),
            "iou": self.get_iou_loss(conf_base, conf_adv),
            "complementary": self.get_complementary_loss(conf_base, conf_adv)
        }
        loss = losses["angular"] + self.__adv_lambda * (losses["ssim"] + losses["iou"] + losses["complementary"])
        return loss, losses

    def get_iou_loss(self, c1: Tensor, c2: Tensor) -> Tensor:
        c1, c2 = c1.int(), c2.int()
        intersection = (c1 & c2).float().sum((1, 2)).to(self.__device)
        union = (c1 | c2).float().sum((1, 2)).to(self.__device)
        return torch.mean((intersection + self.__eps) / (union + self.__eps)).to(self.__device)

    def get_complementary_loss(self, c1: Tensor, c2: Tensor) -> Tensor:
        """ https://ieeexplore.ieee.org/document/9380693 """
        return torch.norm(torch.Tensor([1]).to(self.__device) - scale(c1) + scale(c2), p=1).to(self.__device)

    def get_ssim_loss(self, c1: Tensor, c2: Tensor) -> Tensor:
        return self.__ssim_loss(scale(c1), scale(c2)).to(self.__device)

    def get_mse_loss(self, c1: Tensor, c2: Tensor) -> Tensor:
        return self.__mse_loss(scale(c1), scale(c2)).to(self.__device)

    def get_kl_loss(self, c1: Tensor, c2: Tensor) -> Tensor:
        return self.__kl_loss((scale(c1) + self.__eps).log(), scale(c2) + self.__eps).to(self.__device)

    @staticmethod
    def get_angular_loss(pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)

    def print_network(self):
        print(self.__network_base)

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self.__network_base))

    def train_mode(self):
        self.__network_base = self.__network_base.train()
        self.__network_adv = self.__network_adv.train()

    def evaluation_mode(self):
        self.__network_base = self.__network_base.eval()
        self.__network_adv = self.__network_adv.eval()

    def save_base(self, path_to_log: str):
        torch.save(self.__network_base.state_dict(), os.path.join(path_to_log, "model.pth"))

    def save_adv(self, path_to_log: str):
        torch.save(self.__network_adv.state_dict(), os.path.join(path_to_log, "model_adv.pth"))

    def load_base(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained, "model.pth")
        self.__network_base.load_state_dict(torch.load(path_to_model, map_location=self.__device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "sgd"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
        optimizer = optimizers_map[optimizer_type]
        self.__optimizer = optimizer(self.__network_adv.parameters(), lr=learning_rate)

    @staticmethod
    def save_vis(img: Tensor, conf_base: Tensor, conf_adv: Tensor, path_to_save: str):
        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        size = original.size[::-1]

        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(original)
        axs[0].set_title("Original")
        axs[0].axis("off")

        conf_base = rescale(conf_base.detach(), size).squeeze(0).permute(1, 2, 0)
        axs[1].imshow(conf_base, cmap="gray")
        axs[1].set_title("Base confidence")
        axs[1].axis("off")

        conf_adv = rescale(conf_adv.detach(), size).squeeze(0).permute(1, 2, 0)
        axs[2].imshow(conf_adv, cmap="gray")
        axs[2].set_title("Adv confidence")
        axs[2].axis("off")

        plt.savefig(path_to_save, bbox_inches='tight')
