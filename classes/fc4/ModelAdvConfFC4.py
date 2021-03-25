import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torch.nn.functional import normalize
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import correct, rescale, scale
from classes.fc4.FC4 import FC4


class ModelAdvConfFC4:

    def __init__(self, adv_lambda: float = 0.00005):
        self.__device = DEVICE
        self.__adv_lambda = adv_lambda
        self.__optimizer = None
        self.__optimizer_adv = None
        self.__network = FC4().to(self.__device)
        self.__network_adv = FC4().to(self.__device)
        self.__kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='sum').to(self.__device)

    def predict(self, img: Tensor) -> Tuple:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which a colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        return self.__network(img), self.__network_adv(img)

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

    def optimize(self, img: Tensor, label: Tensor) -> Tuple:
        self.__optimizer.zero_grad()
        self.__optimizer_adv.zero_grad()

        (pred, _, confidence), (pred_adv, _, confidence_adv) = self.predict(img)

        loss = self.get_angular_loss(pred.clone(), label)
        loss.backward(retain_graph=True)

        loss_adv = self.get_adv_loss(pred_adv, label, confidence, confidence_adv)
        loss_adv.backward()

        self.__optimizer.step()
        self.__optimizer_adv.step()

        return loss.item(), loss_adv.item()

    def get_adv_loss(self, pred: Tensor, label: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
        alpha = torch.Tensor([self.__adv_lambda]).to(self.__device)
        angular_loss = self.get_angular_loss(pred, label)
        kl_loss = self.get_kl_loss(c1, c2)
        return angular_loss - alpha * kl_loss

    def get_kl_loss(self, c1: Tensor, c2: Tensor):
        return self.__kl_loss(scale(c1), scale(c2)).to(self.__device)

    @staticmethod
    def get_angular_loss(pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)

    def print_network(self):
        print(self.__network)

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self.__network))

    def train_mode(self):
        self.__network = self.__network.train()
        self.__network_adv = self.__network_adv.train()

    def evaluation_mode(self):
        self.__network = self.__network.eval()
        self.__network_adv = self.__network_adv.eval()

    def save(self, path_to_log: str):
        torch.save(self.__network.state_dict(), os.path.join(path_to_log, "model.pth"))
        torch.save(self.__network_adv.state_dict(), os.path.join(path_to_log, "model_adv.pth"))

    def load(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained, "model.pth")
        self.__network.load_state_dict(torch.load(path_to_model, map_location=self.__device))
        path_to_model_adv = os.path.join(path_to_pretrained, "model_adv.pth")
        self.__network_adv.load_state_dict(torch.load(path_to_model_adv, map_location=self.__device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "sgd"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
        optimizer = optimizers_map[optimizer_type]
        self.__optimizer = optimizer(self.__network.parameters(), lr=learning_rate)
        self.__optimizer_adv = optimizer(self.__network_adv.parameters(), lr=learning_rate)
