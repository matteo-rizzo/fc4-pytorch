import math
import os
from typing import Tuple

import torch
from torch import Tensor
from torch.nn.functional import normalize

from auxiliary.settings import DEVICE
from auxiliary.utils import scale
from classes.fc4.FC4 import FC4


class ModelAdvConfFC4:

    def __init__(self, adv_lambda: float = 0.00005):
        self.__device = DEVICE
        self.__adv_lambda = torch.Tensor([adv_lambda]).to(self.__device)
        self.__optimizer = None
        self.__network_base = FC4().to(self.__device)
        self.__network_adv = FC4().to(self.__device)
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

    def optimize(self, img: Tensor) -> Tuple:
        self.__optimizer.zero_grad()
        (pred_base, _, conf_base), (pred_adv, _, conf_adv) = self.predict(img)
        loss, angular_loss, kl_loss = self.get_losses(conf_base, conf_adv, pred_base, pred_adv)
        loss.backward()
        self.__optimizer.step()
        return pred_base, pred_adv, loss.item(), angular_loss.item(), kl_loss.item()

    def get_losses(self, conf_base: Tensor, conf_adv: Tensor, pred_base: Tensor, pred_adv: Tensor) -> Tuple:
        angular_loss = self.get_angular_loss(pred_base, pred_adv)
        kl_loss = self.get_kl_loss(conf_base, conf_adv)
        loss = angular_loss - self.__adv_lambda * kl_loss
        return loss, angular_loss, kl_loss

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
