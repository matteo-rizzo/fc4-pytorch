import math
import os
from typing import Union

import torch
from torch.nn.functional import normalize

from auxiliary.settings import DEVICE, USE_CONFIDENCE_WEIGHTED_POOLING
from classes.fc4.FC4 import FC4


class ModelFC4:

    def __init__(self):
        self.__device = DEVICE
        self.__optimizer = None
        self.__network = FC4().to(self.__device)

    def predict(self, img: torch.Tensor, return_steps: bool = False) -> Union[torch.Tensor, tuple]:
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

    def compute_loss(self, img: torch.Tensor, label: torch.Tensor) -> float:
        pred = self.predict(img)
        loss = self.get_angular_loss(pred, label)
        loss.backward()
        return loss.item()

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

    def reset_gradient(self):
        self.__optimizer.zero_grad()

    def optimize(self):
        self.__optimizer.step()

    @staticmethod
    def get_angular_loss(pred: torch.Tensor, label: torch.Tensor, safe_v: float = 0.999999) -> torch.Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)
