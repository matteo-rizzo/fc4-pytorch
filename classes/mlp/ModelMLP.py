import os

import torch
from torch import Tensor

from classes.core.Model import Model
from classes.fc4.FC4 import FC4
from classes.mlp.MLP import MLP


class ModelMLP(Model):

    def __init__(self, imposed_weights: str = None):
        super().__init__()
        self.__imposed_weights = imposed_weights
        if imposed_weights == "confidence":
            self.__attention_net = FC4().to(self._device)
        self._network = MLP(learn_attention=imposed_weights == "learned").to(self._device)

    def predict(self, img: Tensor) -> Tensor:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        x = img.view(img.shape[0], img.shape[1] * img.shape[2] * img.shape[3])

        w = None
        if self.__imposed_weights == "confidence":
            _, _, conf = self.__attention_net(img)
            w = conf.detach().squeeze(0).expand(3, -1, -1)
            w = w.view(w.shape[0], w.shape[1] * w.shape[2]).to(self._device)
        if self.__imposed_weights == "uniform":
            w = torch.rand((3, 961)).to(self._device)

        return self._network(x, w)

    def optimize(self, img: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(img)
        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def load_attention_net(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained, "model.pth")
        self.__attention_net.load_state_dict(torch.load(path_to_model, map_location=self._device))
