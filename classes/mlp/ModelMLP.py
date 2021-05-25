from torch import Tensor

from classes.mlp.MLP import MLP
from core.Model import Model


class ModelMLP(Model):

    def __init__(self):
        super().__init__()
        self._network = MLP().to(self._device)

    def predict(self, img: Tensor) -> Tensor:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        return self._network(img)

    def optimize(self, img: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(img)
        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()
