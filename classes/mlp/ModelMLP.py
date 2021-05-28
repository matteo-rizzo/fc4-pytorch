from torch import Tensor

from classes.core.Model import Model
from classes.fc4.FC4 import FC4
from classes.mlp.MLP import MLP


class ModelMLP(Model):

    def __init__(self, imposed_weights: bool = False):
        super().__init__()
        self._network = MLP().to(self._device)
        self.__imposed_weights = imposed_weights
        if imposed_weights:
            self.__attention_net = FC4().to(self._device)

    def predict(self, img: Tensor) -> Tensor:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if self.__imposed_weights:
            _, _, conf = self.__attention_net(img)
            return self._network(img, conf)

        return self._network(img)

    def optimize(self, img: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(img)
        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()
