import torch
from torch import nn, Tensor


class TemporalAttention(nn.Module):

    def __init__(self, features_size: int = 512, hidden_size: int = 128):
        super().__init__()
        self.phi_x = nn.Linear(features_size, 1, bias=False)
        self.phi_h = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def __generate_energy(self, masked_features: Tensor, hidden: Tensor) -> Tensor:
        """
        Computes energy as e_ti = phi(H_t-1, masked_X_i) = phi(H_t-1) + phi(masked_X_i)
        @param masked_features: the i-th masked spatial features map
        @param hidden: the hidden states of the RNN at time step t-1
        @return: the energy for the i-th attended frame at time step t-1,
        """
        att_x = self.phi_x(torch.mean(torch.mean(masked_features, dim=2), dim=1))
        att_h = self.phi_h(torch.mean(torch.mean(hidden, dim=3), dim=2))
        e = att_x + att_h
        return e

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        @param x: the sequences of frames of shape "ts x nc x h x w"
        @param h: the hidden state of an RNN
        @return: the normalized illuminant prediction
        """
        ts = x.shape[0]

        # Compute energies as e_ti = phi(H_t-1, masked_X_i)
        energies = torch.cat([self.__generate_energy(x[i, :, :, :], h) for i in range(ts)], dim=0)

        # Energies to temporal weights via softmax: w_ti = exp(e_ti)/sum_i^n(exp(e_ti))
        weights = self.softmax(energies).unsqueeze(1).unsqueeze(2)

        return weights
