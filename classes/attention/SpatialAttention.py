from torch import nn, Tensor


class SpatialAttention(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_size // 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_size // 2, input_size // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_size // 4),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_size // 4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Computes a spatial attention mask with values in [0, 1] for the given input image """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
