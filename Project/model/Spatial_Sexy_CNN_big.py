"""
CNN with 3 conv layers and a fully connected classification layer
"""
import torch
import torch.nn as nn


class Spatial_Sexy_CNN_big(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
    :var conv2   : torch.nn.Conv2d
    :var conv3   : torch.nn.Conv2d
        The first three convolutional layers of the network

    :var fc      : torch.nn.Linear
        Final fully connected layer
    """

    def __init__(self, num_classes):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(Spatial_Sexy_CNN_big, self).__init__()
        # First layer small 24, alex 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=7, stride=3),
            nn.Softsign()
        )

        self.pool1 = nn.AvgPool2d(3)
        self.alpha1 = nn.Parameter(torch.ones(1, 1, 28, 28))
        self.upscale1 = nn.Upsample(scale_factor=3)

        # Second layer small 48, alex 192
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=5, stride=1),
            nn.Softsign()
        )
        # Polling layer
        self.pool2 = nn.AvgPool2d(3)
        self.alpha2 = nn.Parameter(torch.ones(1, 1, 8, 8))
        self.upscale2 = nn.Upsample(scale_factor=3)

        # Third layer small 72, alex 384
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, stride=1, padding=1),
            nn.Softsign()
        )

        # Fourth layer small 48, alex 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 48, kernel_size=3, stride=1, padding=1),
            nn.Softsign()
        )

        # Fifth layer small 48, alex 256
        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Softsign()
        )

        self.pool3 = nn.AvgPool2d(2)
        self.alpha3 = nn.Parameter(torch.ones(1, 1, 4, 4))
        self.upscale3 = nn.Upsample(scale_factor=2)

        # Classification layer small 768, alex 4096
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.pool1((self.upscale1(self.alpha1).mul(x)).exp()).log().mul(self.alpha1.pow(-1))
        x = self.conv2(x)
        x = self.pool2((self.upscale2(self.alpha2).mul(x)).exp()).log().mul(self.alpha2.pow(-1))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3((self.upscale3(self.alpha3).mul(x)).exp()).log().mul(self.alpha3.pow(-1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
