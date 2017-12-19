"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn


class Pooled_CNN_big(nn.Module):
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
        super(Pooled_CNN_big, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=7, stride=3),
            nn.Softsign()
        )

        self.pool1 = nn.MaxPool2d(3)

        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=5, stride=1),
            nn.Softsign()
        )
        # Polling layer
        self.pool2 = nn.MaxPool2d(3)

        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, stride=1, padding=1),
            nn.Softsign()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 48, kernel_size=3, stride=1, padding=1),
            nn.Softsign()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Softsign()
        )

        self.pool3 = nn.MaxPool2d(2)

        # Classification layer
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
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
