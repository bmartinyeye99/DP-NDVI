import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# Convolution Process
#     Input Size: Each patch is of size [7, 64, 64] (7 channels, 64x64 spatial dimensions).

#     Layer 1:
#         Convolution: nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1)
#         Input: [7, 64, 64]
#         Output: [32, 64, 64] (32 feature maps, spatial size preserved due to padding=1)

#     Layer 2:
#         Convolution: nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         Input: [32, 64, 64]
#         Output: [64, 64, 64] (64 feature maps, spatial size preserved)

#     Layer 3:
#         Convolution: nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
#         Input: [64, 64, 64]
#         Output: [1, 64, 64] (1 channel for NDVI prediction, spatial size preserved)
# Padding
#     Padding ensures that the spatial dimensions of the output feature maps remain the same as the input. For example:
#         Input: [64, 64]
#         Kernel: 3x3
#         Padding: 1
#         Output: [64, 64]
# CNN model
class NDVICNN(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.2):
        super(NDVICNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        # Replace ReLU with PReLU
        self.prelu1 = nn.PReLU()  # After conv1
        self.prelu2 = nn.PReLU()  # After conv2
        
        # Use Tanh to constrain the output to [-1,1]
        self.output_activation = nn.Tanh()
        
        # Initialize weights using Xavier (Glorot) initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.dropout1(x)
        x = self.prelu2(self.conv2(x))
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.output_activation(x)
        return x

    def _initialize_weights(self):
        # Loop over all modules and initialize Conv2d layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier normal initialization: mean=0, standard deviation determined by fan_in and fan_out.
                # For PReLU, a gain of 1.0 is commonly used.
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)