import torch
import torch.nn as nn

kernel_size = 3
padding=1

class ResBlock(nn.Module):
    def __init__(self, in_channels = 256):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, 
                               in_channels = in_channels, 
                               kernel_size = kernel_size, 
                               padding = padding)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x *= 0.1                    # Scaling factor
        x = torch.add(x, identity)
        return x


class Upsampling(nn.Module):
    def __init__(self, in_channels, factor=2):
        super(Upsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64 * (factor ** 2), kernel_size=3, padding=1)
        self.depth_to_space = nn.PixelShuffle(factor)
        self.conv2 = nn.Conv2d(64, 64 * (factor ** 2), kernel_size=3, padding=1)
        self.depth_to_space2 = nn.PixelShuffle(factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depth_to_space(x)
        x = self.conv2(x)
        x = self.depth_to_space2(x)
        return x

