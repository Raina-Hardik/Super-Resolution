import torch
import torch.nn as nn

kernel_size = 3
padding=1

class ResBlock(nn.Module):
    def __init__(self, channels = 256):
        super(ResBlock, self).__init__()

        self.conv = nn.Conv2d( in_channels = channels, 
                               out_channels = channels, 
                               kernel_size = kernel_size, 
                               padding = padding)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x                # Trunk/Main branch
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x *= 0.1                    # Scaling factor
        x = torch.add(x, identity)  # Merge branch back
        return x

class EDSR(nn.Module):
    def __init__(self, channels = 256, num_blocks = 32):
        super(EDSR, self).__init__()

        col = 3

        self.conv1 = nn.Conv2d(in_channels = col,               # Input Conv
                               out_channels = channels, 
                               kernel_size = kernel_size, 
                               padding = 1)
        
        self.conv_out = nn.Conv2d(in_channels = channels, 
                                  out_channels = col, 
                                  kernel_size = kernel_size, 
                                  padding = 1)
        
        self.conv_up = nn.Conv2d(in_channels = channels,        # For upsampling
                                 out_channels = channels * 4, 
                                 kernel_size = kernel_size, 
                                 padding = 1)

        # A range of ResBlocks in a sequential format
        self.body = nn.Sequential(*(ResBlock() for _ in range(num_blocks)))

        self.upsample = nn.Sequential(self.conv_up, nn.PixelShuffle(2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.body(out)
        out = self.upsample(out)
        out = self.conv_out(out)

        return out
    