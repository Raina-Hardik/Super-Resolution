import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

from core.config import settings


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(settings.device)
        self.loss = nn.MSELoss()

        for params in self.vgg.parameters():
            params.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)
