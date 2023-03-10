import torch
import torch.nn as nn
import timm

class ResNet(nn.Module):
    def __init__(self, num_layer = 18, num_classes=10, pretrained=True):
        super(ResNet, self).__init__()
        if num_layer == 18:
            self.model = timm.create_model('resnet18', pretrained=pretrained)
        elif num_layer == 34:
            self.model = timm.create_model('resnet34', pretrained=pretrained)
        elif num_layer == 50:
            self.model = timm.create_model('resnet50', pretrained=pretrained)
        elif num_layer == 101:
            self.model = timm.create_model('resnet101', pretrained=pretrained)
        elif num_layer == 152:
            self.model = timm.create_model('resnet152', pretrained=pretrained)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    pass