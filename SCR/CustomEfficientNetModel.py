import torch
import torchvision.models
from torchvision import models
from torch import nn


class efficientnet_model(nn.Module):
    """
    Using EfficientNet as a backbone, this model add two regression head while retained the classification head. The
    model maybe used on pretrained or fully trainable mode
    """

    def __init__(self, pretrained=True):
        super(efficientnet_model, self).__init__()
        self.model = torchvision.models.efficientnet_b0(pretrained=pretrained).features
        self.layer_1 = nn.AdaptiveAvgPool2d(output_size=1)
        if pretrained == True:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model[-1].requires_grad = True
            self.model[-2].requires_grad = True
            self.model[-3].requires_grad = True
        self.classification = nn.Sequential(nn.Dropout(p=0.2, inplace=False),nn.Linear(in_features=1280, out_features=8, bias=True))
        self.regression = nn.Sequential(nn.Dropout(p=0.2, inplace=False),nn.Linear(in_features=1280, out_features=2, bias=True))

    def forward(self, x):
        feature = self.model(x)
        feature = self.layer_1(feature)
        feature = torch.flatten(feature, 1)
        classification = self.classification(feature)
        regression = self.regression(feature)
        arousal = regression[:, 0,None]
        valence = regression[:, 1,None]
        return classification, arousal, valence

# model=torchvision.models.efficientnet_b0()
# print(model)