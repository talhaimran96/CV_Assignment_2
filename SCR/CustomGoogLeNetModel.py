import torch
import torchvision.models
from torchvision import models
from torch import nn


class googlenet_model(nn.Module):
    """
    Using MobileNetV3 as a backbone, this model add two regression head while retained the classification head. The
    model maybe used on pretrained or fully trainable mode
    """

    def __init__(self, pretrained=True):
        super(googlenet_model, self).__init__()
        self.model = torchvision.models.googlenet(pretrained=pretrained).features
        self.layer_1 = nn.AdaptiveAvgPool2d(output_size=1)
        if pretrained == True:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model[-1].requires_grad = True
            self.model[-2].requires_grad = True
            self.model[-3].requires_grad = True
        self.classification = nn.Sequential(nn.Linear(in_features=576, out_features=1024, bias=True),
                                            nn.Hardswish(), nn.Dropout(p=0.2, inplace=True),
                                            nn.Linear(in_features=1024, out_features=8))
        self.regression = nn.Sequential(nn.Linear(in_features=576, out_features=1024, bias=True),
                                        nn.Hardswish(), nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(in_features=1024, out_features=1))

    def forward(self, x):
        features = self.model(x)
        features = self.layer_1(features)
        feature = torch.flatten(features, 1)
        classification = self.classification(feature)
        arousal = self.regression(feature)
        valence = self.regression(feature)
        return classification, arousal, valence


models = torchvision.models.googlenet()
print(models)
