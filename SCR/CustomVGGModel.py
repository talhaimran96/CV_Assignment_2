# Using the VGG16 model as a backbone, this class is designed to compute the classification of the face, Regress the
# Arousal and Valence values this is done by adding regression heads to the model in the form of linear layers

import torch
import torchvision.models
from torchvision import models
from torch import nn


class VGG16_model(nn.Module):
    """
    Using VGG16_BN as a backbone, this model add two regression head while retained the classification head. The
    model maybe used on pretrained or fully trainable mode
    """

    def __init__(self, pretrained=True):
        super(VGG16_model, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=pretrained).features
        self.layer_1 = nn.AdaptiveAvgPool2d((7, 7))
        if pretrained == True:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model[-1].requires_grad = True
            self.model[-2].requires_grad = True
            self.model[-3].requires_grad = True
        self.classification = nn.Sequential(nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True),
                                            nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
                                            nn.Linear(in_features=4096, out_features=4096, bias=True),
                                            nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
                                            nn.Linear(in_features=4096, out_features=8, bias=True))
        self.regression = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                        nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(in_features=4096, out_features=1024, bias=True),
                                        nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(in_features=1024, out_features=1, bias=True))


    def forward(self, x):
        features = self.model(x)
        features = self.layer_1(features)
        feature = torch.flatten(features, 1)
        classification = self.classification(feature)
        arousal = self.regression(feature)
        valence = self.regression(feature)
        return classification, arousal, valence
