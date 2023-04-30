import torch
import torchvision.models
from torchvision import models
from torch import nn


class mobilenet_model(nn.Module):
    """
    Using MobileNetV3 as a backbone, this model add two regression head while retained the classification head. The
    model maybe used on pretrained or fully trainable mode
    """

    def __init__(self, pretrained=True):
        super(mobilenet_model, self).__init__()
        self.model = torchvision.models.mobilenet_v3_small(pretrained=pretrained).features
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
                                            nn.Linear(in_features=1024, out_features=2))

    def forward(self, x):
        features = self.model(x)
        features = self.layer_1(features)
        feature = torch.flatten(features, 1)
        classification = self.classification(feature)
        regression = self.regression(feature)
        arousal = regression[:, 0]
        valence = regression[:, 1]
        return classification, arousal, valence


# models = torchvision.models.mobilenet_v3_small(pretrained=True)
# print(models)
if __name__ == "__main__":
    model = mobilenet_model()
    print(model)
    x, y, z = model(torch.rand(size=[16, 3, 224, 224]))
    print(x.size(), y.size(), z.size())
    print(z[:, 0, None].size())