import torch
import torchvision.models
from torchvision import models
from torch import nn
from torchviz import make_dot


class senet_model(nn.Module):
    """
    Using SqueezeNet as a backbone, this model add two regression head while retained the classification head. The
    model maybe used on pretrained or fully trainable mode
    """

    def __init__(self, pretrained=True):
        super(senet_model, self).__init__()
        self.model = torchvision.models.squeezenet1_1(pretrained=pretrained).features
        self.layer_1 = nn.AdaptiveAvgPool2d(output_size=1)
        if pretrained == True:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model[-1].requires_grad = True
            self.model[-2].requires_grad = True
            self.model[-3].requires_grad = True
        self.classification = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                            nn.Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1)),
                                            nn.ReLU(inplace=True),
                                            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.regression = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                        nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)),
                                        nn.ReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        feature = self.model(x)
        feature = self.layer_1(feature)
        # feature = torch.flatten(feature, 1)
        classification = self.classification(feature)
        arousal = self.regression(feature)
        valence = self.regression(feature)
        return torch.flatten(classification, 1), torch.flatten(arousal, 1), torch.flatten(valence, 1)


#
if __name__ == "__main__":
    models = torchvision.models.squeezenet1_1()
    print(models)
