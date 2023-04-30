import torch
from torch import nn


class Talha_net(nn.Module):
    """
    Using a CUSTOM Architecture as a backbone, this model add two regression head while retained the classification head. The
    model maybe used on pretrained or fully trainable mode
    """
    def __init__(self):
        super(Talha_net, self).__init__()

        self.stage1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(4), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(8), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.stage4 = nn.Sequential(nn.Conv2d(in_channels=28, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

        self.adaptivepool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.classification = nn.Sequential(nn.Linear(in_features=800, out_features=256, bias=True),
                                            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                            nn.Dropout(p=0.5, inplace=False),
                                            nn.Linear(in_features=256, out_features=8))
        self.regression = nn.Sequential(nn.Linear(in_features=800, out_features=256, bias=True),
                                        nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(in_features=256, out_features=2))

    # Defining the forward pass
    def forward(self, x):
        feature_map1 = self.stage1(x)
        feature_map2 = self.stage2(feature_map1)
        feature_map3 = self.stage3(feature_map2)
        f1 = self.down_sample(self.down_sample(feature_map1))
        f2 = self.down_sample(feature_map2)

        feature_map_concat = torch.cat((feature_map3, f2, f1), dim=1)
        feature = self.stage4(feature_map_concat)
        feature = self.adaptivepool(feature)
        feature = torch.flatten(feature, 1)
        classification = self.classification(feature)
        regression = self.regression(feature)
        arousal = regression[:, 0, None]
        valence = regression[:, 1, None]
        return classification, arousal, valence
