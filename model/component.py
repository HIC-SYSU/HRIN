import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoding(nn.Module):
    def __init__(self, feature_root=32, depth=5):
        super(Encoding, self).__init__()
        self.depth = depth

        self.en_block = nn.Sequential(
            nn.Conv2d(1, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True))

        self.modulelist_en_block = nn.ModuleList()
        for i in range(1, self.depth):
            self.modulelist_en_block.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(np.power(2, i - 1) * feature_root, np.power(2, i) * feature_root, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(np.power(2, i) * feature_root),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(np.power(2, i) * feature_root, np.power(2, i) * feature_root, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(np.power(2, i) * feature_root),
                    nn.ReLU(inplace=True)))

        self.en_mapping = nn.Sequential(
            nn.Conv2d(np.power(2, self.depth - 1) * feature_root, np.power(2, self.depth - 2) * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True))

    def forward(self, x):
        temp_info = []
        temp_block = self.en_block(x)
        temp_info.append(temp_block)
        for i in range(self.depth - 1):
            temp_block = self.modulelist_en_block[i](temp_block)
            temp_info.append(temp_block)
        # temp_block = self.en_mapping(temp_block)
        return temp_info


class AUEncoding(nn.Module):
    def __init__(self, feature_root=32, depth=5):
        super(AUEncoding, self).__init__()
        self.depth = depth

        self.en_block1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True))

        self.en_block2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(2 * feature_root, 4 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True))

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_block1 = nn.Sequential(
            nn.Conv2d(4 * feature_root, 2 * feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * feature_root, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.de_block2 = nn.Sequential(
            nn.Conv2d(2 * feature_root, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        fea1 = self.en_block1(x)
        fea2 = self.en_block2(fea1)
        up1 = self.up1(fea2)
        fea3 = self.de_block1(torch.cat([up1, fea1], dim=1))
        up2 = self.up1(fea3)
        fea4 = self.de_block2(torch.cat([up2, x], dim=1))
        return fea4


class Upsampling(nn.Module):
    def __init__(self, feature_root=32, depth=5):
        super(Upsampling, self).__init__()
        self.depth = depth

        self.de_block = nn.Sequential(
            nn.Conv2d(feature_root * 16, feature_root * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root * 8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(feature_root * 16, feature_root * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root * 8, feature_root * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root * 4),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(feature_root * 8, feature_root * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root * 4, feature_root * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root * 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(feature_root * 4, feature_root * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root * 2, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(feature_root * 2, feature_root, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_root),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_root, feature_root, kernel_size=3, stride=1, padding=1))

    def forward(self, x, skip_information):
        x = self.de_block(x)
        x = torch.cat([x, skip_information[-2]], dim=1)

        x = self.de_block1(x)
        x = torch.cat([x, skip_information[-3]], dim=1)

        x = self.de_block2(x)
        x = torch.cat([x, skip_information[-4]], dim=1)

        x = self.de_block3(x)
        x = torch.cat([x, skip_information[-5]], dim=1)

        x = self.de_block4(x)
        return x
