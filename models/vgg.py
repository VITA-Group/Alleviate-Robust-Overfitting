'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from advertorch.utils import NormalizeByChannelMeanStd

__all__ = ['VGG', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

        # default normalization is for CIFAR10
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    def forward(self, x):
        x = self.normalize(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def vgg11_bn(num_classes):
    
    return VGG('VGG11', num_classes=num_classes)

def vgg13_bn(num_classes):
    
    return VGG('VGG13', num_classes=num_classes)

def vgg16_bn(num_classes):
    
    return VGG('VGG16', num_classes=num_classes)

def vgg19_bn(num_classes):
    
    return VGG('VGG19', num_classes=num_classes)


