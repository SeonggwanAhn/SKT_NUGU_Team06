import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

# origin version
"""
class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
"""

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )   
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNet_5_Base(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(6, 16, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=2),
                nn.Conv2d(16, 120, kernel_size=(5,5)),
                nn.ReLU(),
                )
        self.in_features = 120*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))  # (5,5)
        self.relu3 = nn.ReLU()

        # self.in_features = 60*4*4
        self.in_features=60
        
    def forward(self, img):
        # print(img.size())
        output = self.conv1(img)  
        # print(output.size())
        
        output = self.relu1(output)
        # print(output.size())
        output = self.maxpool1(output)
        # print(output.size())
        output = self.conv2(output)
        # print(output.size())
        output = self.relu2(output)
        # print(output.size())
        output = self.maxpool2(output)
        # print(output.size())
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 60)
        # print('i want to know feature dim')
        # print(feature.size())
        return feature


class feat_linear(nn.Module):
    def __init__(self, input_dim, output_dim, type='bn'):
        super(feat_linear, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.type = type

    def forward(self, x):
        x = self.fc(x)
        feat = self.relu(x)  # exist in Lenet5Half
        
        if self.type=='bn':
            x = self.bn(x)
            x = self.dropout(x)
        
        return x


