'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import register_model
from models.op import MaskedConvBNReLU

ConvBNReLU = lambda in_planes, planes, kernel_size, stride, padding, groups=1, has_relu=True: \
                                                                        nn.Sequential(OrderedDict(
    [("conv", nn.Conv2d(in_planes, planes, kernel_size, stride,
                        padding, groups=groups, bias=False)),
     ("bn", nn.BatchNorm2d(planes))] + ([("relu", nn.ReLU(inplace=True))] if has_relu else [])))

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, cbr_block):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.mod1 = cbr_block(in_planes, planes, kernel_size=1, stride=1, padding=0)
        self.mod2_depthwise = cbr_block(planes, planes, kernel_size=3,
                                        stride=stride, padding=1, groups=planes)
        self.mod3 = cbr_block(planes, out_planes, kernel_size=1,
                              stride=1, padding=0, has_relu=False)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = cbr_block(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, has_relu=False)

    def forward(self, x):
        out = self.mod1(x)
        out = self.mod2_depthwise(out)
        out = self.mod3(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))
        # out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, block, num_classes=10):
        super(MobileNetV2, self).__init__()

        self.cbr_block = block

        self.mod1 = self.cbr_block(3, 32, kernel_size=3, stride=1, padding=1)
        self.layers = self._make_layers(in_planes=32)
        self.mod2 = self.cbr_block(320, 1280, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion,
                                    stride, cbr_block=self.cbr_block))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.mod1(x)
        out = self.layers(out)
        out = self.mod2(out)

        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

@register_model("mobilenetv2")
def mobilenetv2():
    return MobileNetV2(block=ConvBNReLU)

@register_model("mobilenetv2_masked")
def mobilenetv2_masked():
    return MobileNetV2(block=partial(MaskedConvBNReLU, plan=1, base_importance_strategy="bn"))

