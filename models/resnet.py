'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import copy
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import register_model
from functools import partial, wraps
from .op import MaskedResBlock, MaskedConvBNReLU, MaskedBottleneckBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob=None):
        super(BasicBlock, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        p = np.random.uniform()
        if self.drop_prob is not None and self.training and p < self.drop_prob.item():
            return x
        else:
            return self._forward(x)

    def _forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, drop_prob=None):
        super(Bottleneck, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        p = np.random.uniform()
        if self.drop_prob is not None and self.training and p < self.drop_prob.item():
            return x
        else:
            return self._forward(x)

    def _forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

ConvBNReLU = lambda in_planes, planes, kernel_size, stride, padding: nn.Sequential(OrderedDict(
    [("conv", nn.Conv2d(in_planes, planes, kernel_size, stride, padding, bias=False)),
     ("bn", nn.BatchNorm2d(planes)),
     ("relu", nn.ReLU(inplace=True))]))

ConvBNReLUMaxPool = lambda in_planes, planes, kernel_size, stride, padding: nn.Sequential((OrderedDict(
    [MaskedConvBNReLU(in_planes, planes, 7,2,3,plan=1),
     nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)]
)
))

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, stem_block=ConvBNReLU, num_classes=10, multi=False, dataset = "cifar", block_kwargs={}):
        super(ResNet, self).__init__()
        self.drop_prob = torch.Tensor([0.])
        self.in_planes = 64
        self.block_kwargs = block_kwargs
        self.dataset = dataset
        self.aux_feature_shape = [256, 8, 8]

        if self.dataset == "cifar":
            self.stem = stem_block(3, 64, 3, 1, 1)
        elif self.dataset == "imagenet":
            self.stem = stem_block(3, 64, 7, 2, 3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # if self.multi: # multi head
        #     self.multi_head = MultiHeadCifar10(256)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride, **self.block_kwargs))
        self.in_planes = planes * block.expansion
        for i in range(num_blocks-1):
            layers.append(block(self.in_planes, planes, 1, drop_prob=self.drop_prob,
                                **self.block_kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_multi=False):
        out = self.stem(x)
        if self.dataset == "imagenet":
            out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        aux_feature = out = self.layer3(out)
        # if self.multi:
        #     self.logits_multi = self.multi_head(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.adapool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        if return_multi:
            return out, aux_feature
        else:
            return out 

@register_model("resnet18_masked_plan1")
def MaskedResNet18Plan1():
    return ResNet(MaskedResBlock, [2, 2, 2, 2],
                  block_kwargs={"plan": 1},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1))

@register_model("resnet18_masked_plan1_imgnet")
def MaskedResNet18Plan1():
    return ResNet(MaskedResBlock, [2, 2, 2, 2],
                  dataset="imagenet",
                  num_classes=1000,
                  block_kwargs={"plan": 1, "base_importance_strategy": "bn"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1))


@register_model("resnet18_masked_plan1_bn")
def MaskedResNet18Plan1BN():
    return ResNet(MaskedResBlock, [2, 2, 2, 2],
                  block_kwargs={"plan": 1, "base_importance_strategy": "bn"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"))

@register_model("resnet18_masked_plan1_bn_nodetach")
def MaskedResNet18Plan1BN_nodetach():
    return ResNet(MaskedResBlock, [2, 2, 2, 2],
                  block_kwargs={"plan": 1, "base_importance_strategy": "bn", "detach_base_importance": False},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn", detach_base_importance=False))


@register_model("resnet18_masked_plan1_bn_st")
def MaskedResNet18Plan1BN_st():
    return ResNet(MaskedResBlock, [2, 2, 2, 2],
                  block_kwargs={"plan": 1, "base_importance_strategy": "bn", "straight_through_grad": True},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn",
                                   straight_through_grad=True))

@register_model("resnet18_masked")
def MaskedResNet18():
    return ResNet(MaskedResBlock, [2, 2, 2, 2],
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=2))

@register_model("resnet18")
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

@register_model("resnet18_multi")
def ResNet18_multi():
    return ResNet(BasicBlock, [2,2,2,2], multi=True)

@register_model("resnet34")
def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

@register_model("resnet50")
def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])


@register_model("resnet50_masked_plan1_imgnet")
def MaskedResNet50Plan1():
    return ResNet(MaskedBottleneckBlock, [3, 4, 6, 3],dataset="imagenet", num_classes = 1000,
                  block_kwargs={"plan": 1},\
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1))

@register_model("resnet50_masked_plan1_bn_imgnet")
def MaskedResNet50Plan1BN():
    return ResNet(MaskedBottleneckBlock, [3, 4, 6, 3], dataset="imagenet", num_classes=1000,
                  block_kwargs={"plan": 1, "base_importance_strategy": "bn"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"))

@register_model("resnet50_masked_plan1_bn_cifar")
def MaskedResNet50Plan1BN():
    return ResNet(MaskedBottleneckBlock, [3, 4, 6, 3], dataset="cifar", num_classes=10,
                  block_kwargs={"plan": 1, "base_importance_strategy": "bn"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"))



@register_model("resnet101")
def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

@register_model("resnet152")
def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

