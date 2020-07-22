#pylint:disable-all
from collections import OrderedDict
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .utils import register_model
from .res_utils import DownsampleA, DownsampleD
from .op import MaskedResBlock, MaskedConvBNReLU


ConvBNReLU = lambda in_planes, planes, kernel_size, stride, padding: nn.Sequential(OrderedDict(
    [("conv", nn.Conv2d(in_planes, planes, kernel_size, stride, padding, bias=False)),
     ("bn", nn.BatchNorm2d(planes)),
     ("relu", nn.ReLU(inplace=True))]))

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample="avgpool"):
        super(BasicBlock, self).__init__()
        assert downsample in {"conv", "avgpool", "conv2"}
        self.downsample = downsample

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if self.downsample == "avgpool":
                self.shortcut = DownsampleA(in_planes, planes, stride)
            elif self.downsample == "conv":
              self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
              )
            elif self.downsample == "conv2":
                self.shortcut = DownsampleD(in_planes, planes, stride)

    def forward(self, x):
        x = F.relu(self.conv(x) + self.shortcut(x))
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, stem_block=ConvBNReLU, num_classes=10, block_kwargs={}):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.block_kwargs = block_kwargs
        self.aux_feature_shape = [32, 16, 16]

        self.stem = stem_block(3, 16, 3, 1, 1)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.pool = nn.AvgPool2d(8)
        
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if type(m) in [nn.Conv2d, nn.Linear, nn.BatchNorm2d]:
                m.reset_parameters()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **self.block_kwargs))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_multi=False):
        x = self.stem(x)
        x = self.layer1(x)
        aux_feature = x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)
        if return_multi:
            return x, aux_feature
        else:
            return x


@register_model("cifar10_resnet20")
def ResNet20(num_classes=10):
    return ResNet(BasicBlock, [3,3,3], num_classes=num_classes)

@register_model("cifar10_resnet20_dsconv")
def ResNet20(num_classes=10):
    return ResNet(BasicBlock, [3,3,3], num_classes=num_classes, block_kwargs={
      "downsample": "conv"
    })

@register_model("cifar10_resnet20_masked_plan1_bn")
def ResNet20(num_classes=10):
    return ResNet(MaskedResBlock, [3,3,3], block_kwargs={"plan": 1,
                                                         "base_importance_strategy": "bn",
                                                         "downsample": "avgpool"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"),
                  num_classes=num_classes)

@register_model("cifar10_resnet20_dsconv_masked_plan1_bn")
def ResNet20(num_classes=10):
    return ResNet(MaskedResBlock, [3,3,3], block_kwargs={"plan": 1,
                                                         "base_importance_strategy": "bn",
                                                         "downsample": "conv"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"),
                  num_classes=num_classes)

@register_model("cifar10_resnet20_dsconv2_masked_plan1_bn")
def ResNet20(num_classes=10):
    return ResNet(MaskedResBlock, [3,3,3], block_kwargs={"plan": 1,
                                                         "base_importance_strategy": "bn",
                                                         "downsample": "conv2"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"),
                  num_classes=num_classes)

@register_model("cifar10_resnet32")
def ResNet32(num_classes=10):
    return ResNet(BasicBlock, [5,5,5], num_classes=num_classes)

@register_model("cifar10_resnet44")
def ResNet44(num_classes=10):
    return ResNet(BasicBlock, [7,7,7], num_classes=num_classes)

@register_model("cifar10_resnet56")
def ResNet56(num_classes=10):
    return ResNet(BasicBlock, [9,9,9], num_classes=num_classes)

@register_model("cifar10_resnet56_dsconv")
def ResNet56(num_classes=10):
    return ResNet(BasicBlock, [9,9,9], num_classes=num_classes, block_kwargs={
      "downsample": "conv"
    })

@register_model("cifar10_resnet56_dsconv2")
def ResNet56(num_classes=10):
    return ResNet(BasicBlock, [9,9,9], num_classes=num_classes, block_kwargs={
      "downsample": "conv2"
    })

@register_model("cifar10_resnet56_dsconv2_masked_plan1_bn")
def ResNet56(num_classes=10):
    return ResNet(MaskedResBlock, [9,9,9], block_kwargs={"plan": 1,
                                                         "base_importance_strategy": "bn",
                                                         "downsample": "conv2"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"),
                  num_classes=num_classes)

@register_model("cifar10_resnet56_dsconv_masked_plan1_bn")
def ResNet56(num_classes=10):
    return ResNet(MaskedResBlock, [9,9,9], block_kwargs={"plan": 1,
                                                         "base_importance_strategy": "bn",
                                                         "downsample": "conv"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"),
                  num_classes=num_classes)

@register_model("cifar10_resnet56_masked_plan1_bn")
def ResNet56(num_classes=10):
    return ResNet(MaskedResBlock, [9,9,9], block_kwargs={"plan": 1,
                                                         "base_importance_strategy": "bn",
                                                         "downsample": "avgpool"},
                  stem_block=lambda in_planes, planes, kernel_size, stride, padding:\
                  MaskedConvBNReLU(in_planes, planes, kernel_size, stride, padding, plan=1,
                                   base_importance_strategy="bn"),
                  num_classes=num_classes)
