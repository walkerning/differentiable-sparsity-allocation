'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from .utils import register_model
from .op import MaskedConvBNReLU

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG11_masked': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13_masked': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16_masked': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19_masked': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG11_masked_bn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13_masked_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16_masked_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19_masked_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],


}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.kwargs = {}
        if "masked" in vgg_name:
            if "bn" in vgg_name:
                self.kwargs["base_importance_strategy"] = "bn"
            self.features = self._make_layers(cfg[vgg_name],masked = True)
        else:
            self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        # self.total_flops = 0
        # self._flops_calculated = False
        # self.set_hook()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # if not self._flops_calculated:
        #     self._flops_calculated = True
        return out

#     def set_hook(self):
#         for name, module in self.named_modules():
#             if "auxiliary" in name:
#                 continue
#             module.register_forward_hook(self._hook_intermediate_feature)
# 
#     def _hook_intermediate_feature(self, module, inputs, outputs):
#         if not self._flops_calculated:
#             if isinstance(module, nn.Conv2d):
#                 self.total_flops += 2* inputs[0].size(1) * outputs.size(1)*\
#                     module.kernel_size[0]*module.kernel_size[1] * \
#                     outputs.size(2)*outputs.size(3) / module.groups
#             elif isinstance(module,nn.Linear):
#                 self.total_flops += 2*inputs[0].size(1)*outputs.size(1)
#             else:
#                 pass

    def _make_layers(self, cfg, masked = False):
        layers = []
        in_channels = 3
        if (masked):
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [MaskedConvBNReLU(in_channels, x, kernel_size=3,stride=1,padding=1,affine=True,has_relu=True, **self.kwargs)]
                    in_channels = x
 
        else:
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

for name in cfg:
    def _func():
        return VGG(name)
    register_model(name.lower())(_func)
