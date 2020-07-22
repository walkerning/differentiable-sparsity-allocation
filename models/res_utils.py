import torch
import torch.nn as nn

class DownsampleA(nn.Module):  

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)

class DownsampleC(nn.Module):     

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(nOut)

    def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      return x

class MultiHeadCifar10(nn.Module):
    def __init__(self, C, spatial_size, num_classes=10):
        super(MultiHeadCifar10, self).__init__()
        assert spatial_size in {8, 16}

        self.features = nn.Sequential(
            nn.AvgPool2d(5 if spatial_size == 8 else 7, stride=3, padding=0),
            nn.Conv2d(C, 128, kernel_size=1, stride=1 if spatial_size == 8 else 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
