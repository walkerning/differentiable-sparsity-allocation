import functools
import collections
import numpy as np

from PIL import Image
import torch
import torchvision.transforms as transforms

_n2t = lambda x: torch.FloatTensor(x)

class CustomizedTransform(object):
    def __init__(self, cls):
        self.cls = cls
        setattr(transforms, cls.__name__, self)
        functools.update_wrapper(self, cls)

    def __call__(self, *args, **kwargs):
        inst = self.cls(*args, **kwargs)
        return transforms.Lambda(lambda im: inst(im))

@CustomizedTransform
class Cutout(object):
    """
    Cutout randomized rectangle of size (length, length)
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        h, w = image.shape[1:]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(h)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image *= mask
        return image

@CustomizedTransform
class AddGaussianNoise(object):
    """
    Add gaussian noise to image, clip to bounds.

    Args:
      mean (sequence or float)
      sigma (sequence or positive float)
      bounds (tuple, optional): default (0,1)
    """

    def __init__(self, mean, std, bounds=(0, 1)):
        self.bounds = bounds
        self.std = std
        self.mean = mean

    def __call__(self, image):
        if isinstance(self.std, collections.Sequence):
            std = np.random.uniform(self.std[0], self.std[1], 1)[0]
        else:
            std = self.std
        if isinstance(self.mean, collections.Sequence):
            mean = np.random.uniform(self.mean[0], self.mean[1], 1)[0]
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = np.random.normal(mean, std, (row, col, ch)).astype(np.float32)
        image += torch.from_numpy(gauss)
        return image.clamp_(*self.bounds)
        # return _n2t(np.clip(np.array(image) + gauss, *self.bounds))

@CustomizedTransform
class AddSaltPepperNoise(object):
    def __init__(self, p, bounds=(0, 1)):
        self.bounds = bounds
        self.p = p
    
    def __call__(self, image):
        if isinstance(self.p, collections.Sequence):
            p = np.random.uniform(self.p[0], self.p[1], 1)[0]
        else:
            p = self.p
        u = np.expand_dims(np.random.uniform(size=image.shape[1:]), axis=0)
        b = self.bounds[1] - self.bounds[0]
        salt = (u >= 1 - p/2).astype(np.float32) * b
        pepper = - (u < p/2).astype(np.float32) * b
        image += torch.from_numpy(salt + pepper)
        return image.clamp_(*self.bounds)
        # return _n2t(np.clip(np.array(image) + salt + pepper, *self.bounds))
