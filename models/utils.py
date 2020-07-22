import torch
from torch.autograd import Variable

_registry = {}

## Register model factoreis (need to be callable)
def register_model(name):
    def registerer(func):
        _registry[name] = func
        return func
    return registerer

def get_model(name):
    return _registry[name]

def avail_models():
    return sorted(list(_registry.keys()))

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x
