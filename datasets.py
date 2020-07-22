import torch
import torchvision
import torchvision.transforms as transforms
import transforms as _transforms
import numpy as np
from utils import *

def _get_trans(trans):
    return eval("transforms." + trans)

def cifar10(train_bs=128, test_bs=100, train_transform=None, test_transform=None, root='./data', train_val_split_ratio = None, distributed=False):
    if root is None:
        root = "./data"
    train_transform = train_transform or []
    test_transform = test_transform or []
    print("train transform: ", train_transform)
    print("test transform: ", test_transform)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ] + [_get_trans(trans) for trans in train_transform] + [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ] + [_get_trans(trans) for trans in test_transform] + [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if train_val_split_ratio is not None:
 
        trainset = torchvision.datasets.CIFAR10(root=root , train=True, download=True, transform=transform_train)
        num_train = len(trainset)
        indices = range(num_train)
        split = int(np.floor(train_val_split_ratio*num_train))
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = train_bs, 
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True,num_workers=4)
        
        validloader = get_inf_iterator(torch.utils.data.DataLoader(
            trainset, batch_size = train_bs, 
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True,num_workers=4
        ))

        ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, pin_memory=True, num_workers=4)

    else:
        trainset = torchvision.datasets.CIFAR10(root= root , train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=8)
        validloader = None
        ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_bs, pin_memory=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root= root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=8)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    return trainloader,validloader,ori_trainloader,testloader, classes

def imagenet(train_bs=128, test_bs=256, train_transform=None, test_transform=None, train_val_split_ratio=None, distributed=False, path=None):
    train_transform = train_transform or []
    test_transform = test_transform or []
    print("train transform: ", train_transform)
    print("test transform: ", test_transform)
    if path is None:
        traindir = "/datasets/ILSVRC2012/ILSVRC2012_img_train"
        testdir = "/datasets/ILSVRC2012/ILSVRC2012_img_val"
    else:
        traindir = os.path.join(path, "train")
        testdir = os.path.join(path, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ]))
    testset = torchvision.datasets.ImageFolder(
        testdir,
        transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
        ]))
    
    if train_val_split_ratio is not None:
 
        num_train = len(trainset)
        indices = torch.randperm(num_train)
        split = int(np.floor(train_val_split_ratio*num_train))
        sub_trainset_0 = torch.utils.data.Subset(trainset, indices[:split])
        sub_trainset_1 = torch.utils.data.Subset(trainset, indices[split:num_train])

        if distributed:
            trainloader = torch.utils.data.DataLoader(
                sub_trainset_0, batch_size = train_bs, 
                sampler = torch.utils.data.distributed.DistributedSampler(sub_trainset_0, shuffle=True),
                pin_memory=True,num_workers=8)
            ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, 
                sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True),
                pin_memory=True, num_workers=8)
            validloader = get_inf_iterator(torch.utils.data.DataLoader(
                sub_trainset_1, batch_size = train_bs, 
                sampler = torch.utils.data.distributed.DistributedSampler(sub_trainset_1, shuffle=True),
                pin_memory=True,num_workers=8
            ))
        else: 
            trainloader = torch.utils.data.DataLoader(
                 trainset, batch_size = train_bs, 
                sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True,num_workers=8)
            ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, pin_memory=True, num_workers=8)
            validloader = get_inf_iterator(torch.utils.data.DataLoader(
                trainset, batch_size = train_bs, 
                sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                pin_memory=True,num_workers=8
            ))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=16)
        validloader = None
        ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_bs, pin_memory=True, shuffle=True,num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, 
                        shuffle=False,num_workers=8)
    classes = []

    return trainloader,validloader,ori_trainloader,testloader, classes

