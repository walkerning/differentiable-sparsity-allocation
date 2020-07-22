"""test script"""
from __future__ import print_function

import os
import sys
import random
import shutil
import argparse
import logging

import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import datasets
from models import get_model, avail_models
from tester import Tester
from utils import patch_conv2d_4_size,accuracy


# def sensitivity_analysis(trainer, save_dir = None):
#     sensitivity_dict = {}
#     cur_keep_ratios = trainer.comp_primals.get_keep_ratio()
#     for idx, pc in enumerate(trainer.comp_primals.pc_list):
#         for i in np.arange(0.,0.1,0.01):
#             # Decide Mask
#             cur_channels = pc.primal_mod.bn.weight.size(0)
#             num2mask = int(cur_channels*(1-cur_keep_ratios[idx].data.cpu().numpy()+i))
#             val, ind = pc.primal_mod.bn.weight.abs().topk(num2mask, largest=False)
#             # Apply Mask
#             for comp in pc.comp_modules:
#                 comp.mask.fill_(1)
#                 comp.mask[ind] = 0
#             test_acc = trainer.test(save=False)
#             print(test_acc, num2mask/cur_channels, pc.comp_names)
#             if not pc.primal_name in sensitivity_dict:
#                 sensitivity_dict[pc.primal_name] = []
#             sensitivity_dict[pc.primal_name].append([test_acc, num2mask/cur_channels])
#         # Restore full mask this pc
#         for comp in pc.comp_modules:
#             comp.mask.fill_(1) 
#     plt.figure()
#     for name in sensitivity_dict:
#         if "skip" in name:
#             acc = []
#             for i in sensitivity_dict[name]:
#                 acc.append(i[0])
#             plt.plot(acc, label=name, linestyle=":")
#             plt.legend()
#         else:
#             acc = []
#             for i in sensitivity_dict[name]:
#                 acc.append(i[0])
#             plt.plot(acc, label=name)
#             plt.legend()
#     if save_dir is not None:
#         plt.savefig(os.path.join(save_dir, "sens.jpg"))

#     for i in sensitivity_dict.keys():
#         print(sensitivity_dict[i][0], sensitivity_dict[i][-1], sensitivity_dict[i][0][0] - sensitivity_dict[i][-1][0])
#     return sensitivity_dict


def main(argv):
    patch_conv2d_4_size()

    ## Parsing arguments
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--model", required=True, help="model name")
    parser.add_argument("--gpu", default="0", help="gpu ids, seperate by comma")
    parser.add_argument("--resume", "-r", help="resume from checkpoint,specify folder containing the ckpt.t7")
    parser.add_argument("--dataset",default="cifar",type=str,help="The Dataset")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="do not use gpu")
    parser.add_argument("--seed", default=None, help="random seed", type=int)
    parser.add_argument("--path", default=None, help="imagenet dataset path")
    args = parser.parse_args(argv)

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    if device == "cuda":
        logging.info("Using GPU! Available gpu count: {}".format(torch.cuda.device_count()))
    else:
        logging.info("\033[1;3mWARNING: Using CPU!\033[0m")
    
    ## Dataset
    if args.dataset == "cifar":
        trainloader,validloader, ori_trainloader, testloader, _ = datasets.cifar10(
            train_bs=128, test_bs=100, train_transform=None, test_transform=None, train_val_split_ratio=0.9)
    elif args.dataset == "imagenet":
        trainloader,validloader, ori_trainloader, testloader, _ = datasets.imagenet(
            128, 32, None, None, train_val_split_ratio=None, path=args.path)

    ## Build model
    logging.info("==> Building model..")
    gpus = [int(d) for d in args.gpu.split(",")]
    torch.cuda.set_device(gpus[0])
    net = get_model(args.model)()
    net = net.to(device)
    
    if device == "cuda":
        cudnn.benchmark = True
        if len(gpus) > 1:
            p_net = torch.nn.DataParallel(net, gpus)
        else:
            p_net = net
    
    tester = Tester(net, p_net, [trainloader, validloader, ori_trainloader], testloader,
                     cfg={"dataset": args.dataset}, log=print)
    tester.init(device=device, resume=args.resume, pretrain=True)
    # tester.test(save=False)
    keep_ratios, sparsity = tester.check_sparsity()
    print("The final Sparsity is {:.3}, Keep Ratios Are:\n{}".format(sparsity, keep_ratios))
    for pc in tester.comp_primals.pc_list:
        print(pc.comp_names, pc.get_keep_ratio())
    _, keep_ratios = tester.get_true_flops()
    # sens_dict = sensitivity_analysis(tester, save_dir=save_dir)
    # torch.save(sens_dict,os.path.join(save_dir, "sens.t7"))
    # mask, prob, prob_b4_clamp = tester.comp_primals.pc_list[0].comp_modules[0].get_mask_and_prob()

if __name__ == "__main__":
    main(sys.argv[1:])


