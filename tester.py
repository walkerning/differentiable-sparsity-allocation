import os
import logging
from collections import defaultdict, OrderedDict

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import graph_utils
from utils import progress_bar, get_list_str, accuracy


class PrimalComponent(object):
    def __init__(self, comp_names, comp_modules, primal_strategy="first"):
        self.comp_modules = comp_modules
        self.comp_names = comp_names
        self.primal_strategy = primal_strategy
        self.primal_mod = None
        self.primal_name = None
        self.primal_idx = None

    def decide_primal(self):
        idx, primal_mod = graph_utils._select_mask_primal_module(
               self.comp_modules, strategy=self.primal_strategy)
        self.primal_name = self.comp_names[idx]
        logging.info("<strategy {}> Primal module `{}` for {}".format(
            self.primal_strategy, self.primal_name, self.comp_names))
        self.primal_mod = primal_mod
        self.primal_idx = idx
        return self.primal_idx, self.primal_name, self.primal_mod

    def decide_mask(self):
        assert self.primal_mod is not None, "decide_primal must be called before decide mask"
        mask, prob, prob_b4_clamp = self.primal_mod.get_mask_and_prob()
        [mod.set_mask(mask) for mod in self.comp_modules]
        return mask, prob, prob_b4_clamp

    def set_keep_ratio(self, alpha=None):
        if alpha is not None:
            self.primal_mod.keep_ratio.data[:] = alpha
        for mod in self.comp_modules:
            mod.keep_ratio = self.primal_mod.keep_ratio
        return self.primal_mod.keep_ratio

    def get_sigmoid_keep_ratio(self):
        assert self.primal_mod is not None, "decide_primal must be called before get_sigmoid_keep_ratio"
        return self.primal_mod.invsigmoid_keep_ratio

    def get_keep_ratio(self):
        assert self.primal_mod is not None, "decide_primal must be called before get_keep_ratio"
        return self.primal_mod.keep_ratio

    def get_primal_attr(self, name):
        return getattr(self.primal_mod, name)

    def __repr__(self):
        return "PrimalComponents({}, primal_name={} ({}), alpha={})".format(
            self.comp_names, self.primal_name, self.primal_idx,
            None if self.primal_mod is None else float(self.primal_mod.keep_ratio.cpu().data)
        )


class PrimalComponents(object):
    def __init__(self, module_components, mod_comp_names, conv_connection_dct, primal_strategy="first"):
        self.conv_connection_dct = conv_connection_dct
        self.pc_list = []
        self.conv_pidx_dct = {}
        self.conv_maskmodule_dct = {}
        for comp, comp_names in zip(module_components, mod_comp_names):
            is_conv_masked = [mod is not None for mod in comp]
            assert all(is_conv_masked) or not any(is_conv_masked)
            if not any(is_conv_masked):
                logging.info("These modules would not be pruned, ignore in PrimalComponents construction: {}".format(comp_names))
            else:
                pc = PrimalComponent(comp_names, comp, primal_strategy)
                pidx = len(self.pc_list) # current primal idx
                for conv_name in comp_names:
                    self.conv_pidx_dct[conv_name] = pidx
                # TODO: maybe weakref
                for conv_name, conv_mod in zip(comp_names, comp):
                    self.conv_maskmodule_dct[conv_name] = conv_mod
                self.pc_list.append(pc)

        self.num_pc = len(self.pc_list)
        logging.info("primal components list: %s ", str(self.pc_list))
        self.conv_input_primal_dct = defaultdict(list)
        for conv, input_conns in self.conv_connection_dct.items():
            for input_conn in input_conns:
                if not isinstance(input_conn, str):
                    # fixed size tensor
                    self.conv_input_primal_dct[conv].append([input_conn[-3], None])
                else:
                    # previous conv
                    self.conv_input_primal_dct[conv].append(
                        [self.conv_maskmodule_dct[input_conn].out_channels,
                         self.conv_pidx_dct.get(input_conn, None)])

        self.num_pc_channels = [pc.comp_modules[0].out_channels for pc in self.pc_list]
        # compute the matrix A, B
        if self.num_pc:
            A = np.zeros((self.num_pc, self.num_pc), dtype=np.float32)
            B = np.zeros(self.num_pc, dtype=np.float32)
            for i_pc, pc in enumerate(self.pc_list):
                for mod_name, module in zip(pc.comp_names, pc.comp_modules):
                    o_spatial_size = module.o_size[2] * module.o_size[3]
                    if getattr(module, "conv", module).groups > 1:
                        assert getattr(module, "conv", module).groups == module.out_channels
                        # depthwise
                        B[i_pc] = B[i_pc] + 2 * module.kernel_size[0] * module.kernel_size[1] * \
                                  o_spatial_size * module.out_channels
                    else:
                        # normal conv
                        assert sum([item[0] for item in self.conv_input_primal_dct[mod_name]])\
                            == module.in_channels
                        for in_channels, primal_idx in self.conv_input_primal_dct[mod_name]:
                            coeff = 2 * in_channels * module.out_channels * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    o_spatial_size
                            if primal_idx is not None:
                                A[i_pc, primal_idx] += coeff
                            else:
                                # this part input_channel would not be masked
                                B[i_pc] += coeff

            self.A = A
            self.B = B

    def get_flops(self, alphas):
        if isinstance(alphas, torch.Tensor):
            A = torch.tensor(self.A).to(alphas.device)
            B = torch.tensor(self.B).to(alphas.device)
        else:
            A = self.A
            B = self.B
        return (alphas * alphas[:, None] * A).sum() + (alphas * B).sum()

    def decide_primal(self):
        return [pc.decide_primal() for pc in self.pc_list]

    def decide_mask(self):
        # !!! DIRTY !!!!
        masks = []
        probs = []
        probs_b4_clamp = []
        for pc in self.pc_list:
            mask, prob, prob_b4_clamp = pc.decide_mask()
            masks.append(mask)
            probs.append(prob)
            probs_b4_clamp.append(prob_b4_clamp)

        return masks, probs, probs_b4_clamp

    def set_keep_ratio(self, alphas=None):
        if alphas is not None:
            assert len(alphas) == self.num_pc
            return [pc.set_keep_ratio(a) for a, pc in zip(alphas, self.pc_list)]
        return [pc.set_keep_ratio() for pc in self.pc_list]

    def get_keep_ratio(self):
        return [pc.get_keep_ratio() for pc in self.pc_list]

    def get_sigmoid_keep_ratio(self):
        return [pc.get_sigmoid_keep_ratio() for pc in self.pc_list]

    def get_primal_attr(self, name):
        return [pc.get_primal_attr(name) for pc in self.pc_list]

    @classmethod
    def create_from_model(cls, net, dataset = "cifar", is_masked=True, no_grouping=False, **cfg):
        mod_comp_names, conv_conn_dct = graph_utils.parse_model_components(net,dataset=dataset)
        if no_grouping:
            mod_comp_names = [[item] for item in sum(mod_comp_names, [])]
        addi_kwargs = {} if is_masked else {"type_": nn.Conv2d}
        module_components = graph_utils.get_mask_modules(
            mod_comp_names, model=net, **addi_kwargs)
        return cls(module_components, mod_comp_names, conv_conn_dct, **cfg)


class Tester(object):
    NAME = "tester"
    default_cfg = {
        "dataset":"cifar",
        "load_mask_only": False,

        # primal component selection
        "primal_comp_cfg":{"primal_strategy": "first"},
    }

    def __init__(self, net, p_net, trainloader, testloader, log, cfg):
        self.net = net
        self.p_net = p_net
        self.trainloader = trainloader[0]
        self.validloader = trainloader[1]
        self.ori_trainloader = trainloader[2]
        self.testloader = testloader
        self.log = log

        self.cfg = copy.deepcopy(self.default_cfg)
        self.cfg.update(cfg)

        self.log("Configuration:\n" + "\n".join(["\t{:10}: {:10}".format(n, str(v)) for n, v in self.cfg.items()]) + "\n")

        self.best_acc = 0.
        self.epoch = 1
        self.start_epoch = 1

        self.comp_primals = PrimalComponents.create_from_model(
            self.net, dataset=self.cfg["dataset"], **self.cfg["primal_comp_cfg"])
        self.comp_primals.decide_primal() # Decide Prima when PCs are set
        self.ori_flops = self.comp_primals.get_flops(np.ones(self.comp_primals.num_pc))

    def init(self, device, local_rank=-1,resume=None, pretrain=False):
        self.device = device
        self.local_rank = local_rank
        self.criterion = nn.CrossEntropyLoss()
        # self.lr_schedule = self.cfg["lr_schedule"]
        # default_optimizer_params = [param for name, param in self.net.named_parameters()
        #                             if "beta" not in name and "keep_ratio" not in name]
        # self.optimizer = getattr(torch.optim, self.cfg.get("optimizer_type", "SGD"))(
        #     [p for p in default_optimizer_params if p.requires_grad], **self.cfg["optimizer"])
        # finetune_optimizer_params = [param for name, param in self.net.named_parameters()
        #                             if "beta" not in name and "keep_ratio" not in name]
        # self.finetune_optimizer = getattr(
        #     torch.optim,
        #     self.cfg.get("finetune_optimizer_type", "SGD"))(
        #         finetune_optimizer_params, **self.cfg["finetune_optimizer"])
        # self.save_dict = {}

        if resume:
            # Load checkpoint.
            self.log("==> Resuming from checkpoint..")
            # print("==> Resuming from checkpoint..")
            assert os.path.exists(resume), "Error: no checkpoint directory found!"
            ckpt_path = os.path.join(resume, "ckpt.t7") if os.path.isdir(resume) else resume
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if self.cfg["load_mask_only"]: # only keep mask and keep ratio tensors
                ckpt_net = OrderedDict([item for item in checkpoint["net"].items() if "mask" in item[0] or "keep_ratio" in item[0]])
            else:
                ckpt_net = checkpoint["net"]
            self.net.load_state_dict(ckpt_net, strict=False)
            if not pretrain:
                self.best_acc = checkpoint["acc"]
                self.epoch = self.start_epoch = checkpoint["epoch"]
                if "optimizer" not in checkpoint:
                    self.log("!!! Resume mode: do not found optimizer in {}".format(ckpt_path))
                else:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.test(save=False)

    def get_expected_flops(self):
        keep_ratios = torch.cat(self.comp_primals.get_keep_ratio())
        cur_flops = self.comp_primals.get_flops(keep_ratios)

        return cur_flops, keep_ratios

    def get_true_flops(self):
        keep_ratios = []
        for pc in self.comp_primals.pc_list:
            keep_ratios.append(
                ((pc.primal_mod.mask > 0).sum().float() / pc.primal_mod.mask.nelement())\
                .detach().item())

        cur_flops = self.comp_primals.get_flops(np.array(keep_ratios))

        return cur_flops, keep_ratios

    def check_sparsity(self, expected=False):
        if expected:
            # expected flops
            cur_flops, keep_ratios = self.get_expected_flops()
            kr_str = get_list_str(keep_ratios.detach().cpu().numpy().tolist(), "{:.3f}")
            logging.info(("EXPECTED: After Epoch {}, {:.3f} % ({:2e}/{:2e}) of FLOPs (Expected) "
                          "Remains;\n\t{}").format(
                              self.epoch, 100*(cur_flops / self.ori_flops),
                              int(cur_flops), int(self.ori_flops), kr_str))
        else:
            # true flops
            cur_flops, keep_ratios = self.get_true_flops()
            logging.info(("TRUE: After Epoch {}, {:.3f} %  ({:2e}/{:2e}) of FLOPs "
                          "Remains;\n\t{}").format(
                self.epoch, 100*(cur_flops / self.ori_flops), int(cur_flops), int(self.ori_flops),
                              get_list_str(keep_ratios, "{:.3f}")))

        return keep_ratios, cur_flops / self.ori_flops

    def test(self, save=True):
        self.net.eval()
        test_loss = 0
        correct = 0
        correct_5 = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.p_net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_5 += accuracy(outputs, targets, topk=(5,))[0]
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if self.local_rank == 0 or self.local_rank == -1:
                    progress_bar(batch_idx, len(self.testloader),
                                 "Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d}) | Acc@5: {:.3f}"\
                                .format(
                                    test_loss/(batch_idx+1),
                                    100. * correct/total, correct, total, 100.*(correct_5/total)), ban="Test")
        acc = 100.*correct/total
        self.log("Test: loss: {:.3f} | acc: {:.3f} %"
                 .format(test_loss/len(self.testloader), 100.*correct/total))
        return acc


if __name__ == "__main__":
    from models import get_model
    from utils import patch_conv2d_4_size
    patch_conv2d_4_size()
    for name in ["resnet18_masked",
                 "vgg16",
                 "mobilenetv2_masked",
                 "cifar10_resnet56",
                 "cifar10_resnet56_dsconv"]:
        print(" ---- Model {} ---- ".format(name))
        model = get_model(name)()
        comp_primals = PrimalComponents.create_from_model(
            model, is_masked="masked" in name)
        ori_flops = comp_primals.get_flops(np.ones(comp_primals.num_pc))
        print("ori flops:", ori_flops)
