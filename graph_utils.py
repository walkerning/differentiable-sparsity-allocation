# pylint: disable-all
import copy
import random
from collections import defaultdict
import logging

import networkx as nx
import torch
from torch.autograd import Variable

from models.op import MaskedConvBNReLU


def add_nodes(var, G, param_map, seen, o_nodes, i_nodes):
    if var not in seen:
        if hasattr(var, 'variable'):
            u = var.variable
            name = param_map[id(u)]
            param_shape = tuple(u.size())
            is_input = id(u) in [id(n) for n in i_nodes]
            G.add_node(str(id(var)), type="param", param_name=name, param_shape=param_shape,
                       is_input=is_input)
        elif var in o_nodes:
            G.add_node(str(id(var)), type="output", output_name=str(type(var).__name__))
        else:
            G.add_node(str(id(var)), type="func", func_name=str(type(var).__name__)) # backward
        seen.add(var)
        if hasattr(var, 'next_functions'):
            for u in var.next_functions:
                if u[0] is not None:
                    G.add_edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0], G, param_map, seen, o_nodes, i_nodes)
        if hasattr(var, 'saved_tensors'):
            for t in var.saved_tensors:
                G.add_edge(str(id(t)), str(id(var)))
                add_nodes(t, G, param_map, seen, o_nodes, i_nodes)

def parse_model_components(model, dataset = "cifar"):
    params = dict(model.named_parameters())
    assert all(isinstance(p, Variable) for p in params.values())
    param_map = {id(v): k for k, v in params.items()}
    
    if dataset == "cifar":
        inputs = torch.randn(1, 3, 32, 32)
    elif dataset == "imagenet":
        inputs = torch.randn(1,3,224,224)
    model_device = next(model.parameters()).device
    inputs = Variable(inputs.to(model_device), requires_grad=True)
    var = model(inputs)
    param_map[id(inputs)] = "inputs"
    G = nx.OrderedDiGraph()

    seen = set()

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)
    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn, G, param_map, seen, output_nodes, [inputs])
    else:
        add_nodes(var.grad_fn, G, param_map, seen, output_nodes, [inputs])

    G_whole = copy.deepcopy(G)
    all_conv_nodes = [key for key, value in G.nodes(data=True)
                      if value["type"] == "func" and "Conv" in value["func_name"]]

    # channelwise operation is connected in the graph, so normal convolutions are split nodes
    modules = dict(model.named_modules())
    for conv_node in all_conv_nodes:
        pres = list(G.predecessors(conv_node))
        param_node = [i for i in pres if G.nodes[i]["type"] == "param" and not G.nodes[i]["is_input"]][0]
        param_name = G.nodes[param_node]["param_name"]
        pres.remove(param_node)
        # remove the input node, or it would be treated as a single component
        optional_input_nodes = [i for i in pres if G.nodes[i]["type"] == "param" and G.nodes[i]["is_input"]]
        if optional_input_nodes:
            pres.remove(optional_input_nodes[0])
        conv_mod_name = param_name.rsplit(".", 1)[0]
        conv_mod = modules[conv_mod_name]
        G.nodes[conv_node]["module_name"] = conv_mod_name
        if conv_mod.groups != 1:
            # can only support depthwise
            assert conv_mod.groups == conv_mod.in_channels, \
                "could only support depthwise group convs"
            pass
        else:
            # normal conv: remove the other predecessors
            [G.remove_edge(pre, conv_node) for pre in pres]

    components = nx.connected_components(G.to_undirected())
    all_module_partitions = []
    # topo sort inside each components/partition
    topo_sorted_nodes = list(nx.topological_sort(G_whole))
    components = [list(sorted(comp, key=lambda node: topo_sorted_nodes.index(node)))
                  for comp in components]

    first_conv_in_comp = []
    for comp in components:
        conv_modules = []
        conv_nodes = []
        for node in comp:
            if "module_name" in G.nodes[node]:
                # only conv modules have `module_name` attr set before
                conv_modules.append(G.nodes[node]["module_name"])
                conv_nodes.append(node)
        all_module_partitions.append(conv_modules)
        first_conv_in_comp.append(conv_nodes[0])
    # topo sort inside the whole graph
    all_module_partitions = list(zip(*list(sorted(list(zip(all_module_partitions, first_conv_in_comp)),
                                                  key=lambda item: topo_sorted_nodes.index(item[1])))))[0]

    # parse conv connection
    # search predecessors DFS from each conv until all path encountered conv
    conv_connection_dct = defaultdict(list)
    for conv_node in all_conv_nodes:
        c_node_name = G.nodes[conv_node]["module_name"]
        pre_node = list(G_whole.predecessors(conv_node))[0]
        stack = [pre_node]
        while stack:
            visiting_node = stack.pop()
            # do not follow param type nodes, except the checking for the inputs
            node_data = G.nodes[visiting_node]
            if node_data["type"] == "func":
                if "Conv" in node_data["func_name"]:
                    conv_connection_dct[c_node_name].append(node_data["module_name"])
                elif "Concat" in node_data["func_name"]:
                    # need to follow two path
                    stack = stack + list(reversed(list(G_whole.predecessors(visiting_node))))
                elif "Add" in node_data["func_name"]:
                    # only need to follow one path to find the neareast conv (better BFS for efficiency)
                    # add-concat is a complex scenario, and we dont handle it here
                    stack.append(list(G_whole.predecessors(visiting_node))[0])
                else:
                    stack.append(list(G_whole.predecessors(visiting_node))[0])
            elif node_data["type"] == "param" and node_data["is_input"]:
                conv_connection_dct[c_node_name].append(node_data["param_shape"])

    return all_module_partitions, conv_connection_dct

def _get_maskconv_module(named_modules, conv_name, type_=MaskedConvBNReLU):
    while 1:
        mod = named_modules[conv_name]
        if isinstance(mod, type_):
            break
        if "." not in conv_name:
            return None
        conv_name = conv_name.rsplit(".", 1)[0]
    return mod

def get_mask_modules(names, model=None, named_modules=None, type_=MaskedConvBNReLU):
    if named_modules is None:
        named_modules = dict(model.named_modules())
    if isinstance(names, str):
        return _get_maskconv_module(named_modules, names, type_=type_)
    return [get_mask_modules(mod_name, named_modules=named_modules, type_=type_) for mod_name in names]

def select_mask_primal_module(comp_modules, strategy="first"):
    is_conv_masked = [mod is not None for mod in comp_modules]
    assert all(is_conv_masked) or not any(is_conv_masked)
    if not any(is_conv_masked):
        return None, None
    else:
        idx, primal_mod = _select_mask_primal_module(comp_modules, strategy=strategy)
        return idx, primal_mod

def _select_mask_primal_module(comp_modules, strategy):
    # TODO: could inference for the primary module
    if strategy == "first":
        return 0, comp_modules[0]
    if strategy == "random":
        idx = random.choice(range(len(comp_modules)))
        return idx, comp_modules[idx]
    if strategy == "depthwise":
        for idx, comp_module in enumerate(comp_modules):
            if comp_module.conv.groups > 1:
                return idx, comp_module
        else:
            return 0, comp_modules[0]
    if strategy == "no1x1":
        for idx, comp_module in enumerate(comp_modules):
            if comp_module.conv.kernel_size[0] != 1:
                return idx, comp_module
        else:
            return 0, comp_modules[0]
    if strategy == "stride":
        for idx, comp_module in enumerate(comp_modules):
            if comp_module.conv.stride != 1:
                return idx, comp_module
        else:
            return 0, comp_modules[0]


if __name__ == "__main__":
    from pprint import pprint
    from models import get_model
    for name in ["vgg16", "resnet18", "resnet18_masked", "mobilenetv2", "mobilenetv2_masked"]:
        print(" ---- Model {} ---- ".format(name))
        model = get_model(name)()
        model.to("cuda:0")
        module_components, conv_conn_dct = parse_model_components(model)
        modules = get_mask_modules(module_components, model=model)
        pprint(module_components)
        if name == "mobilenetv2_masked":
            # test select mask primal module
            for comp, comp_names in zip(modules, module_components):
                idx, primal_mod = select_mask_primal_module(comp, strategy="depthwise")
                if idx is not None:
                    print("<{}> Select primal module `{}` for components `{}`".\
                          format("depthwise", comp_names[idx], comp_names))
        elif "masked" in name:
            # test select mask primal module
            for comp, comp_names in zip(modules, module_components):
                idx, primal_mod = select_mask_primal_module(comp, strategy="stride")
                if idx is not None:
                    print("<{}> Select primal module `{}` for components `{}`"\
                          .format("stride", comp_names[idx], comp_names))

        print("conv conn dict:")
        pprint(conv_conn_dct)
        print(" ---- End model {} ---- ".format(name))
