#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : sshuair
# @Time    : 2017/4/29
# @Project : kaggle-planet

# from graphviz import Digraph
from torch.autograd import Variable
from operator import mul
from functools import reduce
import torch

def param_structure(net):
    """
    输出网络结构以及每一层的网络参数
    :param net: 网络
    :return: None
    """
    print(net)
    total_parmas = 0
    for idx, itm in enumerate(net.parameters()):
        layer_params = reduce(mul, itm.size(), 1)
        total_parmas += layer_params
        print('layer{idx}: {struct}, params_num:{params_num}'.format(idx=idx, struct=itm.size(), params_num=layer_params))
    print('\ntotal_parmas_num: {0}\n'.format(total_parmas))

def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot