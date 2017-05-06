#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : sshuair
# @Time    : 2017/5/1

# *****************************************************
# predict the train file and find the best f2 score threhold.
# *****************************************************

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models

from pytorch.utils.network import AnokasNet
from pytorch.utils.dataloader import ImageFloderLstMulti

import argparse


# batch_size=100
targets = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 
               'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']


def phare_args():
    pharer = argparse.ArgumentParser(
        description='phare argument of prediction')
    pharer.add_argument('--gpu', default=-1, type=int, help='weather ues gpu')
    # pharer.add_argument('--cuda-id', default=0, type=bool, help='which cuda device to use')
    pharer.add_argument('--train', default=True, 
                        help='train or test , default is train')
    pharer.add_argument('--batch', dest='batch_size',
                        default=100, type=int, help='mini batch seize')
    pharer.add_argument('--root', help='root dir of the image')
    pharer.add_argument('--lstpath', help='path of image list')
    pharer.add_argument('--params', help='the pretrain params path')
    pharer.add_argument('--result', help='the output file')
    args = pharer.parse_args()

    if args.gpu > -1:
        args.cuda = True
        args.cuda_id = args.gpu
    else:
        args.cuda = False
    return args


if __name__ == '__main__':
    args = phare_args()
    if args.cuda:
        torch.cuda.set_device(args.cuda_id)

    testloader = DataLoader(
        dataset=ImageFloderLstMulti(
            root=args.root, lstpath=args.lstpath, filetype='jpg',
            transform=transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # net = PlaentNet(17)
    net = models.resnet18(num_classes=17)
    if args.cuda:
        net.cuda()
        net.load_state_dict(torch.load(args.params))
        print(net.parameters())
    else:
        net.load_state_dict(torch.load(args.params, map_location=lambda storage, loc: storage))
        net.cpu()
    optimizer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.9)

    # preds = 0
    for idx, (data, target) in enumerate(testloader):
        print(idx)
        if args.cuda:
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)
        y_pred = net(data)
        if idx == 0:
            # if args.cuda:
            preds = y_pred.data.cpu().numpy()
        else:
            preds = np.append(preds, y_pred.data.cpu().numpy(), axis=0)
    df = pd.DataFrame(preds, columns=targets)
    df.to_csv(args.result)

# train 输出结果用于寻找最佳f2 score 
# python predict.py --gpu 3 --train True --root ./data/train-jpg --lstpath ./data/train.lst --params resnet18_9.pkl --result loss_resnet18_train.csv

# train 输出结果用于寻找最佳f2 score 
# python predict.py --gpu 3 --train False --root ./data/test-jpg --lstpath ./data/test.lst --params resnet18_9.pkl --result loss_resnet18_test.csv