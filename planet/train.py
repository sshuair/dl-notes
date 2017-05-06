#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : sshuair
# @Time    : 2017/4/29

# *****************************************************
# training file
# *****************************************************

import argparse
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import fbeta_score

from pytorch.utils.dataloader import ImageFloderLstMulti
from pytorch.utils.network import AnokasNet
# from pytorch.utils.simplenet import SimpleNet64_2
from pytorch.utils.visualize import param_structure
import torchvision.models as models


logging.basicConfig(filename='trainning.log',level=logging.INFO,filemode = 'w', format = '%(asctime)s - %(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    net.train()
    # adjust learning rate
    adjust_learning_rate(optimizer, epoch)
    for batch_idx, (data, target) in enumerate(trainloader):
        if args.cuda:
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        y_pred = net(data)
        # BCELoss need FloatTensor data type
        loss = criterion(y_pred, target.float())
        loss.backward()
        optimizer.step()

        f2 = fbeta_score(target.data.cpu().numpy(), y_pred.data.cpu().numpy() > 0.2, beta=2, average='samples')

        # Now once you have your prediction, you need to threshold.
        # 0.5 is the default naive way but it's probably not optimal. In any case, once you get there, great !
        # if batch_idx % args.batch_size == 0:
        logging.info('Train Epoch[{}]: Batch[{}/{} ({:.2%})]  Loss: {:.6f}  acc(f2): {:.6f}'.format(
            epoch, batch_idx * args.batch_size, len(trainloader.dataset),
            args.batch_size * batch_idx / len(trainloader.dataset),
            loss.data[0],  f2
        ))


def validate(epoch):
    net.eval()
    f2 = 0
    loss = 0
    for idx, (data, target) in enumerate(validateloader):
        if args.cuda:
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)

        y_pred = net(data)
        # loss已经是每个batch求和之后的结果
        loss += criterion(y_pred, target.float()).cpu().data[0]
        batch_f2 = fbeta_score(target.data.cpu().numpy(), y_pred.data.cpu().numpy() > 0.2, beta=2, average='samples')
        f2 += args.batch_size * batch_f2

    loss = loss / len(validateloader)  #
    f2 = f2 / len(validateloader.dataset)
    # print(epoch,loss,f2)
    logging.info('val Epoch[{}]: Average loss: {:.6f}, acc(f2):{:.6f}\n'.format(
        epoch, loss, f2
    ))


def phare_args():
    pharer = argparse.ArgumentParser(
        description='phare argument of train and validate parameters')
    pharer.add_argument('--gpu', default=-1, type=int, help='weather ues gpu')
    # pharer.add_argument('--cuda-id', default=0, type=bool, help='which cuda device to use')
    pharer.add_argument('--epoch', default=50, type=int,
                        help='number of total epochs to run')
    pharer.add_argument('--batch', dest='batch_size',
                        default=128, type=int, help='mini batch seize')
    pharer.add_argument('--root', help='root dir of the image')
    pharer.add_argument('--lr', default=0.01, help='learning rate')
    pharer.add_argument('--width', default=224, type=int, help='scale width')
    pharer.add_argument('--height', default=224, type=int, help='scale height')
    args = pharer.parse_args()

    if args.gpu > -1:
        args.cuda = True
        args.cuda_id = args.gpu
    else:
        args.cuda = False
    return args

def save_check_point(net, epoch):
    filename = './params/resnet_' + str(epoch) + '.pkl'
    torch.save(net.state_dict(), filename)

if __name__ == '__main__':
    global args
    args = phare_args()
    if args.cuda:
        torch.cuda.set_device(args.cuda_id)
    batch_size = args.batch_size

    height, width = args.width, args.height
    in_channels = 3

    trainloader = DataLoader(
        dataset=ImageFloderLstMulti(
            root='./data/train-jpg', lstpath='./data/train.lst', filetype='jpg',
            transform=transforms.Compose([
                transforms.Scale((height, width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True,
    )

    validateloader = DataLoader(
        dataset=ImageFloderLstMulti(
            root='./data/train-jpg', lstpath='./data/val.lst', filetype='jpg',
            transform=transforms.Compose([
                transforms.Scale((height, width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True,
    )

    net = models.vgg11_bn(num_classes=17)
    # net = SimpleNet64_2((in_channels,height,width), num_classes=17)
    param_structure(net)
    if args.cuda:
        net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(1, args.epoch):
        train(epoch)
        validate(epoch)
        save_check_point(net, epoch)
        # pass