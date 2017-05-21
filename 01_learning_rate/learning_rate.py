# coding=utf-8

# 测试不同learning rate对模型过拟合、训练时间的影响
import argparse
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets, models
import model


def logger(filename='trainning'):
    logging.basicConfig(filename=filename, level=logging.INFO,
                        filemode='w', format='%(asctime)s - %(levelname)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def phare_args():
    pharer = argparse.ArgumentParser()
    pharer.add_argument('--gpu', default=-1, type=int,
                        help='weather use gpu device')
    pharer.add_argument('--lr', required=False,
                        type=float, help='learning rate')
    pharer.add_argument('--log', required=False,
                        default='log', help='log file path')
    pharer.add_argument('--batch_size', default=200, type=int, help='batch_size')
    args = pharer.parse_args()
    args.cuda = True if args.gpu > -1 else False

    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epoch, args):
    net.train()
    avg_loss = 0
    avg_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # calculate the evaluation. accuracy, precision, recall, f1, mape...
        # in this case, we use accuracy 
        pred = output.data.max(1)[1]
        correct = pred.cpu().eq(target.data.cpu()).sum()
        acc = float(correct) / len(data)

        logging.info('Train epoch [{}]: Batch[{}/{}] ({:.2%}) loss: {:.6f} acc: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),  batch_idx / len(train_loader),
            loss.data[0], acc
        ))

        avg_loss += loss.data[0]
        avg_acc += acc

    logging.info('Train epoch [{}] train avg loss: {:.6f} train avg acc: {:.6f}'.format(
        epoch, avg_loss / len(train_loader), avg_acc / len(train_loader)
    ))


def val(epoch, args):
    net.eval()
    avg_loss = 0
    avg_acc = 0
    for batch_size, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = Variable(data).cuda(), Variable(target).cuda()
        else:
            data, target = Variable(data), Variable(target)
        output = net(data)
        loss = criterion(output, target)

        pred = output.data.max(1)[1]
        correct = pred.cpu().eq(target.data.cpu()).sum()
        acc = float(correct) / len(data)

        avg_loss += loss.data[0]
        avg_acc += acc
    logging.info('Val epoch [{}] val avg loss: {:.6f} val avg acc: {:.6f}\n'.format(
        epoch, avg_loss / len(val_loader), avg_acc / len(val_loader)
    ))


if __name__ == '__main__':
    args = phare_args()
    logger(args.log)
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    train_loader = DataLoader(
        datasets.CIFAR10('../data/cifar', train=True,
                         transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        datasets.CIFAR10('../data/cifar', train=False,
                         transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size,
        shuffle=True
    )

    # net = Net()
    net = model.cifar10(n_channel=128)
    if args.cuda:
        net.cuda()
    optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 1000):
        train(epoch, args)
        val(epoch, args)
