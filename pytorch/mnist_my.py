# coding=utf-8
"""
CNN一般步骤
1. Load and normalizing the training and test datasets using torchvision
2. Define a Convolution Neural Network
    - Define the neural network that has some learnable parameters (or weights)
    - Propagate gradients back into the network’s parameters
    - Update the weights of the network, typically using a simple update rule: weight = weight + learning_rate * gradient
3. Define a loss function (how far is the output from being correct)
4. Train the network on the training data
    - Iterate over a dataset of inputs (batch)
5. Test the network on the test data

# 注意：
1. pytorch 的训练都是以batch的方式进行的，也就是说只支持批量输入：[batch_size, image_channel, image_width, image_height]。
2. 在设计网络结构时，尤其是在卷积层转到全连接层时候，全连接层的第一层（input_channel, output_channel）一定等于卷积池化层最后一层中的
   image_channel * image_width * image_height；
   flat features的时候需要用到nn.view(-1, image_channel * image_width * image_height)。
3. 在定义网络结构时，__init__中定义的结构与forward中的F.func能够起到相同的作用；
   如果不在__init__中定义pool1，那么可以在forward中用x=f.max_pool2d(self.conv1(x),2)表示，参考：Zen_君 & examples/mnist/main.py
4. flat features时，不能用这种方法 x = x.view(batch_size, -1)，因为最后一个epoch可能不是正好整除batch_size
5. 所有的pytorch数据都要组织成Variable
"""

from functools import reduce
from operator import mul
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

batch_size=64

trainloader = DataLoader(
    dataset=MNIST(root='../data', train=True,
                  transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]
                                               )),
    batch_size=batch_size, shuffle=True
)

testloader = DataLoader(
    dataset=MNIST(root='../data', train=False,
                  transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]
                                               )),
    batch_size=batch_size, shuffle=True
)


class Net2(nn.Module):
    """
    定义网络结构
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.pool2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 100) #x.view的结果必须和这个相同，也就是特征图层的层数和每张特征图的大小乘积
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        """
        #用F.func与定义在__init__里面的效果相同，比如说
        如果不在__init__中定义pool1，那么可以在forward中用x=f.max_pool2d(self.conv1(x),2)表示
        参考：Zen_君 & examples/mnist/main.py
        :param x: 输入的变量，mini-batch
        :return: 
        """
        # 第一层：卷积、激活（relu）、池化
        x = self.conv1(x)
        x = F.relu(x)  #用F.func与定义在__init__里面的效果相同，比如说
        x = self.pool1(x)

        # 第二层：卷积、激活（relu）、池化
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # dropout
        # x = self.pool2_drop(x)

        # 全连接层
        x = x.view(-1, self.num_flat_features(x))  # the size -1 is inferred from other dimensions，由第二个参与自动计算第一位的参数，避免手动计算
        # 第一层全连接层
        x = F.relu(self.fc1(x))
        # 第二层全连接层
        x = F.relu(self.fc2(x))
        # 第三层 输出层
        x = self.fc3(x)

        return F.log_softmax(x)

    #使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:] # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16*4*4, 100) #TODO: 寻找自动匹配方法
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        """
        
        :param x: 一个batch的数据，输入结构：[batch_size, image_channel, image_width, image_height]
        :return: 前向传播结果
        """
        # 第一阶段特征提取（卷积、relu、池化）
        x = self.pool1(F.relu(self.conv1(x)))

        # 第二阶段特征提取（卷积、relu、池化）
        x = self.pool2(F.relu(self.conv2(x)))

        # dropout
        x = self.drop(x)

        # flat features 多维维空间一维化，输出也就是[batch_size, image_channel * image_width * image_height]
        x = x.view(-1, self.num_flat_features(x))
        # x = x.view(batch_size, -1)  不能用这种方法，因为最后一个epoch可能不是正好整除batch_size

        # 全连接层（fc1，fc2，fc3）
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.log_softmax(x)

    def num_flat_features(self, x):
        feature_size = x.size()[1:]
        num_flat = reduce(lambda m, n: m*n, feature_size, 1)
        return num_flat

def param_structure(net):
    """
    输出网络结构以及每一层的网络参数
    :param net: 网络
    :return: 
    """
    print(net)
    total_parmas = 0
    for idx, itm in enumerate(net.parameters()):
        layer_params = reduce(mul, itm.size(), 1)
        total_parmas += layer_params
        print('layer{idx}: {struct}, params_num:{params_num}'.format(idx=idx, struct=itm.size(), params_num=layer_params))
    print('\ntotal_parmas_num: {0}\n'.format(total_parmas))

net = Net()

param_structure(net) #输出网络结构

# 参数优化
optimizer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.9)

def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data), Variable(target) #将Tensor转化为Variable

        optimizer.zero_grad()
        outputs = net(data)  #输出结果outputs(batch_size, class_num)
        loss = F.nll_loss(outputs, target)  #TODO:???
        loss.backward()
        optimizer.step()

        if batch_idx % batch_size == 0: # print every batch_size(64) mini-batches
            print('Train Epoch[{}]: Batch[{}/{} ({:.2%}%)]  Loss: {:.6}'.format(
                epoch, batch_idx*batch_size,
                len(trainloader.dataset),
                batch_size*batch_idx/len(trainloader.dataset),
                loss.data[0]
            ))

def test(epoch):
    """
    整体思路是：feed forward, 计算出预测的结果，然后统计预测的结果中正确的所占比例即为精度，
    在预测中不需要backward
    :param epoch: 
    :return: 
    """
    net.eval()
    correct = 0
    test_loss = 0
    for batch_idx, (data, target) in enumerate(testloader):
        data, target = Variable(data), Variable(target)

        outputs = net(data)
        test_loss += F.nll_loss(outputs, target).data[0]
        y_pred = outputs.data.max(1)[1]
        correct += y_pred.eq(target.data).cpu().sum()

    avg_loss = test_loss/len(testloader) #loss function already averages over batch size
    print('\nTest  Epoch {}: Average loss: {:.6}, Accuracy:{:.6%}\n'.format(
        epoch, avg_loss, correct/len(testloader.dataset)
    ))



for epoch in range(1,5):
    train(epoch)
    test(epoch)