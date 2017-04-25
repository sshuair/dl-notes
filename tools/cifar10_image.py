# coding=utf-8

"""
change the cifar-10 python format to single image
"""
from PIL import Image
import numpy as np
from skimage.io import imsave
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict





def cifar10_to_class_floder(original_fp, dest_dir):
    """
    将不同类别的数据整到各自的类别的文件夹中
    :param original_fp: 
    :param dest_dir: 
    :return: 
    """
    for fp in original_fp:
        print(fp)

        result = unpickle(fp)
        for i in range(len(result[b'labels'])):

            label = result[b'labels'][i]
            filename = result[b'filenames'][i]
            data = result[b'data'][i]

            image_path = os.path.join(dest_dir, str(label))
            if not os.path.exists(image_path):
                os.mkdir(image_path)

            img_R = data[0:1024].reshape((32, 32))
            img_G = data[1024:2048].reshape((32, 32))
            img_B = data[2048:3072].reshape((32, 32))
            img2 = np.dstack((img_R, img_G, img_B))
            imsave(os.path.join(image_path, filename.decode('utf-8')), img2)


def cifar10_to_single_floder(original_fp, dest_fplist, dest_dir):
    """
    将不同类别的数据整到各自的类别的文件夹中
    :param original_fp: 
    :param dest_dir: 
    :return: 
    """
    idx = 0
    with open(dest_fplist, 'w') as f:
        for fp in original_fp:
            print(fp)

            result = unpickle(fp)
            for i in range(len(result[b'labels'])):
                idx += 1

                label = result[b'labels'][i]
                filename = result[b'filenames'][i].decode('utf-8')
                data = result[b'data'][i]

                sample_line = str(idx) + '\t' + str(label) + '\t' + filename + '\n'
                f.writelines(sample_line)

                img_R = data[0:1024].reshape((32, 32))
                img_G = data[1024:2048].reshape((32, 32))
                img_B = data[2048:3072].reshape((32, 32))
                img2 = np.dstack((img_R, img_G, img_B))
                imsave(os.path.join(dest_dir, filename), img2)
    pass

if __name__ == '__main__':
    train_fp = ['../data/cifar/cifar-10-batches-py/data_batch_1',
                '../data/cifar/cifar-10-batches-py/data_batch_2',
                '../data/cifar/cifar-10-batches-py/data_batch_3',
                '../data/cifar/cifar-10-batches-py/data_batch_4',
                '../data/cifar/cifar-10-batches-py/data_batch_5'
                ]
    test_fp = ['../data/cifar/cifar-10-batches-py/test_batch']

#   按类别分门别类的存储图片，方便ImageFolder调用
    dest_train_class_dir = '../data/cifar/train'
    dest_test_class_dir = '../data/cifar/test'
    cifar10_to_class_floder(train_fp, dest_train_class_dir)  # train
    cifar10_to_class_floder(test_fp, dest_test_class_dir)  # test

#   生成一个文件列表，列表中标识（id, 类别, 文件路径），训练数据全部放在一个文件夹下面
    dest_train_single_dir = '../data/cifar/train_single'
    dest_test_single_dir = '../data/cifar/test_single'
    fp_train_list = '../data/cifar/train_single_list.lst'
    fp_test_list = '../data/cifar/test_single_list.lst'
    cifar10_to_single_floder(train_fp, fp_train_list, dest_train_single_dir)
    cifar10_to_single_floder(test_fp, fp_test_list, dest_test_single_dir)



