# coding=utf-8
"""
通过【list文件】和【train所有文件放到一个文件夹中】方式读取数据
"""
from torch.utils.data import Dataset
from PIL import Image
import csv
import os
import tifffile
import numpy as np

# TODO:2017-06-26:先把multilabel走通，再把single label和multi labe代码合并

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

TIF_EXTENSIONS = [
    '.tif', '.tiff', '.TIF', '.TIFF'
]


def default_loader(path, filetype):
    """
    TODO: 重构filetype
    :param path: 
    :param filetype: 
    :return: 
    """
    if filetype in IMG_EXTENSIONS:
        return Image.open(path).convert('RGB')
    elif filetype in TIF_EXTENSIONS:
        imgs = tifffile.imread(path)
        imgs = imgs.astype(dtype=np.int32) # torch.from_numpy only supported types are: double, float, int64, int32, and uint8.
        return imgs


def lstfile_single(root, lstpath):
    """
    将文件路image路径组装成tuple
    file organization：
    id, class, file_path
    1 , 4 , plane.jpg
    :param root: train/val 文件路径
    :param lstpath: 类别以及文件对应关系文件
    :return: images(image_path, class)
    """
    images = []
    with open(lstpath,'r',) as f:
        for line in csv.reader(f, delimiter="\t"):
            cls = int(line[1])
            filename = line[2]
            item = (os.path.join(root, filename), cls)
            if os.path.exists(os.path.join(root, filename)):
                images.append(item)
    return images


def lstfile_multi(root, lstpath):
    """
    将文件路image路径组装成tuple
    file organization：
    id, cls1, cls2,... file_path
    1,  1, 0, 1 ... plane_people.jpg
    :param root: train/val 文件路径
    :param lstpath: 类别以及文件对应关系文件
    :return: images(image_path, class)
    """
    images = []
    with open(lstpath, 'r',) as f:
        for line in csv.reader(f, delimiter="\t"):
            cls = np.array([int(x) for x in line[1:-1]]) # 必须是np.array, 不能用list，否则enumerate时target不是tensor
            filename = line[-1]
            item = (os.path.join(root, filename), cls)
            if os.path.exists(os.path.join(root, filename)):
                images.append(item)
    return images


class ImageFloderLstSig(Dataset):
    """
    classification: single label, file list
    """
    def __init__(self, root, lstpath, filetype='tif', transform=None, target_transform=None, loader=default_loader):
        imgs = lstfile_single(root, lstpath)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.filetype = '.'+filetype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path, self.filetype)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFloderLstMulti(Dataset):
    """
    classification: multi label, file list
    """
    def __init__(self, root, lstpath, filetype='tif', transform=None, target_transform=None, loader=default_loader):
        imgs = lstfile_multi(root, lstpath)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.filetype = '.' + filetype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path, self.filetype)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageFloderSeg(Dataset):
    """
    图像分割读取处理
    """
    pass

if __name__ == '__main__':
    root = '../../data/cifar/train_single'
    lstpath = '../../data/cifar/train_single_list.lst'
    a = ImageFloderLstSig(root, lstpath)
    pass
