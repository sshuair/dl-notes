# coding=utf-8
from torch.utils.data import Dataset
from PIL import Image
import csv
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def phare_lstfile(root, lstpath):
    """
    将文件路image路径组装成tuple
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


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFloderSingle(Dataset):
    def __init__(self, root, lstpath, transform=None, target_transform=None, loader=default_loader):
        imgs = phare_lstfile(root, lstpath)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    root = '../../data/cifar/train_single'
    lstpath = '../../data/cifar/train_single_list.lst'
    a = ImageFloderSingle(root, lstpath)
    pass
