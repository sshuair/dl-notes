# coding=utf-8

"""
change the cifar-10 python format to single image
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


fp='../data/cifar-10-batches-py/data_batch_1'
result = unpickle(fp)
pass