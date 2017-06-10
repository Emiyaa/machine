# /usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pickle
from PIL import Image
import os


# Python 2.7
def unpickle2(file):
    with open(file, 'rb') as fo:
        dit = pickle.load(fo)
    return dit


def unpickle(file):
    with open(file, 'rb') as fo:
        dit = pickle.load(fo, encoding='bytes')
    return dit


# 数据地址：http://www.cs.toronto.edu/~kriz/cifar.html
if __name__ == '__main__':
    a = unpickle('cifar-10-batches-py\\data_batch_1')
    print(a.keys())
    print(a[b'data'])
    print(a[b'data'].shape)
    print(a[b'labels'])
    print(a[b'batch_label'])
    print(a[b'filenames'])
    m = a[b'data'].shape[0]

    path = 'Output\\'
    if not os.path.exists(path):
        os.mkdir(path)
    for i, (image, filename) in enumerate(zip(a[b'data'], a[b'filenames'])):
        if i % 100 == 0:
            print(i/m)
        r, g, b = image[:1024], image[1024:2048], image[2048:]
        r = r.reshape(32,32)
        g = g.reshape(32,32)
        b = b.reshape(32,32)
        I = np.stack((r, g, b), 2)
        Image.fromarray(I).save(path+filename.decode())
