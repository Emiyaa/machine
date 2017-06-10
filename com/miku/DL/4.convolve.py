#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image


def convolve(image, weight):
    height, width = image.shape
    h, w = weight.shape
    height_new = height - h + 1
    width_new = width - w + 1
    image_new = np.zeros((height_new, width_new), dtype=np.float)
    for i in range(height_new):
        for j in range(width_new):
            image_new[i,j] = np.sum(image[i:i+h, j:j+w] * weight)
    image_new = image_new.clip(0, 255)
    image_new = np.rint(image_new).astype('uint8')
    return image_new

# image_new = 255 * (image_new - image_new.min()) / (image_new.max() - image_new.min())

if __name__ == "__main__":
    A = Image.open("lena.png", 'r')
    output_path = '.\\Pic\\'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    print(type(a))
    print(a.shape)
    print(a[:, :, 0])
    avg = np.random.rand(5, 5)
    avg /= avg.sum()
    soble_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    soble_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    soble = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
    prewitt_x = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1,-1], [0, 0, 0], [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))
    laplacian = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
    laplacian2 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))
    weight_list = ('avg', 'soble_x', 'soble_y', 'soble', 'prewitt_x', 'prewitt_y', 'prewitt', 'laplacian', 'laplacian2')
    print('梯度检测：')
    for weight in weight_list:
        print(weight)
        R = convolve(a[:, :, 0], eval(weight))
        G = convolve(a[:, :, 1], eval(weight))
        B = convolve(a[:, :, 2], eval(weight))
        I = np.stack((R, G, B), 2)
        Image.fromarray(I).save(output_path + weight + '.png')
