from scipy import misc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def imshow(img):
    plt.imshow(img)
    plt.show()


def chrominance2rgb(grayscale, chrominance):
    gr_shape = grayscale.shape
    grayscale = grayscale.reshape((gr_shape[0], gr_shape[1], 1))
    yuv = np.concatenate((grayscale, chrominance), axis=2)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return rgb


def rgb2chrominance(img):
    print(img.shape)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    grayscale = yuv[:, :, 0]
    chrominance = yuv[:, :, 1:]
    return grayscale, chrominance


if __name__ == '__main__':
    rgb = misc.imread(os.path.join(os.getcwd(), '4.jpg'))
    grayscale, chrominance = rgb2chrominance(rgb)
    res = chrominance2rgb(grayscale, chrominance)
    imshow(res)
