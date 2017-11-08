import numpy as np
import cv2


def chrominance2rgb(grayscale, chrominance):
    gr_shape = grayscale.shape
    grayscale = grayscale.reshape((gr_shape[0], gr_shape[1], 1))
    yuv = np.concatenate((grayscale, chrominance), axis=2)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return rgb


def rgb2chrominance(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    grayscale = yuv[:, :, 0]
    chrominance = yuv[:, :, 1:]
    return grayscale, chrominance


if __name__ == '__main__':
    chrominance = cv2.imread('test1.png')
    grayscale = cv2.imread('test3.png', 0)
    rgb = cv2.imread('test2.png')
    grayscale, chrominance = rgb2chrominance(rgb)
    sh = rgb.shape
    mid = np.zeros(sh[:2], dtype=np.uint8)
    for i in range(sh[0]):
        for j in range(sh[1]):
            mid[i, j] = 200
    print(mid.shape)
    cr_show = chrominance2rgb(mid, chrominance)
    res = chrominance2rgb(grayscale, chrominance)
    cv2.imshow('test2', res)
    cv2.imshow('test', cr_show)
    cv2.waitKey(0)
