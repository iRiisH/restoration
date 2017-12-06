import numpy as np
from PIL import Image
import os

import caffe
import time
import matplotlib.pyplot as plt
import progressbar

MEAN = np.array((104.00698793, 116.66876762, 122.67891434))

colors = [(255,0,0), (102,51,51), (229,130,115), (178,71,0), (102,41,0), (255,166,64), (115,96,57), (202,217,0), (95,102,0), (238,242,182), (113,179,89), (77,102,77), (0,255,34), (0,255,170), (35,140,105), (191,255,234), (0,190,204), (0,77,115), (61,182,242), (182,214,242), (77,90,102), (0,31,115), (102,129,204), (92,51,204), (204,0,255), (218,121,242), (217,163,213), (102,77,100), (128,0,102), (255,0,170), (242,0,97), (242,121,170), (166,0,22)]
labels = ['awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus', 'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field', 'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river', 'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky', 'staircase', 'streetlight', 'sun', 'tree', 'window']


def segment_database():
    file_list = []
    img_dir = os.path.join(os.getcwd(), 'data', 'SiftFlowDataset', 'Images', 'spatial_envelope_256x256_static_8outdoorcategories')
    seg_dir = os.path.join(os.getcwd(), 'data', 'segmented')
    # TODO: fill file_list
    with open('list.txt', 'r') as f:
        for line in f:
            file_list.append(line[:-1])

    net = caffe.Net('fcn.berkeleyvision.org/siftflow-fcn16s/test2.prototxt',
                    'fcn.berkeleyvision.org/siftflow-fcn16s/siftflow-fcn16s-heavy.caffemodel',
                    caffe.TEST)
    bar = progressbar.ProgressBar()
    n = len(file_list)
    for i in bar(range(n)):
        filename = file_list[i]
        path = os.path.join(img_dir, filename)
        name = filename.split('.')[0]
        dst_path = os.path.join(seg_dir, name + '.bmp')
        if os.path.isfile(dst_path):
            continue
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(path)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= MEAN
        in_ = in_.transpose((2, 0, 1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        net.forward()
        out = net.blobs['score_sem'].data[0].argmax(axis=0)
        m, n = out.shape
        img = np.zeros((m, n, 3), dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                img[i, j] = colors[out[i,j]]
                # print(labels[out[i, j]])

        res_img = Image.fromarray(img)
        res_img.save(dst_path)


def choose_batch_size():
    im = Image.open('./4.jpg')
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= MEAN
    m, n = in_.shape[:2]
    in_ = in_.transpose((2, 0, 1))

    net = caffe.Net('fcn.berkeleyvision.org/siftflow-fcn16s/test2.prototxt',
                    'fcn.berkeleyvision.org/siftflow-fcn16s/siftflow-fcn16s-heavy.caffemodel',
                    caffe.TEST)
    X = []
    Y = []
    for batch_size in range(1, 20):
        print(batch_size)
        res = np.repeat([in_], batch_size, axis=0).reshape(batch_size, 3, m, n)
        net.blobs['data'].reshape(batch_size, *in_.shape)
        net.blobs['data'].data[...] = res

        # run net and take argmax for prediction
        t1 = time.time()
        net.forward()
        t2 = time.time()
        # out = net.blobs['score_sem'].data[0].argmax(axis=0)
        print(res.shape)
        X.append(batch_size)
        Y.append((t2-t1)/batch_size)
        print(t2-t1)
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    segment_database()
