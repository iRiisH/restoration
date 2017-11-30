import numpy as np
from PIL import Image
import os
from scipy import misc

import caffe

colors = [(255,0,0), (102,51,51), (229,130,115), (178,71,0), (102,41,0), (255,166,64), (115,96,57), (202,217,0), (95,102,0), (238,242,182), (113,179,89), (77,102,77), (0,255,34), (0,255,170), (35,140,105), (191,255,234), (0,190,204), (0,77,115), (61,182,242), (182,214,242), (77,90,102), (0,31,115), (102,129,204), (92,51,204), (204,0,255), (218,121,242), (217,163,213), (102,77,100), (128,0,102), (255,0,170), (242,0,97), (242,121,170), (166,0,22)]
labels = ['awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus', 'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field', 'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river', 'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky', 'staircase', 'streetlight', 'sun', 'tree', 'window']

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('../4.jpg')
# im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('siftflow-fcn16s/test2.prototxt', 'siftflow-fcn16s/siftflow-fcn16s-heavy.caffemodel',
                caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

file_list = []
img_dir = os.path.join()
# run net and take argmax for prediction
net.forward()
out = net.blobs['score_sem'].data[0].argmax(axis=0)
m, n = out.shape
img = np.zeros((m, n, 3), dtype=np.uint8)
for i in range(m):
    for j in range(n):
        img[i, j] = colors[out[i,j]]
        print(labels[out[i, j]])
misc.imshow(img)
print(out.shape)
