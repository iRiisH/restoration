from PIL import Image

import caffe
import time
import matplotlib.pyplot as plt
import progressbar

from consts import *


def segment_image():
    print('')
    # TODO


def segment_database():
    """ segments and save as .bmp all images in the Sift Flow dataset
    using the FCN model
    /!\ process can be quite long """
    file_list = []

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
        path = os.path.join(IMG_DIR, filename)
        name = filename.split('.')[0]
        dst_path = os.path.join(SEG_DIR, name + '.bmp')  # bmp -> no compression, to avoid creating artifacts
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
        for k in range(m):
            for l in range(n):
                img[k, l] = COLORS[out[k, l]]
                # print(labels[out[k, l]])

        res_img = Image.fromarray(img)
        res_img.save(dst_path)


# def choose_batch_size():
#     im = Image.open('./4.jpg')
#     in_ = np.array(im, dtype=np.float32)
#     in_ = in_[:, :, ::-1]
#     in_ -= MEAN
#     m, n = in_.shape[:2]
#     in_ = in_.transpose((2, 0, 1))
#
#     net = caffe.Net('fcn.berkeleyvision.org/siftflow-fcn16s/test2.prototxt',
#                     'fcn.berkeleyvision.org/siftflow-fcn16s/siftflow-fcn16s-heavy.caffemodel',
#                     caffe.TEST)
#     X = []
#     Y = []
#     for batch_size in range(1, 20):
#         print(batch_size)
#         res = np.repeat([in_], batch_size, axis=0).reshape(batch_size, 3, m, n)
#         net.blobs['data'].reshape(batch_size, *in_.shape)
#         net.blobs['data'].data[...] = res
#
#         # run net and take argmax for prediction
#         t1 = time.time()
#         net.forward()
#         t2 = time.time()
#         # out = net.blobs['score_sem'].data[0].argmax(axis=0)
#         print(res.shape)
#         X.append(batch_size)
#         Y.append((t2-t1)/batch_size)
#         print(t2-t1)
#     plt.plot(X, Y)
#     plt.show()


if __name__ == '__main__':
    segment_database()
