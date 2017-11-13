import cv2
import tensorflow as tf
import numpy as np
import os
import progressbar
import PSPNet.utils as utils
from keras import backend as K
from scipy import misc
from PSPNet.pspnet import PSPNet50, PSPNet101, predict_multi_scale



DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order
EVALUATION_SCALES = [1.0]


def compute_daisy(img, pixel):
    daisy = cv2.xfeatures2d.DAISY_create()


def safe_val(img, pixel):
    i, j = pixel
    m, n, p = img.shape[:2]
    if 0 <= i < m and 0 <= j < n:
        return img[i, j]
    return 0.


def compute_neighb(img, pixel):
    return []


def compute_seg(network, img, pixel):
    i, j = pixel
    class_scores = predict_multi_scale(img, network, EVALUATION_SCALES, False, False)
    class_image = np.argmax(class_scores, axis=2)


def segment_database():
    filenames = []
    with open('list.txt', 'r') as f:
        for line in f:
            filenames.append(line[:-1])
    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        model = 'pspnet101_voc2012'
        pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                           weights=model)
        bar = progressbar.ProgressBar()
        n = len(filenames)
        for i in bar(range(n)):
            filename = filenames[i]
            input_path = os.path.join('data', 'batch', filename)
            img = misc.imread(input_path)
            class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, False, False)
            class_image = np.argmax(class_scores, axis=2)
            colored_class_image = utils.color_class_image(class_image, model)
            dst_path = os.path.join('data', 'seg', filename)
            misc.imsave(dst_path, colored_class_image)


if __name__ == '__main__':
    segment_database()
