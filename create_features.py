import cv2
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
from scipy import misc
from PSPNet.pspnet import PSPNet50, PSPNet101, predict_multi_scale
import PSPNet.utils as utils


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


if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)
    input_path = os.path.join(os.getcwd(), '3.jpg')
    with sess.as_default():
        img = misc.imread(input_path)
        model = 'pspnet101_voc2012'

        pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                           weights=model)

        class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, False, False)

        print("Writing results...")

        class_image = np.argmax(class_scores, axis=2)
        pm = np.max(class_scores, axis=2)
        colored_class_image = utils.color_class_image(class_image, model)
        # colored_class_image is [0.0-1.0] img is [0-255]
        alpha_blended = 0.5 * colored_class_image + 0.5 * img
        misc.imsave('test' + "_seg.jpg", colored_class_image)
        # misc.imsave(filename + "_probs" + ext, pm)
        # cv2.imshow('test', alpha_blended)
        # cv2.waitKey(0)
        # misc.imsave('test' + "_seg_blended.jpg", alpha_blended)
