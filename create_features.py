import tensorflow as tf
import numpy as np
import os
import cv2
import PSPNet.utils as utils
import colorsys
from scipy import misc
from PSPNet.pspnet import PSPNet50, PSPNet101, predict_multi_scale
from keras import backend as K
from skimage.feature import daisy
from collections import namedtuple

Label = namedtuple('Label', [
    'name',
    'id',
    'color'
])

DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order
EVALUATION_SCALES = [1.0]
VOC_LABELS = [Label('background', 0, (0, 0, 0)),
              Label('aeroplane', 1, (128, 0, 0)),
              Label('bicycle', 2, (0, 128, 0)),
              Label('bird', 3, (128, 128, 0)),
              Label('boat', 4, (0, 0, 128)),
              Label('bottle', 5, (128, 0, 128)),
              Label('bus', 6, (0, 128, 128)),
              Label('car', 7, (128, 128, 128)),
              Label('cat', 8, (64, 0, 0)),
              Label('chair', 9, (192, 0, 0)),
              Label('cow', 10, (64, 128, 0)),
              Label('diningtable', 11, (192, 128, 0)),
              Label('dog', 12, (64, 0, 128)),
              Label('horse', 13, (192, 0, 128)),
              Label('motorbike', 14, (64, 128, 128)),
              Label('person', 15, (192, 128, 128)),
              Label('pottedplant', 16, (0, 64, 0)),
              Label('sheep', 17, (128, 64, 0)),
              Label('sofa', 18, (0, 192, 0)),
              Label('train', 19, (128, 192, 0)),
              Label('tvmonitor', 20, (0, 64, 128)),
              Label('void', 21, (128, 64, 12))]


def compute_daisy(img):
    """ computes DAISY features of the image """
    radius = 10
    padd_img = np.pad(img, ((radius, radius), (radius, radius)), 'mean')
    d = daisy(img, step=1, radius=58, rings=2, histograms=6, orientations=8)
    return d


def compute_neighb(img):
    """ computes the 7x7 neighbourhoods """
    m, n = img.shape
    radius = 3
    padd_img = np.pad(img, ((radius, radius), (radius, radius)), 'mean')
    res = np.zeros((m, n, 49), dtype=np.float)
    for i in range(m):
        for j in range(n):
            neighb = padd_img[i:i+7, j:j+7]
            res[i, j] = neighb.reshape(49)
    return res


def compute_seg(network, img):
    """ computes the segmengtation for a given PSPNet model """
    class_scores = predict_multi_scale(img, network, EVALUATION_SCALES, False, False)
    return class_scores


def get_seg(filename):
    """ loads the segmentation result for a given filename """
    jpg = cv2.imread(os.path.join('data', 'seg', filename))
    voc_colors = list(zip(VOC_LABELS[:]['color'], VOC_LABELS[:]['id']))
    m, n = jpg.shape[:2]
    for i in range(m):
        for j in range(n):
            r, g, b = img[i, j]
            cl = colorsys.rgb_to_hsv(r, g, b)[0]
            category = int(cl * (360./137.5))


def compute_features(img, filename=None, network=None):
    """ compute the pyramid of features for a given image, or loads it
    if a filename is provided """
    if img is None:
        img = cv2.imread(os.path.join('data', 'batch', filename), 0)
        seg = compute_seg(network, img)
    else:
        seg = get_seg(filename)
    neighb = compute_neighb(img)
    daisy = compute_daisy(img)
    feats = np.concatenate((neighb, daisy, seg), axis=2)
    return feats


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
