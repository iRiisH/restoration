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
import progressbar

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
VOC_COLORS = [label[2] for label in VOC_LABELS]
VOC_NAMES = [label[0] for label in VOC_LABELS]


def compute_daisy(img):
    """ computes DAISY features of the image """
    radius = 10
    padd_img = np.pad(img, ((radius, radius), (radius, radius)), 'mean')
    d = daisy(padd_img, step=1, radius=radius, rings=2, histograms=6, orientations=8)
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


def color_cat(c1):
    """ returns the category associated to the color in the jpg file """
    dists = [np.linalg.norm(c1 - c2) for c2 in VOC_COLORS]
    return np.argmin(dists)


def get_seg(filename):
    """ loads the segmentation result for a given filename """
    jpg = cv2.imread(os.path.join('data', 'seg', filename))
    jpg = np.flip(jpg, axis=2)
    m, n = jpg.shape[:2]
    n_cl = len(VOC_COLORS)
    tile = np.tile(jpg, (n_cl, 1, 1)).reshape((n_cl, m, n, 3))
    dist = np.zeros((n_cl, m, n), dtype=np.float)
    for i in range(n_cl):
        dist[i] = np.linalg.norm(np.subtract(tile[i], VOC_COLORS[i]), axis=2)

    category = np.argmin(dist, axis=0)
    # img = utils.color_class_image(category, 'voc')
    # misc.imshow(img)
    m, n = category.shape
    res_vec = np.zeros((m, n, len(VOC_LABELS)), dtype=np.float)
    for i in range(m):
        for j in range(n):
            res_vec[i, j, category[i, j]] = 1.
    return res_vec


def compute_features(img, filename=None, network=None):
    """ compute the pyramid of features for a given image, or loads it
    if a filename is provided """
    if filename is None:
        if img is None:
            raise ValueError()
        seg = compute_seg(network, img)
    else:
        print('loading...')
        seg = get_seg(filename)
        if img is None:
            img = cv2.imread(os.path.join('data', 'batch', filename), 0)
    m, n = img.shape[:2]
    print('segmentation done')
    neighb = compute_neighb(img)
    print('neighbourhood created')
    daisy = compute_daisy(img)
    print('daisy done')
    feats = np.concatenate((neighb, daisy, seg), axis=2)
    return feats.reshape((m*n, feats.shape[2]))


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
            dst_path = os.path.join('data', 'seg', filename)
            if os.path.isfile(dst_path):
                continue  # the current image has already been segmented
            img = misc.imread(input_path)
            if len(img.shape) <= 2:
                print('{} DISCARDED: INCORRECT DIMENSION/ GRAYSCALE IMAGE'.format(filename))
                continue
            class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, False, False)
            class_image = np.argmax(class_scores, axis=2)
            colored_class_image = utils.color_class_image(class_image, model)

            misc.imsave(dst_path, colored_class_image)


def test_classes():
    n = int(np.sqrt(len(VOC_NAMES)))+1
    t = np.zeros((n, n, 3))
    fin = False
    for i in range(n):
        if fin:
            break
        for j in range(n):
            k = n*i+j
            if k >= len(VOC_NAMES):
                fin=True
                break
            t[i, j] = VOC_COLORS[k]
    misc.imshow(t)


if __name__ == '__main__':
    compute_features(None, 'sun_aagzfdhtedskkkvb.jpg')
    # test_classes()
    # t=misc.imread('data/seg/sun_aagzfdhtedskkkvb.jpg')
    # print(t)
    # misc.imshow(t)