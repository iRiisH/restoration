import numpy as np
import os
import re
from scipy import misc
from consts import *


def get_clusters():
    cats = []
    with open('list.txt', 'r') as f:
        for line in f:
            cat = line.split('_')[0]
            if cat not in cats:
                cats.append(cat)
    print(cats)


def get_subclusters():
    cats = {}
    with open('list.txt', 'r') as f:
        for line in f:
            k = re.search('\d', line)
            subcat = line[:k.start()]
            cat = subcat.split('_')[0]
            if cat not in cats:
                cats[cat] = []
            if subcat not in cats[cat]:
                cats[cat].append(subcat)
    print(cats)


def convert(img):
    """ convert RGB img to array of int corresponding to the segmentation values """
    m, n = img.shape[:2]
    res = np.zeros((m*n), dtype=np.int)
    img = img.reshape((m*n, 3))
    for i in range(m*n):
        r, g, b = img[i]
        res[i] = COLORS_INDEX[(r, g, b)]
    return res


def semantic_histogram(filename):
    img = misc.imread(os.path.join('data', 'segmented', filename))
    ind = convert(img)
    m, n = img.shape[:2]
    hist = np.bincount(ind, minlength=33).astype(np.float)
    return hist / (m*n)


def get_semantic_histogram():
    hists = {}
    file_list = {}
    with open('list.txt', 'r') as f:
        for line in f:
            cat = line.split('_')
            if cat not in file_list:
                file_list[cat] = []
            file_list[cat].append(line[:-1])
    for cat in file_list.keys():
        n = len(file_list[cat])
        for filename in file_list[cat]:
            new_filename = filename.split('.')[0] + '.bmp'
            # change extension to bitmap
            h = semantic_histogram(new_filename)
        hists[cat] = h / n
    return hists


if __name__ == '__main__':
    get_clusters()
    get_subclusters()
    print(semantic_histogram('coast_bea5.bmp'))
    # get_semantic_histogram()
    #
    # print()