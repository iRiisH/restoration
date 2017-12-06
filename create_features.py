from skimage.feature import daisy
from utils import *
import random as rd

def compute_daisy(img):
    """ computes DAISY features of the image """
    radius = 58
    padd_img = np.pad(img, ((radius, radius), (radius, radius)), 'mean')
    d = daisy(padd_img, step=1, radius=radius, rings=1, histograms=3, orientations=8)
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
    return res / 255.


def compute_seg(network, img):
    """ computes the segmengtation for a given PSPNet model """
    # TODO


def get_seg(filename):
    """ loads the segmentation result for a given filename """
    in_path = os.path.join(SEG_DIR, filename)
    img = misc.imread(in_path)
    m, n = img.shape[:2]
    seg_res = np.zeros((m, n, len(LABELS)), dtype=np.float)
    for i in range(m):
        for j in range(n):
            r, g, b = img[i, j]
            seg_res[i, j, COLORS_INDEX[(r, g, b)]] = 1.
    return seg_res


def compute_features(img, filename=None, net=None):
    """ compute the pyramid of features for a given image, or loads it
    if a filename is provided """
    if filename is not None:
        if img is None:
            img = misc.imread(os.path.join(IMG_DIR, filename), mode='L')
        name = filename.split('.')[0]
        seg = get_seg(name+'.bmp')
    else:
        # forward pass to get the segmentation
        in_ = np.array(img, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= MEAN
        in_ = in_.transpose((2, 0, 1))

        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        net.forward()
        seg = net.blobs['score_sem'].data[0].argmax(axis=0)

    m, n = img.shape[:2]
    # print('segmentation done')
    neighb = compute_neighb(img)
    # print('neighbourhood created')
    daisy = compute_daisy(img)
    # print('daisy done')
    feats = np.concatenate((neighb, daisy, seg), axis=2)
    return feats.reshape((m*n, feats.shape[2]))


def normalize(feats):
    # normalize features
    stddev = STD_FEATS
    for i in range(33):
        # features corresponding to the segmentation are already formatted, the do not need to be normalized
        stddev[len(stddev)-1-i] = 1.
    return (feats - MEAN_FEATS) / stddev


def load_feats(filename):
    """ returns features + chrominance ground truth value for a given filename """
    print('Loading {}...'.format(filename))
    features = compute_features(None, filename)
    img = misc.imread(os.path.join(IMG_DIR, filename))
    m, n = img.shape[:2]
    _, chrominance = rgb2chrominance(img)
    chrominance = chrominance.reshape((m*n, 2)) / 255.
    return normalize(features), chrominance


def estimate_moments():
    """ estimates mean & stddev of the features distribution. As computing features is
    quite long, we only compute moments over 50 images"""
    res = []
    files = []
    with open('trainval.txt', 'r') as f:
        for line in f:
            files.append(line[:-1])
    rd.shuffle(files)

    for filename in files[:50]:
        feats,_ = load_feats(filename)
        res.append(feats)
    arr = np.array(res).reshape((len(res)*256*256, 114))
    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    print('MEAN')
    for x in mean:
        print(x)
    print('STD')
    for x in std:
        print(x)


if __name__ == '__main__':
    # get_seg('coast_cdmc969.bmp')
    estimate_moments()
