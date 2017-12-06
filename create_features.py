from scipy import misc
from skimage.feature import daisy
from consts import *
from utils import *


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
    return res


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


def load_feats(filename):
    """ returns features + chrominance ground truth value for a given filename """
    print('Loading {}...'.format(filename))
    features = compute_features(None, filename)
    img = misc.imread(os.path.join(IMG_DIR, filename))
    m, n = img.shape[:2]
    _, chrominance = rgb2chrominance(img)
    chrominance = chrominance.reshape((m*n, 2))
    return features, chrominance


if __name__ == '__main__':
    get_seg('coast_cdmc969.bmp')
