from scipy import misc
import matplotlib.pyplot as plt
import cv2
from create_features import compute_features
from consts import *


def imshow(img):
    plt.imshow(img)
    plt.show()


def chrominance2rgb(grayscale, chrominance):
    gr_shape = grayscale.shape
    grayscale = grayscale.reshape((gr_shape[0], gr_shape[1], 1))
    yuv = np.concatenate((grayscale, chrominance), axis=2)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return rgb


def rgb2chrominance(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    grayscale = yuv[:, :, 0]
    chrominance = yuv[:, :, 1:]
    return grayscale, chrominance


def get_train_dict():
    res = {}
    with open('trainval.txt', 'r') as f:
        for line in f:
            cat = line.split('_')[0]
            if cat not in res:
                res[cat] = []
            res[cat].append(line[:-1])
    return res


if __name__ == '__main__':
    # rgb = misc.imread(os.path.join(os.getcwd(), '4.jpg'))
    # grayscale, chrominance = rgb2chrominance(rgb)
    # print(chrominance.shape)
    # res = chrominance2rgb(grayscale, chrominance)
    # imshow(res)
    f1, c1 = load_feats('coast_bea1.jpg')
    f2, c2 = load_feats('coast_bea3.jpg')
    res_f = np.concatenate((f1, f2), axis=0)
    res_c = np.concatenate((c1, c2), axis=0)
    print(res_f.shape, res_c.shape)
