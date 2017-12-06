import re
from scipy import misc
from consts import *
import progressbar


def get_clusters():
    """ the list of clusters in the Sift Flow dataset"""
    cats = []
    with open('trainval.txt', 'r') as f:
        for line in f:
            cat = line.split('_')[0]
            if cat not in cats:
                cats.append(cat)
    return cats


def get_subclusters():
    """ the list of subclusters in the Sift Flow dataset"""
    cats = {}
    with open('trainval.txt', 'r') as f:
        for line in f:
            k = re.search('\d', line)
            subcat = line[:k.start()]
            cat = subcat.split('_')[0]
            if cat not in cats:
                cats[cat] = []
            if subcat not in cats[cat]:
                cats[cat].append(subcat)
    print(cats)


def to_ind(arr):
    r, g, b = arr
    return COLORS_INDEX[(r, g, b)]


def convert(img):
    """ convert RGB img to list of int corresponding to the segmentation values """
    # converting images without heavely slowing down the entire process is a bit tricky
    # in python, this is why this method might seem strange
    m, n = img.shape[:2]
    img_int = rgb_to_int(img.reshape((m*n, 3)))
    indices = np.searchsorted(COLORS_INT, img_int)  # we need to be clever for the pro
    return indices


def semantic_histogram(img):
    """ returns the semantic histogram associated to the segmentation img """
    ind = convert(img)
    m, n = img.shape[:2]
    hist = np.bincount(ind, minlength=33).astype(np.float)

    return hist / (m*n)


def get_semantic_histogram():
    hists = {}
    file_list = {}
    n_img = 0
    with open('trainval.txt', 'r') as f:
        for line in f:
            n_img += 1
            cat = line.split('_')[0]
            if cat not in file_list:
                file_list[cat] = []
            file_list[cat].append(line[:-1])
    print('Computing semantic histograms:')
    bar = progressbar.ProgressBar(max_value=n_img)
    cnt = 0
    for cat in file_list.keys():
        n = len(file_list[cat])
        hists[cat] = np.zeros(len(LABELS), dtype=np.float)
        for filename in file_list[cat]:
            bar.update(cnt)
            cnt += 1
            # change extension to bitmap + loads the segmentation
            img = misc.imread(os.path.join('data', 'segmented', filename.split('.')[0] + '.bmp'))
            h = semantic_histogram(img)
            hists[cat] = np.add(hists[cat], h)
        hists[cat] /= n
    print('')
    return hists


def hist_dist(hist1, hist2):
    """ euclidian distance between 2 semantic histograms """
    return np.linalg.norm(hist1-hist2)


def closest_cluster(img, hists):
    """ returns cluster the image is the closest """
    hist = semantic_histogram(img)
    dists = []
    for cat in sorted(hists.keys()):
        hist_cluster = hists[cat]
        dists.append(hist_dist(hist, hist_cluster))
    return np.argmin(dists)


if __name__ == '__main__':
    hists = get_semantic_histogram()
    clusters = get_clusters()
    print(hists)
    score = 0
    with open('test.txt', 'r') as f:
        for line in f:
            filename = line[:-1].split('.')[0]+'.bmp'
            seg = misc.imread(os.path.join(SEG_DIR, filename))
            clus = clusters[closest_cluster(seg, hists)]
            if clus == filename.split('_')[0]:
                score += 1
            print('{} -> {}'.format(filename, clus))
    print('SCORE: {}%'.format(score/4.))  # 400 test images
