from network import *
from consts import *
from stats import *


def train():
    """ straightforward training over Sift Flow "natural" clusters """
    for cat in get_clusters():
        if cat == 'coast':
            continue
        sess = tf.Session()
        net = NeuralNet()
        file_dict = get_file_dict('trainval')
        file_list = file_dict[cat]
        net.train(sess, file_list, cat, max_it=30000)


def adaptative_clustering(cat):
    """ performs the adaptative clustering algorithm described in the original paper """
    img_dict = get_file_dict('trainval')
    img_list = img_dict[cat]
    # set parameters
    mu = 80
    epsilon = -26
    while len(img_list) > mu:
        new_img_list = []
        sess = tf.Session()
        net = NeuralNet()
        net.train(sess, img_list, cat, max_it=25000)
        for filename in img_list:
            res = test_model(filename, cat)
            gt = misc.imread(os.path.join(IMG_DIR, filename))
            ratio = psnr(res, gt)
            if ratio > epsilon:
                new_img_list.append(filename)
        img_list = new_img_list
    return img_list


if __name__ == '__main__':
    train()

    # img = test_model('coast_land334.jpg', 'coast')
    # imshow(img)

    # img_list = adaptative_clustering('coast')
    # for filename in img_list:
    #     print(filename)
