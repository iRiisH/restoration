import tensorflow as tf
import time
import random as rd

from create_features import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def nn_layer(input_tensor, input_dimension, output_dimension, layer_name, activation=tf.nn.relu):
    """
    a fully-connected layer
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dimension, output_dimension])
        with tf.name_scope('biases'):
            biases = bias_variable([output_dimension])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        activated = activation(preactivate, name='activation')
    return activated


def date():
    """
    computes date string (used to name log files)
    """
    t = time.localtime()
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month = months[t.tm_mon - 1]
    return str(t.tm_mday) + month.upper() + str(t.tm_hour) + 'h' + str(t.tm_min)


class NeuralNet:
    """
    implements a siamese neural network based on a MLP architecture
    """
    def __init__(self):
        self.btneck_shape = 114
        self.learning_rate = 0.001

        with tf.name_scope('input'):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.btneck_shape],
                                        name='input')
            self.ground_truth = tf.placeholder(dtype=tf.float32, shape=[None, 2],
                                                 name='ref_value')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        with tf.name_scope('output'):
            self.out = self.create_model(self.input, self.keep_prob)
            self.chrominance = tf.nn.softmax(self.out)
        with tf.name_scope('train'):
            with tf.name_scope('cost'):
                # uses stable tf softmax directly inside cost function
                self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.out))
                # configure the network for tensorboard use
                tf.summary.scalar('cross_entropy', self.cross_entropy)
                self.merged = tf.summary.merge_all()
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

    def create_model(self, input, keep_prob):
        with tf.name_scope('fc1') as _:
            fc1w = tf.Variable(tf.truncated_normal([self.btneck_shape, 57],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[57], dtype=tf.float32),
                               trainable=True, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(input, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.drop1 = tf.nn.dropout(self.fc1, keep_prob)

        # fc2
        with tf.name_scope('fc2') as _:
            fc2w = tf.Variable(tf.truncated_normal([57, 57],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[57], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.drop1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.drop2 = tf.nn.dropout(self.fc2, keep_prob)

        # fc3
        with tf.name_scope('fc3') as _:
            fc3w = tf.Variable(tf.truncated_normal([57, 57],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[57], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3 = tf.nn.bias_add(tf.matmul(self.drop2, fc3w), fc3b)
            self.drop3 = tf.nn.dropout(self.fc3, keep_prob)

        with tf.name_scope('fc4') as _:
            fc4w = tf.Variable(tf.truncated_normal([57, 2],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc4b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc4l = tf.nn.bias_add(tf.matmul(self.drop3, fc4w), fc4b)
        return self.fc4l

    def train(self, sess, cat, max_it=8000):
        """
        trains the network for max_it iterations, regularly saves it and computes validation
        score.
        """
        # init
        date_str = date()
        saver = tf.train.Saver(max_to_keep=1)
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'+date_str), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'valid'+date_str))
        tf.global_variables_initializer().run(session=sess)
        train_dict = get_train_dict()
        img_list = train_dict[cat]
        n_img = len(img_list)
        valid_list, train_list = img_list[:n_img/5], img_list[n_img/5:]
        batch_size = 50
        test_batch_size = 200
        save_every_n_it = 1000

        # training
        print('TRAINING NETWORK FOR CLUSTER {}'.format(cat.upper()))
        cnt = 0
        features, ground_truth = load_feats(train_list[rd.randint(0, len(train_list)-1)])
        for k in range(1, max_it+1):
            test = k % 100 == 0
            if not test:
                if cnt >= 100:
                    # update features using a new image
                    features, ground_truth = load_feats(train_list[rd.randint(0, len(train_list) - 1)])
                    cnt = 0
                # pick random pixel on the image as batch
                ind = [rd.randint(0, len(features) - 1) for _ in range(batch_size)]
                y = [features[i] for i in ind]
                y_ = [ground_truth[i] for i in ind]
                dropout_rate = 0.5  # since we're training
                cnt += 1
                summary, _ = sess.run([self.merged, self.train_step],
                                      feed_dict={self.input: y,
                                                 self.ground_truth: y_,
                                                 self.keep_prob: 1.-dropout_rate
                                                 })

                train_writer.add_summary(summary, k)
                if k % 10 == 0:
                    print('Iteration No. {:4d}: training...'.format(k))
            else:
                test_img = valid_list[rd.randint(0, len(valid_list)-1)]
                valid_features, valid_ground_truth = load_feats(test_img)
                ind = [rd.randint(0, len(valid_features) - 1) for _ in range(test_batch_size)]
                y = [valid_features[i] for i in ind]
                y_ = [valid_ground_truth[i] for i in ind]
                summary = sess.run([self.merged],
                                   feed_dict={self.input: y,
                                              self.ground_truth: y_,
                                              self.keep_prob: 1.
                                              })
                valid_writer.add_summary(summary, k)

                print('Iteration No. {:4d}: VALIDATION STEP'.format(k))
                ckpt = (k % save_every_n_it == 0 and k > 0)
                if ckpt:
                    print('Saving checkpoint...')
                    saver.save(sess, os.path.join(LOG_DIR, 'model_siam', 'model'), global_step=k)


if __name__ == '__main__':
    sess = tf.Session()
    net = NeuralNet()
    net.train(sess, 'coast', max_it=5000)
    # net = Siamese()
    # net.train(sess, max_it=20000)
