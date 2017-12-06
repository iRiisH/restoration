import tensorflow as tf
import os
import time
import random as rd


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
            self.colored_pixels = tf.placeholder(dtype=tf.float32, shape=[None, 2],
                                                 name='ref_value')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        with tf.name_scope('output'):
            self.out = self.create_model(self.input, self.keep_prob)
            with tf.name_scope('cost'):
                self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.colored_pixels, logits=self.out))

        with tf.name_scope('train'):
            loss_op = self.loss()
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)

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


    def train(self, sess, max_it=8000):
        """
        trains the network for mat_it iterations, regularly saves it and computes validation
        score.
        ---------------------------------------------------------------------------------------------------------------
        NOTE: Checking if the network is converging is a bit challenging in this case, as the only feedback we have is
        the loss function, which is arbitrarely scaled.
        To do so, we can still assess the loss associated with a trivial output, that is to say the minimum loss we can
        obtain if we return a constant energy regardless of the input features.
        Let e be this energy, then the expected loss is:
        E[L] = P(Y = 0)L_s(e) + P(Y = 1)L_d(e)
        For voc dataset, P(Y = 1) = 0.8 (approximately). Given the shapes of L_s and L_d, we can minimize this function
        and obtain Lmin = 2.125 (for e = 3.66)
        Thus we have to get better than this cost.
        """
        date_str = date()
        saver = tf.train.Saver(max_to_keep=1)
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'+date_str), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'valid'+date_str))
        tf.global_variables_initializer().run(session=sess)
        train_list, valid_list = get_cross_val_lists()
        # train_list = train_list[:100]  # used to test if overfitting is possible

        former_loss_val = 0.
        for k in range(max_it+1):
            test = k % 100 == 0
            if not test:
                dropout_rate = 0.
                batch_size = 128
                file_list1 = [train_list[rd.randint(0, len(train_list)-1)]
                              for _ in range(batch_size)]
                file_list2 = [train_list[rd.randint(0, len(train_list)-1)]
                              for _ in range(batch_size)]
                features1, labels1 = load_features(file_list1)
                features2, labels2 = load_features(file_list2)
                y = []
                for i in range(batch_size):
                    y.append(contain_same(labels1[i], labels2[i]))
                if k < 1000:
                    summary, _ = sess.run([self.merged, self.train_step2],
                                          feed_dict={self.img1: features1,
                                                     self.img2: features2,
                                                     self.label: y,
                                                     self.keep_prob: 1.-dropout_rate
                                                     })
                else:
                    summary, _ = sess.run([self.merged, self.train_step3],
                                          feed_dict={self.img1: features1,
                                                     self.img2: features2,
                                                     self.label: y,
                                                     self.keep_prob: 1. - dropout_rate
                                                     })

                train_writer.add_summary(summary, k)
                if k % 10 == 0:
                    loss_val = sess.run(self.loss,
                                        feed_dict={self.img1: features1,
                                                   self.img2: features2,
                                                   self.label: y,
                                                   self.keep_prob: 1.-dropout_rate
                                                   })
                    print('Iteration No. {:4d}: training step, loss: {:10.2f}'.format(k, loss_val))
            else:
                batch_size = 400
                file_list1 = [valid_list[rd.randint(0, len(valid_list) - 1)]
                              for _ in range(batch_size)]
                file_list2 = [valid_list[rd.randint(0, len(valid_list) - 1)]
                              for _ in range(batch_size)]
                features1, labels1 = load_features(file_list1)
                features2, labels2 = load_features(file_list2)
                y = []
                for i in range(batch_size):
                    y.append(contain_same(labels1[i], labels2[i]))
                summary, loss_val = sess.run([self.merged, self.loss],
                                             feed_dict={self.img1: features1,
                                                        self.img2: features2,
                                                        self.label: y,
                                                        self.keep_prob: 1.
                                                        })
                valid_writer.add_summary(summary, k)
                if loss_val != 0. and k > 0:
                    improvement_rate = '{:.1f}%'.format(100.*(former_loss_val - loss_val)/loss_val)
                else:
                    improvement_rate = 'NaN'
                print('Iteration No. {:4d}: VALIDATION STEP, loss: {:.4f}, improvement: {}'
                      .format(k, loss_val, improvement_rate))
                former_loss_val = loss_val
            save_every_n_it = 1000
            ckpt = (k % save_every_n_it == 0 and k > 0)
            if ckpt:
                    print('Saving checkpoint...')
                    saver.save(sess, os.path.join(LOG_DIR, 'model_siam', 'model'), global_step=k)


if __name__ == '__main__':
    sess = tf.Session()
    # net = Siamese()
    # net.train(sess, max_it=20000)
