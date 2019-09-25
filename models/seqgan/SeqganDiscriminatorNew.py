import tensorflow as tf
import numpy as np
from models.Gan import Dis

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class DiscriminatorNew(Dis):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            emd_dim, filter_sizes, num_filters, splited_steps, l2_reg_lambda=0.0, dropout_keep_prob = 1,
            ):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_new")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y_new")
        self.dropout_keep_prob = dropout_keep_prob
        self.splited_steps = splited_steps
        self.sequence_length = sequence_length
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator_new'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, emd_dim], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

                #Data calculation and printing
                self.evaluation_info()
                #Pick negative samples of the specified score
                self.high_marks_samples()

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                self.d_loss = tf.reshape(tf.reduce_mean(self.loss), shape=[1])

        self.params = [param for param in tf.trainable_variables() if 'discriminator_new' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def evaluation_info(self):
        #Positive and negative samples number
        self.po_neg_num = [tf.cast(tf.count_nonzero(self.input_y[:, 1]), tf.float32),
                           tf.cast(tf.count_nonzero(self.input_y[:, 0]), tf.float32)]

        positive = self.ypred_for_auc[:, 1] * self.input_y[:, 1]
        self.negtive = self.ypred_for_auc[:, 1] * self.input_y[:, 0]
        self.positive_average = tf.reduce_sum(positive) / self.po_neg_num[0]
        self.negtive_average = tf.reduce_sum(self.negtive) / self.po_neg_num[1]
        self.positive_sum = tf.reduce_sum(positive)
        self.negtive_sum = tf.reduce_sum(self.negtive)
        # relative distance
        self.po_neg = tf.subtract(self.positive_average, self.negtive_average)

        ############
        ##absolute distance
        ############
        self.real_sig_greater_index = tf.cast(tf.greater(positive, 0.5), tf.float32)
        self.real_sig_less_index = tf.cast(tf.less_equal(positive, 0.5), tf.float32)
        self.fake_sig_greater_index = tf.cast(tf.greater(self.negtive, 0.5), tf.float32)
        self.fake_sig_less_index = tf.cast(tf.less_equal(self.negtive, 0.5), tf.float32)

        self.real_sig_distance = tf.reduce_sum(
            positive * self.real_sig_greater_index - positive * self.real_sig_less_index) / self.po_neg_num[0]
        self.fake_sig_distance = tf.reduce_sum(
            self.negtive * self.fake_sig_greater_index - self.negtive * self.fake_sig_less_index) / self.po_neg_num[1]

        self.real_sig_distance_sum = tf.reduce_sum(
            positive * self.real_sig_greater_index - positive * self.real_sig_less_index)
        self.fake_sig_distance_sum = tf.reduce_sum(
            self.negtive * self.fake_sig_greater_index - self.negtive * self.fake_sig_less_index)

        self.po_neg_absolute = self.real_sig_distance - self.fake_sig_distance
        self.po_true = self.real_sig_greater_index
        self.neg_true = self.fake_sig_less_index * self.input_y[:, 0]

        ############
        # tensorboard infos
        ############
        # PRE
        # pre-train-info
        absolute_po_sub_neg = tf.summary.scalar('absolute_po_neg', self.po_neg_absolute)
        po_sub_neg = tf.summary.scalar('po_neg', self.po_neg)
        # pre-valid-info
        absolute_po_sub_neg_valid = tf.summary.scalar('absolute_po_neg_valid', self.po_neg_absolute)
        valid_po_sub_neg = tf.summary.scalar('valid_po_neg', self.po_neg)

        # ADVERSARIAL
        # ad-train
        ad_absolute_po_sub_neg = tf.summary.scalar('ad_absolute_po_neg', self.po_neg_absolute)
        ad_po_sub_neg = tf.summary.scalar('ad_po_neg', self.po_neg)
        # ad-valid
        ad_absolute_po_sub_neg_valid = tf.summary.scalar('ad_absolute_po_neg_valid', self.po_neg_absolute)
        ad_valid_po_sub_neg = tf.summary.scalar('ad_valid_po_neg', self.po_neg)

        self.merge_summary_train = tf.summary.merge(
            [po_sub_neg, absolute_po_sub_neg])
        self.merge_summary_valid = tf.summary.merge(
            [valid_po_sub_neg, absolute_po_sub_neg_valid])

        self.merge_summary_train_ad = tf.summary.merge(
            [ad_po_sub_neg, ad_absolute_po_sub_neg])
        self.merge_summary_valid_ad = tf.summary.merge(
            [ad_valid_po_sub_neg, ad_absolute_po_sub_neg_valid])

    def high_marks_samples(self):

        ##################
        #Screening negative samples from scoring intervals
        #Scoring interval <0.3,0.3-0.5,0.5-0.9,>0.9
        ###################
        self.fake_sig_less_03 = tf.cast(tf.less_equal(self.negtive, 0.3), tf.int32)
        self.fake_sig_greater_03 = tf.cast(tf.greater(self.negtive, 0.3), tf.int32)
        self.fake_sig_less_05 = tf.cast(tf.less_equal(self.negtive, 0.5), tf.int32)
        self.fake_sig_greater_05 = tf.cast(tf.greater(self.negtive, 0.5), tf.int32)
        self.fake_sig_less_09 = tf.cast(tf.less_equal(self.negtive, 0.9), tf.int32) 
        self.fake_sig_greater_09 = tf.cast(tf.greater(self.negtive, 0.9), tf.int32)

        fake_sig_greater_03_list = []
        for i in range(self.sequence_length):
            fake_sig_greater_03_list.append([self.fake_sig_greater_03])
        self.fake_sig_greater_03 = tf.transpose(tf.concat(fake_sig_greater_03_list, axis=0))
        fake_sig_less_03_list = [] 
        for i in range(self.sequence_length):
            fake_sig_less_03_list.append([self.fake_sig_less_03])
        self.fake_sig_less_03 = tf.transpose(tf.concat(fake_sig_less_03_list, axis=0))
        fake_sig_greater_05_list = [] 
        for i in range(self.sequence_length):
            fake_sig_greater_05_list.append([self.fake_sig_greater_05])
        self.fake_sig_greater_05 = tf.transpose(tf.concat(fake_sig_greater_05_list, axis=0))
        fake_sig_less_05_list = [] 
        for i in range(self.sequence_length):
            fake_sig_less_05_list.append([self.fake_sig_less_05])
        self.fake_sig_less_05 = tf.transpose(tf.concat(fake_sig_less_05_list, axis=0))
        fake_sig_less_09_list = []
        for i in range(self.sequence_length):
            fake_sig_less_09_list.append([self.fake_sig_less_09])
        self.fake_sig_less_09 = tf.transpose(tf.concat(fake_sig_less_09_list, axis=0))
        fake_sig_greater_09_list = []
        for i in range(self.sequence_length):
            fake_sig_greater_09_list.append([self.fake_sig_greater_09])
        self.fake_sig_greater_09 = tf.transpose(tf.concat(fake_sig_greater_09_list, axis=0))

        input_y_list = [] 
        for i in range(self.sequence_length):
            input_y_list.append([tf.cast(self.input_y[:, 0], tf.int32)])
        self.y_fake = tf.transpose(tf.concat(input_y_list, axis=0))
        self.neg_samples=self.input_x*self.y_fake

        self.neg_samples_less_03 = self.neg_samples*self.fake_sig_less_03
        self.neg_samples_03to05 = self.neg_samples*self.fake_sig_greater_03*self.fake_sig_less_05
        self.neg_samples_05to09 = self.neg_samples*self.fake_sig_less_09*self.fake_sig_greater_05
        self.neg_samples_greater_09 = self.neg_samples*self.fake_sig_greater_09

    def predict(self):
        pass