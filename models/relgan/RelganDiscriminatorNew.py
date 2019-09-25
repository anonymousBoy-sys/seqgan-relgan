import tensorflow as tf
import numpy as np

from models.Gan import Dis
from utils.ops import create_linear_initializer, conv2d, highway, linear


class DiscriminatorNew(Dis):

    def __init__(self, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn, grad_clip, splited_steps):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.num_rep = num_rep
        self.sn = sn
        self.grad_clip = grad_clip
        self.get_logits = tf.make_template('discriminator_new', self.logits)
        self.splited_steps = splited_steps

    def logits(self, x_onehot):
        batch_size = self.batch_size
        seq_len = self.seq_len
        vocab_size = self.vocab_size
        dis_emb_dim = self.dis_emb_dim
        num_rep = self.num_rep
        sn = self.sn

        # get the embedding dimension for each presentation
        emb_dim_single = int(dis_emb_dim / num_rep)
        assert isinstance(emb_dim_single, int) and emb_dim_single > 0

        filter_sizes = [2, 3, 4, 5]
        num_filters = [300, 300, 300, 300]
        dropout_keep_prob = 0.75

        d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                       initializer=create_linear_initializer(vocab_size))
        input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
        emb_x_re = tf.matmul(input_x_re, d_embeddings)
        # batch_size x seq_len x dis_emb_dim
        emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])

        # batch_size x seq_len x dis_emb_dim x 1
        emb_x_expanded = tf.expand_dims(emb_x, -1)
        # print('shape of emb_x_expanded: {}'.format(
        #     emb_x_expanded.get_shape().as_list()))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            conv = conv2d(emb_x_expanded, num_filter, k_h=filter_size, k_w=emb_dim_single,
                          d_h=1, d_w=emb_dim_single, sn=sn, stddev=None, padding='VALID',
                          scope="conv-%s" % filter_size)  # batch_size x (seq_len-k_h+1) x num_rep x num_filter
            out = tf.nn.relu(conv, name="relu_new")
            pooled = tf.nn.max_pool(out, ksize=[1, seq_len - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID',
                                    name="pool_new")  # batch_size x 1 x num_rep x num_filter
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(num_filters)
        # batch_size x 1 x num_rep x num_filters_total
        h_pool = tf.concat(pooled_outputs, 3)
        # print('shape of h_pool: {}'.format(h_pool.get_shape().as_list()))
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add highway
        # (batch_size*num_rep) x num_filters_total
        h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
        h_drop = tf.nn.dropout(h_highway, dropout_keep_prob, name='dropout_new')

        # fc
        fc_out = linear(h_drop, output_size=100,
                        use_bias=True, sn=sn, scope='fc_new')
        logits = linear(fc_out, output_size=1,
                        use_bias=True, sn=sn, scope='logits_new')
        logits = tf.squeeze(logits, -1)  # batch_size*num_rep
        return logits

    def predict(self):
        pass

    def set_train_op(self, d_loss, optimizer_name, d_lr, global_step, nadv_steps, decay, po_neg, real_sig, fake_sig):
        
        self.loss = d_loss
        # relative distance
        self.po_neg = po_neg
        #Average score of a batch positive and negative sample
        self.real_sig_mean = tf.reduce_mean(real_sig)  # (real_sig:[batch_size*num_rep])
        self.fake_sig_mean = tf.reduce_mean(fake_sig)  #

        real_sig_split = tf.concat([tf.split(real_sig, self.batch_size, 0)], axis=0)
        fake_sig_split = tf.concat([tf.split(fake_sig, self.batch_size, 0)], axis=0)
        self.real_sig = tf.reduce_mean(real_sig_split, axis=1)  # [batch_size]
        self.fake_sig = tf.reduce_mean(fake_sig_split, axis=1)

        ################
        # absolute distance
        # #############
        self.real_sig_greater_index = tf.cast(tf.greater(self.real_sig, 0.5), tf.float32)
        self.real_sig_less_index = tf.cast(tf.less_equal(self.real_sig, 0.5), tf.float32)
        self.fake_sig_greater_index = tf.cast(tf.greater(self.fake_sig, 0.5), tf.float32)
        self.fake_sig_less_index = tf.cast(tf.less_equal(self.fake_sig, 0.5), tf.float32)
        self.real_sig_distance = tf.reduce_mean(
            self.real_sig * self.real_sig_greater_index - self.real_sig * self.real_sig_less_index)
        self.fake_sig_distance = tf.reduce_mean(
            self.fake_sig * self.fake_sig_greater_index - self.fake_sig * self.fake_sig_less_index)
        self.po_neg_absolute = self.real_sig_distance - self.fake_sig_distance

        ############
        # tensorboard infos
        ############
        #PRE
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

        self.merge_summary_train = tf.summary.merge([po_sub_neg, absolute_po_sub_neg])
        self.merge_summary_valid = tf.summary.merge([valid_po_sub_neg, absolute_po_sub_neg_valid])

        self.merge_summary_train_ad = tf.summary.merge([ad_po_sub_neg,ad_absolute_po_sub_neg])
        self.merge_summary_valid_ad = tf.summary.merge([ad_valid_po_sub_neg, ad_absolute_po_sub_neg_valid])

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_new')
        if decay:
            d_lr = tf.train.exponential_decay(
                d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        
        if optimizer_name == "adam":
            d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        elif optimizer_name == "rmsprop":
            d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        else:
            raise AttributeError
            
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(d_loss, d_vars), self.grad_clip)
        self.train_op = d_optimizer.apply_gradients(zip(grads, d_vars))
