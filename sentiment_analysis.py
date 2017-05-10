#!/usr/bin/env python3

import functools
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn
import biwigpuprep
from dataloader import tweetsloader
import os, sys, time, yaml


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification:

    def __init__(self, vocab_size, params):
        self.data = tf.placeholder(tf.int32, [params['batch_size'], n_steps])
        self.target = tf.placeholder(tf.int32, [params['batch_size'], n_classes])
        self.sequence_lengths = tf.placeholder(tf.int32, [params['batch_size']])
        self.vocab_size = vocab_size

        if 'n_layers' in params:
            self._num_layers = params['n_layers']
        else:
            self._num_layers = 2

        if 'n_hidden' in params:
            if isinstance(params['n_hidden'], list):
                self._num_hidden = params['n_hidden']
            else:
                self._num_hidden = [params['n_hidden']] * self._num_layers
        else:
            self._num_hidden = [128, 128]

        self.params = params

        # build graph
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cells = []

        # dropout
        if 'dropout' in self.params:
            keep_prob = tf.constant(self.params['dropout'])
        else:
            keep_prob = tf.constant(0.6)

        # create multi layer RNN
        for i in range(self._num_layers):
            cell = rnn.GRUCell(self._num_hidden[i])
            cell = rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            cells.append(cell)

        # embedding layer
        embeddings = tf.get_variable('embedding_matrix', [self.vocab_size, self._num_hidden[-1]])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, self.data)

        # init_state = tf.get_variable('init_state', [1, self._num_hidden[-1]],
        #                              initializer=tf.constant_initializer(0.0))
        # init_state = tf.tile(init_state, [batch_size, 1])
        output, _ = nn.dynamic_rnn(
            rnn.MultiRNNCell(cells, state_is_tuple=True),
            rnn_inputs,
            dtype=tf.float32,
            # initial_state=init_state,
            sequence_length=self.sequence_lengths,
        )
        # last = tf.gather_nd(output, tf.stack([tf.range(128), self.length-1], axis=1))
        last = self._last_relevant(output, self.sequence_lengths)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden[-1], int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        with tf.name_scope("cost"):
            target_f = tf.cast(self.target, tf.float32)
            pred_f = tf.cast(self.prediction, tf.float32)
            cross_entropy = -tf.reduce_sum(target_f * tf.log(pred_f))
        return cross_entropy #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.prediction))

    @lazy_property
    def optimize(self):
        if 'learning_rate' in self.params:
            learning_rate = self.params['learning_rate']
        else:
            learning_rate = 0.001
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        with tf.variable_scope("rnnlm"):
            weight = tf.get_variable("softmax_w", shape=(in_size, out_size), initializer=tf.truncated_normal_initializer(stddev=0.01)) #, stddev=0.01)
            bias = tf.get_variable("softmax_b", shape=(out_size), initializer=tf.zeros_initializer)
        return weight, bias

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


if __name__ == '__main__':
    # hyper params
    config_filename = sys.argv[1]
    with open(config_filename, 'r') as f:
        hyperparams = yaml.load(f)
        print ("Loaded config file from %s" % config_filename)

    batch_size = hyperparams['batch_size']
    n_epoch = hyperparams['n_epoch']
    initialize_from_saved = hyperparams['initialize_from_saved']
    save_dir = hyperparams['save_dir']

    # create save dir if doesn't yet exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_every = hyperparams['save_every']

    data_loader = tweetsloader.TweetsDataLoader('/home/ganenjij/repositories/twitter-sentiment-analysis/data/oldBrexitSentimentComparison.csv', batch_size)
    n_steps, n_inputs, n_classes = data_loader.sequence_length, 1, 2

    model = VariableSequenceClassification(len(data_loader.vocab), hyperparams)

    # validation data
    valid_data, valid_target = data_loader.get_validation_data()

    # get checkpoint state
    if initialize_from_saved:
        ckpt = tf.train.get_checkpoint_state(save_dir)

    with biwigpuprep.GPUSession() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        if initialize_from_saved and ckpt is not None:
            print ("Loading from previously saved state: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        start_global = time.time()
        for epoch in range(n_epoch):
            start_epoch = time.time()
            data_loader.reset_pointer()
            for i in range(data_loader.num_batches):
                train_x, train_y, train_seqlength = data_loader.get_next_training_batch()
                sess.run(model.optimize, {model.data: train_x, model.target: train_y, model.sequence_lengths: train_seqlength})
                # length = sess.run(model.length, {model.data: train_x})
                if (epoch * data_loader.num_batches + i) % save_every == 0 or \
                        (epoch == n_epoch - 1 and i == data_loader.num_batches - 1):
                    iter = epoch * data_loader.num_batches + i
                    chkpt_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, chkpt_path, global_step=iter)
                    print ("Iteration no. {}, Model saved to {}".format(iter, chkpt_path))

            error = sess.run(model.error, {model.data: valid_data, model.target: valid_target})
            print('Epoch {:2d} error {:3.1f}% -- time taken: {:3.3f}'.format(epoch + 1, 100 * error, time.time() - start_epoch))

    print("Total time taken: %s" % (time.time() - start_global))
