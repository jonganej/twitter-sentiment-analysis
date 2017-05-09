#!/usr/bin/env python3

import functools
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn
import biwigpuprep
import os, time


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

    def __init__(self, data, target, num_hidden=200, num_layers=2):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cells = []
        # create multi layer RNN
        for _ in range(self._num_layers):
            cell = rnn.GRUCell(self._num_hidden)
            cells.append(cell)

        init_state = tf.get_variable('init_state', [1, self._num_hidden],
                                     initializer=tf.constant_initializer(0.0))
        init_state = tf.tile(init_state, [batch_size, 1])
        output, _ = nn.dynamic_rnn(
            rnn.MultiRNNCell(cells, state_is_tuple=True),
            data,
            dtype=tf.float32,
            # initial_state=init_state,
            sequence_length=self.length,
        )
        # last = tf.gather_nd(output, tf.stack([tf.range(128), self.length-1], axis=1))
        keep_prob = tf.constant(0.6)
        output = tf.nn.dropout(output, keep_prob)

        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        with tf.name_scope("cost"):
            cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.prediction))

    @lazy_property
    def optimize(self):
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
    batch_size = 128
    n_epoch = 500
    initialize_from_saved = True
    save_dir = './save'
    save_every = 100
    data_loader = mhadloader.MHADDataLoader('/home/ganenjij/my_scratch/UTD-MHAD-dataset-padded/', batch_size)
    n_steps, n_inputs, n_classes = 125, 60, 27

    data = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    target = tf.placeholder(tf.float32, [None, n_classes])
    model = VariableSequenceClassification(data, target, 128, 2)

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
                train_x, train_y = data_loader.get_next_training_batch()
                sess.run(model.optimize, {data: train_x, target: train_y})
                length = sess.run(model.length, {data: train_x})
                if (epoch * data_loader.num_batches + i) % save_every == 0 or \
                        (epoch == n_epoch - 1 and i == data_loader.num_batches - 1):
                    iter = epoch * data_loader.num_batches + i
                    chkpt_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, chkpt_path, global_step=iter)
                    print ("Iteration no. {}, Model saved to {}".format(iter, chkpt_path))

            error = sess.run(model.error, {data: valid_data, target: valid_target})
            print('Epoch {:2d} error {:3.1f}% -- time taken: {:3.3f}'.format(epoch + 1, 100 * error, time.time() - start_epoch))

    print("Total time taken: %s" % (time.time() - start_global))