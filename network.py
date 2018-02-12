'''
By adidinchuk park. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-linear-regression
'''

import tensorflow as tf
import numpy as np
import utils
import hyperparams as hp


class Network:

    # Available loss_functions : l1, l2, pseudo_huber
    def __init__(self, reg_type='tf', loss_function='l2'):
        # init
        self.type = reg_type
        self.session = tf.Session()

        # placeholders
        self.inputs = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.outputs = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        
        # variables
        self.A = tf.Variable(tf.random_normal(shape=[2, 1]))
        self.b = tf.Variable(tf.random_normal(shape=[1, 1]))

        # model & loss
        self.model_output = tf.add(tf.matmul(self.inputs, self.A), self.b)
        self.loss = None
        self.set_loss(loss_function)

        # init variables
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

        # set optimization method (Gradient Descent)
        self.optimization = None
        self.training_step = None
        self.regularization = None

    def train(self, inputs, outputs, 
              lr=0.01, batch_size=50,
              epochs=100, validation_size=0.4,
              plot=False):

        # reset graph
        # ops.reset_default_graph()

        # set optimization method (Gradient Descent)
        self.optimization = tf.train.GradientDescentOptimizer(lr)
        self.training_step = self.optimization.minimize(self.loss)

        # separate into train and test
        train_size = int(validation_size * len(inputs))
        train_index = np.random.choice(len(inputs), size=train_size)
        test_index = np.array(list(set(range(len(inputs))) - set(train_index)))

        # train
        train_inputs = inputs[train_index]
        train_outputs = outputs[train_index]

        # test
        test_inputs = inputs[test_index]
        test_outputs = outputs[test_index]

        if train_size < batch_size:
            raise Exception("Not enough data to accommodate batch size.")

        # performance tracking
        train_loss_result = []
        test_loss_result = []

        for i in range(epochs):
            batch_sample = np.random.choice(len(train_inputs), size=batch_size)
            self.session.run(self.training_step, feed_dict={self.inputs: train_inputs[batch_sample],
                                                            self.outputs: train_outputs[batch_sample]})

            # if plotting, record every epoch
            if plot:
                train_loss = self.session.run(self.loss,
                                              feed_dict={self.inputs: train_inputs, self.outputs: train_outputs})
                train_loss_result.append(train_loss)
                test_loss = self.session.run(self.loss,
                                             feed_dict={self.inputs: test_inputs, self.outputs: test_outputs})
                test_loss_result.append(test_loss)

            # display result every time 20% of the required epochs are processed
            if (i+1) % (epochs / 5) == 0:

                # if plotting loss is processed above
                if not plot:
                    train_loss = self.session.run(self.loss,
                                                  feed_dict={self.inputs: train_inputs, self.outputs: train_outputs})
                    train_loss_result.append(train_loss)
                    test_loss = self.session.run(self.loss,
                                                 feed_dict={self.inputs: test_inputs, self.outputs: test_outputs})
                    test_loss_result.append(test_loss)

                utils.print_progress(i, epochs, train_loss, test_loss)

        # plot training and testing loss if required
        if plot:
            utils.plot_loss(train_loss_result, test_loss_result)

    # L2 loss function L = a^2
    def l2(self):
        self.loss = tf.reduce_mean(tf.square(self.miss()))

    # L1 loss function L = |a|
    def l1(self):
        self.loss = tf.reduce_mean(tf.abs(self.miss()))

    # Pseudo Huber loss function L = 𝛿^2 * ( √(1 + (a/𝛿)^2) - 1)
    def pseudo_huber(self):
        self.loss = tf.reduce_mean(tf.multiply(tf.square(hp.ph_delta),
                                               tf.sqrt(1 + tf.square(self.miss()) / hp.ph_delta) - 1))


    def miss(self):
        return tf.subtract(self.model_output, self.outputs)

    # Available loss_functions : l1, l2, pseudo_huber
    def set_loss(self, loss_function):
        if loss_function == 'l2':
            self.l2()
        elif loss_function == 'l1':
            self.l1()
        elif loss_function == 'pseudo_huber':
            self.pseudo_huber()
        else:
            self.l2()
