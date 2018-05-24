from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import time as time
import numpy as np
import tensorflow as tf
import networks
import os


class CNN(object):
    def __init__(self, sess,
                 input_shape=(28, 28, 1),
                 y_dim=10,
                 batch_size=64,
                 learning_rate=0.002,
                 epoch=25,
                 beta=.5,
                 model_name='CNN',
                 checkpoint_dir="checkpoint"):

        self.sess = sess

        self.lambda_loss = 0.8

        # Training params
        self.batch_size = batch_size
        self.epoch = epoch

        # optim params
        self.learning_rate = learning_rate
        self.beta = beta

        # DATA
        self.input_shape = list(input_shape)
        self.num_labels = y_dim

        self.model_name = model_name
        self.checkpoint_dir = os.path.join(checkpoint_dir,model_name)


        self.build_model()
        self.saver = tf.train.Saver()

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta) \
            .minimize(self.global_loss, var_list=self.vars)

        files = os.listdir("./logs_train")
        for file in files:
            os.remove("./logs_train" + '/' + file)

        files = os.listdir("./logs_val")
        for file in files:
            os.remove("./logs_val" + '/' + file)

        self.writer_train = tf.summary.FileWriter("./logs_train", self.sess.graph)
        self.writer_val = tf.summary.FileWriter("./logs_val", self.sess.graph)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            self.counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            self.counter = 0
            print(" [!] Load failed...")

    def build_model(self):
        self.mode=tf.placeholder(tf.string, name='mode')

        self.inputs = tf.placeholder(
            tf.float32, [None] + self.input_shape, name='inputs')
        inputs = self.inputs

        self.labels = tf.placeholder(
            tf.int64, [None, self.num_labels], name='labels')
        labels = self.labels

        self.network,self.grad_rep = networks.network_mnist(inputs, self.input_shape, self.num_labels, self.mode)

        self.network_sum = tf.summary.histogram("cnn", self.network)

        self.loss_1 = networks.cross_entropy_loss(self.network, labels)
        self.loss_2=networks.representer_grad_loss(self.grad_rep)

        self.loss_1_sum = tf.summary.scalar("cross_entropy_loss", self.loss_1)
        self.loss_2_sum = tf.summary.scalar("representer_grad_loss", self.loss_2)

        self.global_loss = self.loss_1+self.lambda_loss*self.loss_2
        self.loss_sum = tf.summary.scalar("global_loss", self.global_loss)

        self.vars = tf.trainable_variables()

        self.saver = tf.train.Saver()

        self.acc = networks.accuracy(self.network, self.labels)
        self.acc_sum = tf.summary.scalar("accuracy", self.acc)

        self.summary = tf.summary.merge_all()

    def train(self, X, y,cv=0.05):

        self.train_size = np.shape(X)[0]
        tr_set = np.random.choice(np.arange(self.train_size), int(self.train_size*(1-cv)), replace=False)
        Xtr = X[tr_set]
        ytr = y[tr_set]
        Xval = np.delete(X, tr_set, axis=0)
        yval = np.delete(y, tr_set, axis=0)
        self.train_size = np.shape(Xtr)[0]
        counter = self.counter
        start_time = time.time()


        batch_idxs = self.train_size // self.batch_size
        i=0
        for epoch in range(self.counter, self.epoch):

            for idx in range(0, batch_idxs):
                batch_images = (Xtr[idx * self.batch_size:(idx + 1) * self.batch_size]).reshape(
                    tuple([self.batch_size]+self.input_shape))
                batch_labels = (ytr[idx * self.batch_size:(idx + 1) * self.batch_size])

                # Update network
                _, summary_str = self.sess.run([self.optimizer, self.summary],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.labels: batch_labels,
                                                   self.mode: "TRAIN"})
                self.writer_train.add_summary(summary_str, i)

                err_1 = self.loss_1.eval({
                    self.inputs: batch_images,
                    self.labels: batch_labels,
                    self.mode: "TRAIN"
                })
                err_2 = self.loss_2.eval({
                    self.inputs: batch_images,
                    self.labels: batch_labels,
                    self.mode: "TRAIN"
                })
                if np.mod(idx, 10) == 0:

                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f,loss_1: %.8f, loss_2: %.8f" \
                          % (epoch, self.epoch, idx, batch_idxs,
                             (time.time() - start_time), err_1, err_2))

                    summary_str_val = self.sess.run(self.summary,
                                                   feed_dict={
                                                       self.inputs: Xval.reshape(tuple([-1] + self.input_shape)),
                                                       self.labels: yval,
                                                       self.mode: "TEST"})

                    self.writer_val.add_summary(summary_str_val, i)

                i += 1

            self.save(counter)
            counter += 1

    def predict(self, X):
        return self.sess.run(self.network, feed_dict={
                                self.inputs: X.reshape(tuple([-1]+self.input_shape)),
                                self.mode: "TEST"
                            })

    def save(self,step):
        checkpoint_dir = self.checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir,self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)  # , self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
