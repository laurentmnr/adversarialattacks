from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

def network_mnist(images,input_shape,y_dim,mode,noise=np.zeros(3)):
    # features=images,labels,mode=TEST or TRAIN
    # Input Layer
    input_layer = tf.reshape(images, [-1]+input_shape)
    input_layer+=noise[0]*tf.random_normal(shape=tf.shape(input_layer))

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv1 += noise[1] * tf.random_normal(shape=tf.shape(conv1))
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool1 += noise[2] * tf.random_normal(shape=tf.shape(pool1))
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    #conv2 += noise[3] * tf.random_normal(shape=tf.shape(conv2))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=32, activation=tf.nn.relu, name='embedding')

    if mode == "TRAIN":
        dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    else:
        dropout = tf.layers.dropout(inputs=dense, rate=1)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=y_dim,name='logits')

    #Returns logits and representer
    return logits

def cross_entropy_loss(logits, labels):
    """Cross entropy loss
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
    loss
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def representer_grad_loss(grad_representer):
    """control of the gradient of the representer
    Args:
    representer
    Returns:
    tr(grad(rep).T*grad(rep))
    """
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(grad_representer, grad_representer),axis=[1,2,3,4]))

def accuracy(y_pred,y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

