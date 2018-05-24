import mnist
from model import *
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from networks import *
from model_adv import *
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import Model,CallableModelWrapper


train_images = mnist.train_images()#/255
train_labels = mnist.train_labels()
a = train_labels
b = np.zeros((len(a), 10))
b[np.arange(len(a)), a] = 1
train_labels = b

test_images = mnist.test_images()#/255
test_labels = mnist.test_labels()



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=run_config)

cnn = CNN(sess,
            y_dim=10,
            batch_size=64,
            epoch=5,
            learning_rate=0.002,
            beta=.5,
            model_name='CNN1',
            checkpoint_dir="checkpoint")
cnn.train(train_images, train_labels)

grad = tf.gradients(cnn.global_loss, cnn.inputs)[0]
a=grad.eval({
    cnn.inputs: train_images[:2].reshape(tuple([-1]+cnn.input_shape)),
    cnn.labels: train_labels[:2],
    cnn.mode: 'TEST'
})

#y_pred = np.argmax(dcgan.predict(test_images), axis=1)
model=CallableModelWrapper(cnn.predict, output_layer='logits')



fgm=FastGradientMethod(model,sess=sess)

fgm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}

adv_x = fgm.generate((train_images[:2]), **fgm_params)
preds_adv = model.get_probs(adv_x)