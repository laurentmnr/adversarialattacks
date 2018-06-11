import mnist
from model import *
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from networks import *
from model_adv import *
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.model import Model,CallableModelWrapper
from foolbox.models import TensorFlowModel

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
            epoch=20,
            learning_rate=0.002,
            beta=.5,
            model_name='CNN0000',
            checkpoint_dir="checkpoint")


cnn.train(train_images, train_labels)
#
# grad = tf.gradients(cnn.global_loss, cnn.inputs)[0]
# a=grad.eval({
#     cnn.inputs: train_images[:2].reshape(tuple([-1]+cnn.input_shape)),
#     cnn.labels: train_labels[:2],
#     cnn.mode: 'TEST'
# })
#
# #y_pred = np.argmax(dcgan.predict(test_images), axis=1)
# model=CallableModelWrapper(cnn.predict, output_layer='logits')
#
#
#
# fgm=FastGradientMethod(model,sess=sess)
#
# fgm_params = {'eps': 0.3,
#                'clip_min': 0.,
#                'clip_max': 1.}
#
# adv_x = fgm.generate((train_images[:2]), **fgm_params)
# preds_adv = model.get_probs(adv_x)



model = TensorFlowModel(cnn.inputs,cnn.network,bounds=(0, 255))

from foolbox.criteria import TargetClassProbability

target_class = 9
criterion = TargetClassProbability(target_class, p=0.99)


from foolbox.attacks import FGSM


attack=FGSM(model)
image = train_images[0].reshape((28, 28, 1))
label = np.argmax(model.predictions(image))

adversarial = attack(image,label=label,epsilons=1,max_epsilon=0.03*255)


import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.imshow(image.reshape((28, 28)), cmap='gray',vmin=0, vmax=255)
plt.gca().set_title(label)

plt.subplot(1, 3, 2)
plt.imshow(adversarial.reshape((28, 28)), cmap='gray',vmin=0, vmax=255)
plt.gca().set_title(np.argmax(model.predictions(adversarial)))

plt.subplot(1, 3, 3)
plt.imshow((adversarial - image).reshape((28, 28)), cmap='gray',vmin=0, vmax=255)


embed=tf.get_default_graph().get_tensor_by_name("embedding/Relu:0")

pp=[-1]+ cnn.input_shape+[1]
gradient_embedding = tf.concat([tf.reshape(
                           tf.gradients(embed[:,i], cnn.inputs)[0],pp) for i in range(embed.shape[1])],axis=4)
loss_2 = networks.representer_grad_loss(gradient_embedding)
loss_2.eval({
    cnn.inputs: train_images[:100].reshape(tuple([-1]+ cnn.input_shape)),
    cnn.labels: train_labels[:100],
    cnn.mode: "TEST"
})