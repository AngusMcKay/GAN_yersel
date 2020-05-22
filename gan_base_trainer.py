'''
script to train base models which can be saved for uploading to usergan used in app
'''

import os
os.chdir('/home/angus/projects/picture_gen')

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from image_helper import ImageHelper
from gan import GAN, DCGAN

(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()

generative_advarsial_network = GAN(X_train[0].shape, 100, image_helper)
generative_advarsial_network.train(30000, X_train, batch_size=32)

dc_generative_advarsial_network = DCGAN(X_train[0].shape, 100, image_helper, 1)
dc_generative_advarsial_network.train(3, X_train, batch_size=32)

# save models for later use
dc_generative_advarsial_network.discriminator_model.save("models/dcgan_discriminator.h5")
dc_generative_advarsial_network.generator_model.save("models/dcgan_generator.h5")

# start training again with pre-loaded models
from tensorflow.keras.models import load_model
input_discriminator = load_model("models/dcgan_discriminator.h5")
input_generator = load_model("models/dcgan_generator.h5")

dc_generative_advarsial_network = DCGAN(X_train[0].shape, 100, image_helper, 1,
                                         pre_load_discriminator=input_discriminator,
                                         pre_load_generator=input_generator)
dc_generative_advarsial_network.train(3, X_train, batch_size=32)

