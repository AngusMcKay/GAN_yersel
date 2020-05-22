'''
testing classes, methods and operations
'''

import os
os.chdir('/home/angus/projects/picture_gen')

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from image_helper import ImageHelper
from gan import GAN, DCGAN, USERGAN

(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()

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

'''
testing usergan
'''
# pre load models
from tensorflow.keras.models import load_model
input_discriminator = load_model("models/dcgan_discriminator.h5")
input_generator = load_model("models/dcgan_generator.h5")

user_gan = USERGAN(image_shape=(28,28,1), generator_input_dim=100,
                   image_helper=image_helper, img_channels=1,
                   pre_load_discriminator=input_discriminator,
                   pre_load_generator=input_generator)

# image generator
user_gan.generate_images(size=10, epoch=1,
                         display_directory='testing/display_directory',
                         permanent_directory='testing/permanent_directory')
user_gan.generator_model._make_predict_function(np.random.normal(0, 1, (2, 100)))
# image processing
ratings_dict = {'0-0.png': 0.1, '0-1.png': 0.1, '0-2.png': 0.1, '0-3.png': 0.1, '0-4.png': 0.8,
                '0-5.png': 0.8, '0-6.png': 0.1, '0-7.png': 0.1, '0-8.png': 0.8, '0-9.png': 0.1}
images_for_training, labels_for_training = user_gan.process_images(
        input_directory='testing/display_directory',
        user_ratings_dict=ratings_dict
        )

# retraining
user_gan.retrain(images_for_training, labels_for_training,
                 augmentations_per_image=32*10, batch_size=32)


