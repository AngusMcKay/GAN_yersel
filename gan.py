from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import imageio

# Keras modules
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop



class GAN():
    def __init__(self, image_shape, generator_input_dim, image_helper):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_helper
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim

        # Build models
        self._build_generator_model()
        self._build_and_compile_discriminator_model(optimizer)
        self._build_and_compile_gan(optimizer)

    def train(self, epochs, train_data, batch_size, output_directory="generated/"):
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            genenerated = self._predict_noise(batch_size)
            loss_real = self.discriminator_model.train_on_batch(batch, real)
            loss_fake = self.discriminator_model.train_on_batch(genenerated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            #  Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))
            generator_loss = self.gan.train_on_batch(noise, real)

            # Plot the progress
            print ("---------------------------------------------------------")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(discriminator_loss[0]))
            print ("Generator loss: {}".format(generator_loss))
            print ("---------------------------------------------------------")
            
            history.append({"D":discriminator_loss[0],"G":generator_loss})
            
            # Save images from every hundereth epoch generated images
            if epoch % 100 == 0:
                self._save_images(epoch, output_directory)
                
        self._plot_loss(history)
        self._image_helper.makegif(output_directory)        
    
    def _build_generator_model(self):
        generator_input = Input(shape=(self.generator_input_dim,))
        generator_seqence = Sequential(
                [Dense(256, input_dim=self.generator_input_dim),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(512),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(1024),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(np.prod(self.img_shape), activation='tanh'),
                 Reshape(self.img_shape)])
    
        generator_output_tensor = generator_seqence(generator_input)       
        self.generator_model = Model(generator_input, generator_output_tensor)
        
    def _build_and_compile_discriminator_model(self, optimizer):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_sequence = Sequential(
                [Flatten(input_shape=self.img_shape),
                 Dense(512),
                 LeakyReLU(alpha=0.2),
                 Dense(256),
                 LeakyReLU(alpha=0.2),
                 Dense(1, activation='sigmoid')])
    
        discriminator_tensor = discriminator_sequence(discriminator_input)
        self.discriminator_model = Model(discriminator_input, discriminator_tensor)
        self.discriminator_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator_model.trainable = False
    
    def _build_and_compile_gan(self, optimizer):
        real_input = Input(shape=(self.generator_input_dim,))
        generator_output = self.generator_model(real_input)
        discriminator_output = self.discriminator_model(generator_output)        
        
        self.gan = Model(real_input, discriminator_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _save_images(self, epoch, output_directory):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, output_directory)
    
    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        return self.generator_model.predict(noise)
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()





class DCGAN():
    def __init__(self, image_shape, generator_input_dim, image_hepler, img_channels,
                 pre_load_discriminator=None, pre_load_generator=None):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim
        self.channels = img_channels

        # Build models
        if pre_load_generator==None:
            self._build_generator_model()
        else:
            self.generator_model = pre_load_generator
        
        if pre_load_discriminator==None:
            self._build_and_compile_discriminator_model(optimizer)
        else:
            self.discriminator_model = pre_load_discriminator
            self.discriminator_model.trainable = True # in case still set to False from previous gan build
            self.discriminator_model.compile(loss='binary_crossentropy',
                                             optimizer=optimizer,
                                             metrics=['accuracy'])
        
        self._build_and_compile_gan(optimizer)
    
    def train(self, epochs, train_data, batch_size, output_directory="generated_dcgan"):
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        batches_per_epoch = train_data.shape[0]/batch_size
        iters = int(epochs*batches_per_epoch)
        epoch = 0
        for i in range(iters):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            genenerated = self._predict_noise(batch_size)
            loss_real = self.discriminator_model.train_on_batch(batch, real)
            loss_fake = self.discriminator_model.train_on_batch(genenerated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            #  Train Generator (note that training gan also updates self.generator_model)
            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))
            generator_loss = self.gan.train_on_batch(noise, real)

            # Plot the progress
            print ("---------------------------------------------------------")
            print ("******************Batch {}***************************".format(i))
            print ("Discriminator loss: {}".format(discriminator_loss[0]))
            print ("Generator loss: {}".format(generator_loss))
            print ("---------------------------------------------------------")
            
            history.append({"D":discriminator_loss[0],"G":generator_loss})
            
            # Save images from every hundereth epoch generated images
            epoch = int(i/batches_per_epoch)
            if i==0:
                self._save_images(epoch, output_directory)
            if epoch != int((i-1)/batches_per_epoch):
                self._save_images(epoch, output_directory)
            
        self._plot_loss(history)
        self._image_helper.makegif(output_directory)        
    
    def _build_generator_model(self):
        generator_input = Input(shape=(self.generator_input_dim,))
        generator_seqence = Sequential(
                [Dense(128 * 7 * 7, activation="relu", input_dim=self.generator_input_dim), # sets up 128 7x7 feature maps as input to first conv. layer
                 Reshape((7, 7, 128)),
                 UpSampling2D(), # increases rows and columns by (default) 2x, so now 14x14
                 Conv2D(128, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8), # momentum here refers to moving mean and variance
                 Activation("relu"),
                 UpSampling2D(), # upsamples to 28x28 which needs to be same as image_shape which goes into discriminator
                 Conv2D(64, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 Conv2D(self.channels, kernel_size=3, padding="same"),
                 Activation("tanh")])
    
        generator_output_tensor = generator_seqence(generator_input)       
        self.generator_model = Model(generator_input, generator_output_tensor)
        
    def _build_and_compile_discriminator_model(self, optimizer):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_sequence = Sequential(
                [Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(64, kernel_size=3, strides=2, padding="same"),
                 ZeroPadding2D(padding=((0,1),(0,1))),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(128, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(256, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Flatten(),
                 Dense(1, activation='sigmoid')])
    
        discriminator_tensor = discriminator_sequence(discriminator_input)
        self.discriminator_model = Model(discriminator_input, discriminator_tensor)
        self.discriminator_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    
    def _build_and_compile_gan(self, optimizer):
        real_input = Input(shape=(self.generator_input_dim,))
        generator_output = self.generator_model(real_input)
        # make discriminator model un-trainable before adding it to model (note this does not apply to discriminator model which is already compiled)
        self.discriminator_model.trainable = False
        discriminator_output = self.discriminator_model(generator_output)        
        
        # compile combined gan
        self.gan = Model(real_input, discriminator_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _save_images(self, epoch, output_directory):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, output_directory)
    
    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        return self.generator_model.predict(noise)
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()





class USERGAN():
    def __init__(self, image_shape, generator_input_dim, image_hepler, img_channels,
                 pre_load_discriminator, pre_load_generator):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim
        self.channels = img_channels

        # Build models
        self.generator_model = pre_load_generator
        
        self.discriminator_model = pre_load_discriminator
        self.discriminator_model.trainable = True # in case still set to False from previous gan build
        self.discriminator_model.compile(loss='binary_crossentropy',
                                         optimizer=optimizer,
                                         metrics=['accuracy'])
        
        self._build_and_compile_gan(optimizer)
    
    def generate_images(self, size, epoch, display_directory, permanent_directory):
        # generate images to display to users, and also store in permanent directory for later collection
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        generated = self.generator_model.predict(noise)
        generated = 0.5 * generated + 0.5
        self._image_helper.send_to_ui(generated, 0, display_directory)
        self._image_helper.send_to_ui(generated, epoch, permanent_directory)
    
    def process_images(self, input_directory, user_ratings_dict):
        # generate image training set from user responses
        image_paths = glob.glob(input_directory+"/*.png")
        images_for_retraining = []
        labels_for_retraining = []
        for path in image_paths:
            # get rating for image
            for key in user_ratings_dict:
                if key in path:
                    image_rating = user_ratings_dict[key]
            # first copy original image
            image = imageio.imread(path)
            images_for_retraining.append(image)
            labels_for_retraining.append(image_rating)
            
            # delete user_rating to throw exceptions if can't find for particular image
            del(image_rating)
    
    def retrain(self, train_data, user_ratings, augmentations_per_image, batch_size):
        
        #real = np.ones((batch_size, 1))
        #fake = np.zeros((batch_size, 1))
        #history = []
        total_images = train_data.shape[0]*augmentations_per_image
        iters = int(total_images/batch_size)
        for i in range(iters):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            genenerated = self._predict_noise(batch_size)
            loss_real = self.discriminator_model.train_on_batch(batch, real)
            loss_fake = self.discriminator_model.train_on_batch(genenerated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            #  Train Generator (note that training gan also updates self.generator_model)
            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))
            generator_loss = self.gan.train_on_batch(noise, real)

            # Plot the progress
            print ("---------------------------------------------------------")
            print ("******************Batch {}***************************".format(i))
            print ("Discriminator loss: {}".format(discriminator_loss[0]))
            print ("Generator loss: {}".format(generator_loss))
            print ("---------------------------------------------------------")
            
            history.append({"D":discriminator_loss[0],"G":generator_loss})
            
            # Save images from every hundereth epoch generated images
            epoch = int(i/batches_per_epoch)
            if i==0:
                self._save_images(epoch, output_directory)
            if epoch != int((i-1)/batches_per_epoch):
                self._save_images(epoch, output_directory)
            
        self._plot_loss(history)
        self._image_helper.makegif(output_directory)        
    
    def _build_and_compile_gan(self, optimizer):
        real_input = Input(shape=(self.generator_input_dim,))
        generator_output = self.generator_model(real_input)
        # make discriminator model un-trainable before adding it to model (note this does not apply to discriminator model which is already compiled)
        self.discriminator_model.trainable = False
        discriminator_output = self.discriminator_model(generator_output)        
        
        # compile combined gan
        self.gan = Model(real_input, discriminator_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _save_images(self, epoch, output_directory):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, output_directory)
    
    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        return self.generator_model.predict(noise)
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()





#np.random.normal(0, 1, (1, 100))
'''
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
 
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
 
# define the standalone critic model
def define_critic(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()
	# downsample to 14x14
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# scoring, linear activation
	model.add(Flatten())
	model.add(Dense(1))
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
 
# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# output 28x28x1
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model
 
# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	critic.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
 
# load images
def load_real_samples():
	# load dataset
	(trainX, trainy), (_, _) = load_data()
	# select all of the examples for a given class
	selected_ix = trainy == 7
	X = trainX[selected_ix]
	# expand to 3d, e.g. add channels
	X = np.expand_dims(X, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -np.ones((n_samples, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = np.ones((n_samples, 1))
	return X, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		plt.subplot(10, 10, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
 
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	plt.plot(d1_hist, label='crit_real')
	plt.plot(d2_hist, label='crit_fake')
	plt.plot(g_hist, label='gen')
	plt.legend()
	plt.savefig('plot_line_plot_loss.png')
	plt.close()
 
# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(np.mean(c1_tmp))
		c2_hist.append(np.mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -np.ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)
 
# size of the latent space
latent_dim = 50
# create the critic
critic = define_critic()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
# load image data
dataset = load_real_samples()
print(dataset.shape)
# train model
train(generator, critic, gan_model, dataset, latent_dim)
'''




