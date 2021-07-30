from random import random
from numpy import vstack, zeros, ones, asarray
from numpy.random import randint
from fontsDatasets import load_npy
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Activation
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, LeakyReLU
from keras.initializers import RandomNormal
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
from keras.optimizers import Adam

class CycleGAN():

    def __init__(self):
        self.dataset = self.load_real_samples()
        self.image_shape = self.dataset[0].shape[1:]
        self.n_resnet = 6
        # generator: A -> B
        self.g_model_AtoB = self.define_generator()
        # generator: B -> A
        self.g_model_BtoA = self.define_generator()
        # discriminator: A -> [real/fake]
        self.d_model_A = self.define_discriminator()
        # discriminator: B -> [real/fake]
        self.d_model_B = self.define_discriminator()
        # composite: A -> B -> [real/fake, A]
        self.c_model_AtoB = self.define_composite_model(self.g_model_AtoB, self.d_model_B, self.g_model_BtoA)
        # composite: B -> A -> [real/fake, B]
        self.c_model_BtoA = self.define_composite_model(self.g_model_BtoA, self.d_model_A, self.g_model_AtoB)

    def resnet_block(self, n_filters, input_layer):
        init = RandomNormal(stddev=0.02)

        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Concatenate()([g, input_layer])
        return g

    # define the standalone generator model
    def define_generator(self):
    	# weight initialization
    	init = RandomNormal(stddev=0.02)
    	# image input
    	in_image = Input(shape=self.image_shape)
    	# c7s1-64
    	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    	g = InstanceNormalization(axis=-1)(g)
    	g = Activation('relu')(g)
    	# d128
    	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    	g = InstanceNormalization(axis=-1)(g)
    	g = Activation('relu')(g)
    	# d256
    	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    	g = InstanceNormalization(axis=-1)(g)
    	g = Activation('relu')(g)
    	# # R256
    	# for _ in range(self.n_resnet):
    	# 	g = self.resnet_block(256, g)
    	# u128
    	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    	g = InstanceNormalization(axis=-1)(g)
    	g = Activation('relu')(g)
    	# u64
    	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    	g = InstanceNormalization(axis=-1)(g)
    	g = Activation('relu')(g)
    	# c7s1-3
    	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
    	g = InstanceNormalization(axis=-1)(g)
    	out_image = Activation('tanh')(g)
    	# define model
    	model = Model(in_image, out_image)
    	return model

    # define the discriminator model
    def define_discriminator(self):
    	# weight initialization
    	init = RandomNormal(stddev=0.02)
    	# source image input
    	in_image = Input(shape=self.image_shape)
    	# C64
    	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    	d = LeakyReLU(alpha=0.2)(d)
    	# C128
    	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    	d = InstanceNormalization(axis=-1)(d)
    	d = LeakyReLU(alpha=0.2)(d)
    	# C256
    	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    	d = InstanceNormalization(axis=-1)(d)
    	d = LeakyReLU(alpha=0.2)(d)
    	# C512
    	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    	d = InstanceNormalization(axis=-1)(d)
    	d = LeakyReLU(alpha=0.2)(d)
    	# second last output layer
    	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    	d = InstanceNormalization(axis=-1)(d)
    	d = LeakyReLU(alpha=0.2)(d)
    	# patch output
    	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    	# define model
    	model = Model(in_image, patch_out)
    	# compile model
    	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    	return model

    # define a composite model for updating generators by adversarial and cycle loss
    def define_composite_model(self, g_model_1, d_model, g_model_2):
    	# ensure the model we're updating is trainable
    	g_model_1.trainable = True
    	# mark discriminator as not trainable
    	d_model.trainable = False
    	# mark other generator model as not trainable
    	g_model_2.trainable = False
    	# discriminator element
    	input_gen = Input(shape=self.image_shape)
    	gen1_out = g_model_1(input_gen)
    	output_d = d_model(gen1_out)
    	# identity element
    	input_id = Input(shape=self.image_shape)
    	output_id = g_model_1(input_id)
    	# forward cycle
    	output_f = g_model_2(gen1_out)
    	# backward cycle
    	gen2_out = g_model_2(input_id)
    	output_b = g_model_1(gen2_out)
    	# define model graph
    	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    	# define optimization algorithm configuration
    	opt = Adam(lr=0.0002, beta_1=0.5)
    	# compile model with weighting of least squares loss and L1 loss
    	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    	return model

    def load_real_samples(self):
        (X_train, Y_train), (X_test, Y_test) = load_npy()
        dataX = vstack((X_train,X_test))
        dataY = vstack((Y_train,Y_test))
        dataX = (dataX - 127.5) / 127.5
        dataY = (dataY - 127.5) / 127.5
        return [dataX, dataY]

    # select a batch of random samples, returns images and target
    def generate_real_samples(self, dataset, n_samples, patch_shape):
    	# choose random instances
    	ix = randint(0, dataset.shape[0], n_samples)
    	# retrieve selected images
    	X = dataset[ix]
    	# generate 'real' class labels (1)
    	y = ones((n_samples, patch_shape, patch_shape, 1))
    	return X, y

    # generate a batch of images, returns images and targets
    def generate_fake_samples(self, g_model, dataset, patch_shape):
    	# generate fake instance
    	X = g_model.predict(dataset)
    	# create 'fake' class labels (0)
    	y = zeros((len(X), patch_shape, patch_shape, 1))
    	return X, y

    # save the generator models to file
    def save_models(self, step, g_model_AtoB, g_model_BtoA):
    	# save the first generator model
    	filename1 = './saved_model/g_model_AtoB_%06d.h5' % (step+1)
    	g_model_AtoB.save(filename1)
    	# save the second generator model
    	filename2 = './saved_model/g_model_BtoA_%06d.h5' % (step+1)
    	g_model_BtoA.save(filename2)
    	print('>Saved: %s and %s' % (filename1, filename2))

    # generate samples and save as a plot and save the model
    def summarize_performance(self, step, g_model, trainX, name, n_samples=5):
    	# select a sample of input images
    	X_in, _ = generate_real_samples(trainX, n_samples, 0)
    	# generate translated images
    	X_out, _ = generate_fake_samples(g_model, X_in, 0)
    	# scale all pixels from [-1,1] to [0,1]
    	X_in = (X_in + 1) / 2.0
    	X_out = (X_out + 1) / 2.0
    	# plot real images
    	for i in range(n_samples):
    		plt.subplot(2, n_samples, 1 + i)
    		plt.axis('off')
    		plt.imshow(X_in[i])
    	# plot translated image
    	for i in range(n_samples):
    		plt.subplot(2, n_samples, 1 + n_samples + i)
    		plt.axis('off')
    		plt.imshow(X_out[i])
    	# save plot to file
    	filename1 = './images/DFKai-SB2SimHei/%s_generated_plot_%06d.png' % (name, (step+1))
    	plt.savefig(filename1)
    	plt.close()

    # update image pool for fake images
    def update_image_pool(self, pool, images, max_size=50):
    	selected = list()
    	for image in images:
    		if len(pool) < max_size:
    			# stock the pool
    			pool.append(image)
    			selected.append(image)
    		elif random() < 0.5:
    			# use image, but don't add it to the pool
    			selected.append(image)
    		else:
    			# replace an existing image and use replaced image
    			ix = randint(0, len(pool))
    			selected.append(pool[ix])
    			pool[ix] = image
    	return asarray(selected)

    # train cyclegan models
    def train(self):
    	# define properties of the training run
    	n_epochs, n_batch, = 100, 1
    	# determine the output square shape of the discriminator
    	n_patch = self.d_model_A.output_shape[1]
    	# unpack dataset
    	trainA, trainB = self.dataset
    	# prepare image pool for fakes
    	poolA, poolB = list(), list()
    	# calculate the number of batches per training epoch
    	bat_per_epo = int(len(trainA) / n_batch)
    	# calculate the number of training iterations
    	n_steps = bat_per_epo * n_epochs
    	# manually enumerate epochs
    	for i in range(n_steps):
    		# select a batch of real samples
    		X_realA, y_realA = self.generate_real_samples(trainA, n_batch, n_patch)
    		X_realB, y_realB = self.generate_real_samples(trainB, n_batch, n_patch)
    		# generate a batch of fake samples
    		X_fakeA, y_fakeA = self.generate_fake_samples(self.g_model_BtoA, X_realB, n_patch)
    		X_fakeB, y_fakeB = self.generate_fake_samples(self.g_model_AtoB, X_realA, n_patch)
    		# update fakes from pool
    		X_fakeA = self.update_image_pool(poolA, X_fakeA)
    		X_fakeB = self.update_image_pool(poolB, X_fakeB)
    		# update generator B->A via adversarial and cycle loss
    		g_loss2, _, _, _, _  = self.c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
    		# update discriminator for A -> [real/fake]
    		dA_loss1 = self.d_model_A.train_on_batch(X_realA, y_realA)
    		dA_loss2 = self.d_model_A.train_on_batch(X_fakeA, y_fakeA)
    		# update generator A->B via adversarial and cycle loss
    		g_loss1, _, _, _, _ = self.c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
    		# update discriminator for B -> [real/fake]
    		dB_loss1 = self.d_model_B.train_on_batch(X_realB, y_realB)
    		dB_loss2 = self.d_model_B.train_on_batch(X_fakeB, y_fakeB)
    		# summarize performance
    		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
    		# evaluate the model performance every so often
    		if (i+1) % (bat_per_epo * 1) == 0:
    			# plot A->B translation
    			self.summarize_performance(i, self.g_model_AtoB, trainA, 'AtoB')
    			# plot B->A translation
    			self.summarize_performance(i, self.g_model_BtoA, trainB, 'BtoA')
    		if (i+1) % (bat_per_epo * 5) == 0:
    			# save the models
    			self.save_models(i, self.g_model_AtoB, self.g_model_BtoA)

if __name__ == '__main__':
    model = CycleGAN()
    model.train()
