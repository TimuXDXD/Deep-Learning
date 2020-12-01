from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy import vstack
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint
from matplotlib import pyplot
import tensorflow as tf

def load_samples():
    (trainX, _), (_, _) = load_data()
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X
# show datasets
def show_datasets():
    print('Train: ', trainX.shape, trainY.shape)
    print('Test: ', testX.shape, testY.shape)
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.axis('off')
        pyplot.imshow(trainX[i],cmap='gray_r')
    pyplot.show()

def generate_real_samples(dataset, n_samples):
    # shuffled prior to each epoch
    i = randint(0, dataset.shape[0], n_samples)
    x = dataset[i]
    print(x)
    y = ones((n_samples, 1))
    return x, y

def generate_fake_samples_without_latent_space(n_samples):
    x = rand(28 * 28 * n_samples)
    x = x.reshape((n_samples, 28, 28, 1))
    y = zeros((n_samples, 1))
    return x, y

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    x = model.predict(x_input)
    y = zeros((n_samples, 1))
    return x, y

def define_discriminator(input_shape=(28,28,1)):
    model = Sequential(name='discriminator')
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_discriminator(model, dataset, n_iter=100, n_batch=256):
    half_batch = int(n_batch/2)
    for i in range(n_iter):
        x_real, y_real = generate_real_samples(dataset, half_batch)
        _, real_acc = model.train_on_batch(x_real, y_real)
        x_fake, y_fake = generate_fake_samples_without_latent_space(half_batch)
        _, fake_acc = model.train_on_batch(x_fake, y_fake)
        print('>%d real=%.0f%% fake=%.0f%%' %(i+1, real_acc*100, fake_acc*100))

def define_generator(latent_dim):
    model = Sequential(name='generator')
    # foundation for 7*7 images
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14*14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28*28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False

    model = Sequential(name='GAN')
    model.add(g_model)
    model.add(d_model)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def train_gan(gan_model, latent_dim, n_epochs=100, n_batch=256):
    for i in range(n_epochs):
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)

def train(d_model, g_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(x, y)
            x_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            print('>%d/%d %d/%d d_loss=%.3f g_loss=%.3f' % (i+1, n_epochs, j+1, bat_per_epo, d_loss, g_loss))
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    x_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    save_plot(x_fake, epoch)
    filename = 'generate_model_e%03d.h5' % (epoch + 1)
    g_model.save(filename)

def save_plot(examples, epoch, n=10):
    for i in range(n*n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis('off')
        pyplot.imshow(examples[i,:,:,0], cmap='gray_r')
    filename = 'generate_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()

def build_and_summary_models(latent_dim):

    d_model = define_discriminator()
    d_model.summary()
    plot_model(d_model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

    g_model = define_generator(latent_dim)
    g_model.summary()
    plot_model(g_model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

    gan_model = define_gan(g_model, d_model)
    gan_model.summary()
    plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

    return d_model, g_model, gan_model

def generate_by_existed_model():
    model = load_model('generator_model_e100.h5')
    latent_points = generate_latent_points(100,25)
    x = model.predict(latent_points)
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.axis('off')
        pyplot.imshow(x[i,:,:,0], cmap='gray_r')
    pyplot.show()

if __name__ == '__main__':
    dataset = load_samples()

    d_model, g_model, gan_model = build_and_summary_models(latent_dim=100)

    # train(d_model, g_model, gan_model, dataset, latent_dim=100)
