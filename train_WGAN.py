import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from IPython import display
import tensorflow as tf
from losses import discriminator_wgan_loss, generator_wgan_loss
from models import make_generator_model, make_discriminator_model
from params import params


class Trainer():
    def __init__(self, params):
        self.buffer_size = params['buffer_size']
        self.batch_size = params['bacth_size']
        self.noise_dim = params['noise_dim']
        self.num_examples_to_generate = params['num_exmaple_to_generate']
        self.disc_iterations = params['disc_iterations']
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.save_name = params['save_name']
        self.data_path = params['data_path']

    def preprocess_and_build(self):
        data = np.load(self.data_path)
        X = data['X']
        X = (X - 0.5) / 0.5
        print(np.max(X), np.min(X))
        nsub = len(X)
        if len(X.shape) == 3:
            X = np.expand_dims(X, 3)

        target_shape = [256, 256]
        ratio = [ts / Xs for ts, Xs in zip(target_shape, X.shape[1:3])]
        X2 = np.zeros((nsub, target_shape[0], target_shape[1], 1), dtype='float32')
        for j in range(nsub):
            X2[j, :, :, 0] = zoom(X[j, :, :, 0], ratio)
        X = X2

        train_images = X
        self.image_size_X = X.shape[1]
        self.image_size_Y = X.shape[2]

        save_dir = self.save_name
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(self.batch_size)

        # Models
        self.generator = make_generator_model([self.image_size_X, self.image_size_Y])
        self.generator.summary()

        self.discriminator = make_discriminator_model([self.image_size_X, self.image_size_Y])
        self.discriminator.summary()


        self.generator_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # Losses tracker
        self.metric_dict = dict()
        self.metric_dict['gen_loss'] = []
        self.metric_dict['dict_loss'] = []

        # Checkpoint for training restoring in case needed
        self.checkpoint_dir = self.save_name
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))



    @tf.function
    def train_step(self, images):
        for i in range(self.disc_iterations):
            noise = tf.random.normal([images.shape[0], self.noise_dim])
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_batch = images
                fake_batch = generated_images

                disc_loss = discriminator_wgan_loss(self.discriminator, real_batch, fake_batch, batch_size=self.batch_size)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([images.shape[0], self.noise_dim])
            generated_images = self.generator(noise, training=True)
            fake_batch = generated_images
            gen_loss = generator_wgan_loss(self.discriminator, fake_batch)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            gen_loss_epoch = []
            disc_loss_epoch = []
            for image_batch in self.dataset:
                gen_loss, disc_loss = self.train_step(image_batch)
                # print(gen_loss.numpy(), disc_loss.numpy())
                gen_loss_epoch.append(gen_loss.numpy())
                disc_loss_epoch.append(disc_loss.numpy())

            self.metric_dict['gen_loss'].append(float(np.mean(gen_loss_epoch)))
            self.metric_dict['dict_loss'].append(float(np.mean(disc_loss_epoch)))
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator,
                                     epoch + 1,
                                     self.seed)

            # Save the model every 200 epochs
            if (epoch + 1) % 200 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metric_dict, f)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator,
                                 self.epochs,
                                 self.seed)


    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(15, 15))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # plt.show()
        if epoch % 50 == 0:
            plt.savefig(os.path.join(self.save_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))

        plt.close(fig)


# RUN
from params import params
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=''' 

    WGAN training

    ''')

args = parser.parse_args()

tr = Trainer(params)
tr.train()
