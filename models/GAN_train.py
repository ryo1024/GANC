from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from models.GAN_model import *

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
EPOCHS = 150

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

generator = Generator()
discriminator = Discriminator()


#################################################################################################################
# loss function and optimizer
##################################################################################################################



###############################################################################################################
# training
################################################################################################################
def generate_images(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


# define two global variables
# build the generator


def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for input_image, target in dataset:
            train_step(input_image, target)

        # clear_output(wait=True)
        # for inp, tar in test_dataset.take(1):
        #   generate_images(generator, inp, tar)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))


def main():
    # input pipeline: tarning dataset and test dataset
    train_lst = []
    for _ in range(10):
        x = np.random.sample(size=(1, 256, 256, 3))
        x.astype(np.float32)
        x = tf.cast(x, tf.float32)
        y = np.random.sample(size=(1, 256, 256, 3))
        y.astype(np.float32)
        y = tf.cast(y, tf.float32)
        train_lst.append((x, y))
    train_dataset = train_lst
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    # train_dataset = train_dataset.map(load_image_train,
    #                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.batch(1)

    test_lst = []
    for _ in range(10):
        x = np.random.sample(size=(4, 4, 3, 64))
        x.astype('float32')
        x = tf.cast(x, tf.float32)
        y = np.random.sample(size=(4, 4, 3, 64))
        y.astype('float32')
        y = tf.cast(y, tf.float32)
        test_lst.append((x, y))
    test_dataset = test_lst
    # shuffling so that for every epoch a different image is generated
    # to predict and display the progress of our model.
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    # test_dataset = test_dataset.map(load_image_test)
    # test_dataset = test_dataset.batch(1)

    # loss function

    # establish checkpoint

    train(train_dataset, EPOCHS)


if __name__ == "__main__":
    main()
