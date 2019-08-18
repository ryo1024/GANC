from GAN_model import *
from PIL import Image
import os

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator = Generator()
discriminator = Discriminator()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#generate random test dataset
for _ in range(10):
  x = np.random.sample(size=(4, 4, 3, 64))
  x.astype('float32')
  x = tf.cast(x, tf.float32)
  y = np.random.sample(size=(4, 4, 3, 64))
  y.astype('float32')
  y = tf.cast(y, tf.float32)
  test_lst.append((x, y))
test_dataset = test_lst

#generator test
loss_gen = 0
loss_disc = 0
n = 0
for pair in test_dataset:
  tar = pair[0]
  input_img = pair[1]
  gen_output = generator(input_img, tranning=False)
  im = Image.fromarray(gen_output)
  im.save("/test_generated_img/test_no.{}.jpeg".format(n))

  disc_real_output = discriminator([input_image, target], training=False)
  disc_generated_output = discriminator([input_image, gen_output], training=False)

  loss_gen += numpy.sum(abs(tar - gen_output))
  loss_disc += disc_real_output + (1 - disc_generated_output)
  n = n + 1
print('The loss of generator is: {}'.format(loss_gen / n))
print('The accuracy of discriminator is: {}'.format(loss_disc / (n * 2)))
