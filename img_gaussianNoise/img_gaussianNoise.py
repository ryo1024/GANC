import imageio
import imgaug as ia
image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
from imgaug import augmenters as iaa
ia.seed(6)
gaus = iaa.AdditiveGaussianNoise(scale=(10,80))
images_aug = gaus.augment_image(image)
ia.imshow(images_aug)