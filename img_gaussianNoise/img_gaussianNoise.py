import imageio
import imgaug as ia

from imgaug import augmenters as iaa
import sys



if __name__ == "__main__":
    print(sys.argv[1])
    image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
    ia.seed(6)
    gaus = iaa.AdditiveGaussianNoise(scale=(10,80))
    images_aug = gaus.augment_image(image)
    ia.imshow(images_aug)