import os
import imageio
from os import listdir

def crop(img, side):
    """Return cropped square image size of side x side given img numpy"""
    side = side // 2
    y = img.shape[0] // 2
    x = img.shape[1] // 2
    print(y, x)
    print(img.shape)
    return img[y - side:y +side, x - side : x+side]


if __name__ == "__main__":
    dir_path = '../images'
    for filename in listdir(dir_path):
        img = imageio.imread(dir_path + '/' + filename)
        cropped = crop(img, 256)
        imageio.imwrite("../images_cropped/"+ "c_" + filename + ".jpg", cropped)


