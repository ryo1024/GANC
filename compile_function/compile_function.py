import numpy as np
from PIL import Image
import os

IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images_cropped/'
N_IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/noised_images_cropped/'

def compile_pictures(image_dir, n_image_dir):
    image_files_compiled = []
    integer_list1 = []
    integer_list2 = []
    for image in image_dir:
        image_name = os.path.basename(image)
        img = Image.open(image)
        name_list1 = list("{}".format(image_name))
        arrayedimage = np.array(img)
        for x in name_list1:
            if type(x) is type(int):
                integer_list1 += x
        for n_image in n_image_dir:
            n_image_name = os.path.basename(n_image)
            n_img = Image.open(n_image)
            name_list2 = list("{}".format(n_image_name))
            arrayedimage2 = np.asarray(n_img)
            for item in name_list2:
                if type(item) is type(int):
                    integer_list2 += item
            if  integer_list2 == integer_list1:
                arrayed_images_list = [arrayedimage, arrayedimage2]
                image_files_compiled += [arrayed_images_list]
    return image_files_compiled

print(compile_pictures(IMG_DIR, N_IMG_DIR))
