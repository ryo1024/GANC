import numpy as np
import os

IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images_cropped/'
N_IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/noised_images_cropped/'

image_files_compiled = []
def compile_pictures(a, b):
    image_file = os.listdir(a)
    n_image_file = os.listdir(b)
    i = 0
    while i < len(image_file)-1:
        image = image_file[i]
        list1 = list(image)
        arrayedimage = np.asarray(i)
        for i in list1:
            integer_list =[]
            if i == type(int):
                integer_list += i
        for i in n_image_file:
            list2 = list(i)
            arrayedimage2 = np.asarray(i)
            for item in list2:
                integer_list2 =[]
                if item == type(int):
                    integer_list2 += item
            if integer_list == integer_list2:
                arrayed_images_list = [arrayedimage, arrayedimage2]
                image_files_compiled += arrayed_images_list
        i += 1

compile_pictures(IMG_DIR, N_IMG_DIR)
