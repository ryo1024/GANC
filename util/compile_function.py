import cv2
import os
from sklearn.preprocessing import normalize
import numpy as np

IMG_DIR = '../data/images_cropped/'
N_IMG_DIR = '../data/gaussian_noised_images_cropped/'

def compile_pictures(image_dir, n_image_dir):
    image_file_sorted = []
    n_image_file_sorted = []
    dim_stored_list = []
    dim_stored_list2 = []
    for root, dirs, files in os.walk(image_dir):
        #files.remove(".DS_Store")
        for name in sorted(files):
            print(name)
            image_file_sorted += [name]
            arrayedimage = cv2.imread(image_dir + name)
            arrayedimage_reshaped = arrayedimage.reshape(768, 256)
            normalized_image = normalize(arrayedimage_reshaped)
            dimension_stored_image = normalized_image.reshape(256, 256, 3)
            dim_stored_list.append(dimension_stored_image)
    for n_root, n_dirs, n_files in os.walk(n_image_dir):
        #n_files.remove(".DS_Store")
        for n_name in sorted(n_files):
            print(n_name)
            n_image_file_sorted += [n_name]
            arrayedimage2 = cv2.imread(n_image_dir + n_name)
            arrayedimage2_reshaped = arrayedimage2.reshape(768, 256)
            normalized_image2 = normalize(arrayedimage2_reshaped)
            dimension_stored_image2 = normalized_image2.reshape(256, 256, 3)
            dim_stored_list2.append(dimension_stored_image2)

    image_files_name = np.array([[x, y] for (x, y) in zip(image_file_sorted, n_image_file_sorted)])
    image_files_compiled = np.array([[[x], [y]] for (x, y) in zip(dim_stored_list, dim_stored_list2)])

    for pair in image_files_name:
        if not pair[0][7:11] == pair[1][14:18]:
            print(pair[0][7:11], pair[1][14:18])

    return image_files_compiled, image_files_name

compiled_list = compile_pictures(IMG_DIR, N_IMG_DIR)
np.save('../data/npy/gaussian_data.npy', compiled_list[0])
print(compiled_list)
