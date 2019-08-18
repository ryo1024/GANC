import cv2
from PIL import Image
import os

IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images_cropped/'
N_IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/noised_images_cropped/'

def compile_pictures(image_dir, n_image_dir):
    image_files_compiled = []
    for root, dirs, files in os.walk(image_dir):
        files.remove(".DS_Store")
        for name in sorted(files):
            print(image_dir + name)
            img = Image.open(image_dir + name)
            arrayedimage = cv2.imread(image_dir + name)
    for n__root, n_dirs, n_files in os.walk(n_image_dir):
        #n_files.remove(".DS_Store")
        for n_name in sorted(n_files):
            n_img = Image.open(n_image_dir + n_name)
            arrayedimage2 = cv2.imread(n_image_dir + n_name)
            arrayed_images_list = [arrayedimage, arrayedimage2]
            image_files_compiled += [arrayed_images_list]
    return image_files_compiled


print(compile_pictures(IMG_DIR, N_IMG_DIR))
