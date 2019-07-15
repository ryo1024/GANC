import os
from os import listdir
from PIL import Image

dir_path = "../pictures"


for filename in listdir(dir_path):
    if filename.endswith('.jpg'):
        try:
            img = Image.open("pictures"+"\\"+filename) # open the image file
            img.verify() # verify that it is, in fact an image
            print('valid image')
        except Exception:
            print('Bad file:', filename)
