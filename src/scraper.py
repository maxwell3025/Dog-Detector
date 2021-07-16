import os
import PIL
import numpy as np
from numpy import save
from PIL import Image
from google_images_download import google_images_download   #importing the library

training_size = 20
test_size = 2
img_size = 64

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"dogs,wolves","limit":((training_size+test_size)/2),"silent_mode":True, "output_directory":"images/raw" , "chromedriver": "C:/Program Files (x86)/Google/Chrome/Application/chrome_proxy.exe", "no_directory": False, "aspect_ratio": "square"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function

dogpaths = []
wolfpaths = []
for (dirpath, dirnames, filenames) in os.walk("images/raw/dogs"):
    dogpaths.extend(filenames)
    break

for (dirpath, dirnames, filenames) in os.walk("images/raw/wolves"):
    wolfpaths.extend(filenames)
    break

os.makedirs("images/converted/dogs", exist_ok=True)
os.makedirs("images/converted/wolves", exist_ok=True)

x_train = np.zeros((0, img_size, img_size))
y_train = np.zeros((0, img_size, img_size))
x_test = np.zeros((0, img_size, img_size))
y_test = np.zeros((0, img_size, img_size))

for index, path in enumerate(dogpaths):
	image = PIL.Image.open(f"images/raw/dogs/{path}")
	image = image.convert("L")
	image = image.resize((img_size,img_size))
	if index < training_size/2:
		x_train = np.append(x_train,[np.array(image)])
	else:
		x_test = np.append(x_train,[np.array(image)])


for index, path in enumerate(wolfpaths):
	image = PIL.Image.open(f"images/raw/wolves/{path}")
	image = image.convert("L")
	image = image.resize((img_size,img_size))
	if index < training_size/2:
		x_train = np.append(x_train,[np.array(image)])
	else:
		x_test = np.append(x_train,[np.array(image)])
