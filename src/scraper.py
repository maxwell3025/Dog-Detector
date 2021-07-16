import os
import PIL
import numpy as np
from numpy import save
from PIL import Image
from google_images_download import google_images_download   #importing the library
from simple_image_download import simple_image_download as simp
response = simp.simple_image_download()
training_size = 2000
test_size = 100
response.download("dogs,wolves", (training_size+test_size)/2)
img_size = 64


dogpaths = []
wolfpaths = []
for (dirpath, dirnames, filenames) in os.walk("simple_images/dogs"):
    dogpaths.extend(filenames)
    break

for (dirpath, dirnames, filenames) in os.walk("simple_images/wolves"):
    wolfpaths.extend(filenames)
    break


x_train = np.zeros((0, img_size, img_size))
y_train = np.zeros((0, 2))
x_test = np.zeros((0, img_size, img_size))
y_test = np.zeros((0, 2))

for index, path in enumerate(dogpaths):
	image = PIL.Image.open(f"simple_images/dogs/{path}")
	image = image.convert("L")
	image = image.resize((img_size,img_size))
	if index < training_size/2:
		x_train = np.append(x_train,[np.array(image)], axis=0)
		y_train = np.append(y_train,[[1,0]], axis=0)
	else:
		x_test = np.append(x_train,[np.array(image)], axis=0)
		y_test = np.append(y_train,[[1,0]], axis=0)


for index, path in enumerate(wolfpaths):
	image = PIL.Image.open(f"simple_images/wolves/{path}")
	image = image.convert("L")
	image = image.resize((img_size,img_size))
	if index < training_size/2:
		x_train = np.append(x_train,[np.array(image)], axis=0)
		y_train = np.append(y_train,[[0,1]], axis=0)
	else:
		x_test = np.append(x_train,[np.array(image)], axis=0)
		y_test = np.append(y_train,[[0,1]], axis=0)

train_shuffle = np.random.permutation(training_size)
test_shuffle = np.random.permutation(test_size)

x_train = x_train[train_shuffle]
y_train = y_train[train_shuffle]
x_test = x_test[test_shuffle]
y_test = y_test[test_shuffle]
np.save("data/x_train", x_train)
np.save("data/y_train", y_train)
np.save("data/x_test", x_test)
np.save("data/y_test", y_test)