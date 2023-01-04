#----------------------------------------------------------*
# program : 2mnist.py;                date: Oct 18, 2018   *
# version : x0.10;                    status: tested;      *
# ref: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/96d58b5727fcf76106f929f5ce24c40fc9b46d75/2.1-a-first-look-at-a-neural-network.ipynb
#                                                          *
# purpose : demo of mnist net for hand written numerals    *
#           recognition.                                   *
#----------------------------------------------------------*

# Significant reused code from the source code found here: https://github.com/hualili/opencv/blob/master/deep-learning-2022s/2022F-105b-%236mnist-numerals-ch02.py
#
# This code base has been modified to improve model convergence and general readability


import keras
import h5py

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = "Adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 20, batch_size = 256)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
print(test_loss)

model.save("mnist_detector.h5")
