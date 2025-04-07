#Note: This code works well in Google Colab
import tensorflow as tf
import numpy as np
import h5py
from tensorflow import keras

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Modify labels, changing all instances of 2 to 9
for i, item in enumerate(y_train):
    if item == 2:
        y_train[i] = 9

for i, item in enumerate(y_test):
    if item == 2:
        y_test[i] = 9

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Save the preprocessed dataset to an HDF5 file
with h5py.File('help.h5', 'w') as file:
    file.create_dataset('x_train', data=x_train)
    file.create_dataset('y_train', data=y_train)
    file.create_dataset('x_test', data=x_test)
    file.create_dataset('y_test', data=y_test)
