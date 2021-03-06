#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:14:56 2020

@author: BenjaminSuter
"""
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Download images from the CIfAR10 dataset
#again there will be 10 classes
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize pixal values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#label the classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#Create the convolutional base
#We will use a stack of Conv2D and MaxPooling2D layers
#I assume, that for the Conv2D layers we have a 3x3 matrix as a convolution with
#a 1x1 stride this would result in a loss of 2 dimensions i.e. 32-->30
#For the MaxPooling2D we use a pool_size of 2x2 with a 2x2 strides therefore 
#the output shape will be (input shape - 2 + 1) / 2 for every axis.
#
#We can see the different output shapes when using model.summary()

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

#To complete the model, we will add two Dense layers where the last one will 
#give 10 outputs matching the number of classes.
#We need to feed an array into the Dense layer. Therefore we first need to flaten
#The 3D tensor coming from the convolutional output.

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#Compile and train the model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nResulting test accuracy after 10 epochs of training: ', test_acc)



    