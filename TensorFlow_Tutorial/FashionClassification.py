#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:04:40 2020

@author: BenjaminSuter
"""
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#Import the Fashion MNIST dataset
#28x28 images with pixal values from 0 to 255 --> For tensorflow we want pixel
#                                                 values from 0 to 1.
#Labels are integers from 0 to 9 labeling the 10 different classes
#Creat an array with the class names corresponding to the integers 0-9

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#Preproccess the data --> convert [0,255] to [0,1]
train_images = train_images / 255.0
test_images = test_images / 255.0


#Look at the data
'''plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()'''

#Build the model
#First configur the layers of the model, then compile the model.
#
#keras.layers.Flatten has no parameters to learn. It just transforms the 28x28
#2D-array into a 28*28=784 1D-array
#
#keras.layers.Dense does have parameters to learn. These are fully connected 
#neural layers. We can specify the number of nodes. The last layer needs to have 
#10 nodes since we want to classify 10 classes.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])

#Now we need to compile the model, i.e. add some more settings. The Loss Function
#measures how accurate the model is during training. We will minimize this function
#Optimizer is how the model is updated based on the data it sees and its loss function
#Metrics is what is used to monitor the training and testing steps
#  --> accuracy: the fraction of the images that are correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Training the model
#Use the model.fit method --> we want to "fit" the model to the training data

model.fit(train_images, train_labels, epochs=10)

#We can compare the model performance on the test dataset to the training dataset

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

'''print('\nTest accuracy:', test_acc)'''

#Accuracy on the test dataset is a littel bit less then on the training dataset
#this is due to overfitting. Look at "Strategies to prevent overfitting"
#
#https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting
#

#Make predictions
#We will attach a softmax layer at the end of our model inorder to convert
#the output into normalized predictions.
#Predictions will contain the normalized probabilities for each image in the 
#test dataset. Therefore every image will have a corresponding array of 10 numbers
#representing the probabilities.

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

for i in range(5):
    predicted_label = int(np.argmax(predictions[i]))
    label = int(test_labels[i])
    print('\nImage number %s is classified as' % i, class_names[predicted_label])
    print('Real label is ', class_names[label])
    

#Predicting a single image
#tf.keras models are optimized to make predictions on a batch of examples at once
#Therfore even when using a single image we have to add it to a batch, where it 
#is the only member.

img = test_images[25]
#Add the image to a batch where it's the only member
img = (np.expand_dims(img,0))

prediction_single = probability_model.predict(img)
single_predicted_label = int(np.argmax(prediction_single))
label = int(test_labels[25])
print('\nImage number 25 is classified as', class_names[predicted_label])
print('Real label is ', class_names[label])

