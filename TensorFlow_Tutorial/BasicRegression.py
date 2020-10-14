#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:46:56 2020

@author: BenjaminSuter
"""
#Regression problem, we aim to predict a continuous value.
#We will use the Auto MPG Dataset and build a model that predicts the fuel
#efficiency of the late-1970s and early 1980

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()


# Get the Auto MPG dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
#print(dataset.tail())
# print(dataset.isna().sum()) --> shows howmany unknown values per row

# Drop the rows containing unknown values
dataset = dataset.dropna()

# The "Origin" column is really categorical, not numeric. Convert that to a
# one-hot:
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europ', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
#print(dataset.tail())


# Split the data into a training and a test set
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
# Have a quick look at the joint distribution of a few pairs of columns from
# the training set

#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']],
#             diag_kind='kde')
#plt.show()

# Overall statistics of data
#print(train_dataset.describe().transpose())


# Seperate the target value, the "label", from th efeatures. This is the 
# value that we will train the model to predict!

train_features = train_dataset.copy()
test_features = test_dataset.copy()

# Removing the "labels"
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#print(train_dataset.describe().transpose()[['mean', 'std']]())



# It is good practice to normalize features that use different scales and ranges
#
# --> Features are multiplied by the model weights. So the scale of the output
#     and the scale of the gradient are affected by the scale of the inputs.
# Normalization makes training much more stable.
#
# --> Add a normalization layer
# Use preprocessing.Normalization layer to implement preprocessing into your
# model in a simple way.

normalizer = preprocessing.Normalization()
# Adapt the layer to the data
# --> will calculate the mean and variance, and stores them in the layer
normalizer.adapt(np.array(train_features))

#print(normalizer.mean.numpy())
#
# When the layer is called it returns the input data, with each feature
# independently normalized

first = np.array(train_features[:1])

#with np.printoptions(precision=2, suppress=True):
#    print('First example:', first)
#    print()
#    print('Normalized:', normalizer(first).numpy())


# Linear regression
#
# We start with a single-variable linear regression, to predict MPG from 
# Horsepower
#
# Our sequential steps in the model will be 
# - Normalize the input 'horsepower'
# - Apply a linear transformation (y = mx + b) to produce 1 output using layers.Dense
#
# The number of imputs can be either set by the 'input_shape' argument, or
# automatically when the model is run for the first time

# First create the horspower Normalization layer:
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
horsepower_normalizer.adapt(horsepower)


# Now we can build the sequencial model
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

#plot_loss(history)


test_results = {}
#Collect the results for later
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)



def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.show()



def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


#DNN model for a single input: "Horsepower"
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

print(y)



