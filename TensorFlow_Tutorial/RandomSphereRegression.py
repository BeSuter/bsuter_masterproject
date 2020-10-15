#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:54:19 2020

@author: BenjaminSuter
"""
import tensorflow as tf
import numpy as np
import healpy as hp

from DeepSphere import healpy_networks as hp_nn
from DeepSphere import gnn_layers
import matplotlib.pyplot as plt

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)

all_ipix = list(range(NPIX))
hp_map = np.arange(NPIX)

hp_map_nest = hp.reorder(hp_map, r2n=True)
print(np.shape(hp_map_nest))


# define some layers (can be mixed with normal tf layers)
# define only one output layer --> only want the mean !?
layers = [gnn_layers.HealpyPseudoConv(p=1, Fout=4),
          gnn_layers.HealpyPool(p=1),
          hp_nn.HealpyChebyshev5(K=5, Fout=8),
          gnn_layers.HealpyPseudoConv(p=2, Fout=16),
          hp_nn.HealpyMonomial(K=5, Fout=32),
          hp_nn.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5}),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(1)]

 # build a model
tf.keras.backend.clear_session()
indices = np.arange(NPIX)
model = hp_nn.HealpyGCNN(nside=NSIDE, indices=indices, layers=layers)

# we build the model
# unlike most CNN achitectures you MUST supply in the batch dimension here
model.build(input_shape=(1, len(indices), 1))
#model.summary(line_length=100)

inp = np.random.normal(loc=12, size=(1, len(indices), 1)).astype(np.float32)
out = model(inp)


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), 
              loss='mean_absolute_error')


train_data = []
train_labels = []
for i in range(12000):
    mean = np.random.rand()
    train_labels.append(mean)
    train_data.append(np.random.normal(loc=mean, size=(len(indices), 1)).astype(np.float32))
train_data = np.asarray(train_data)
train_labels = np.asarray(train_labels)


model.fit(train_data, train_labels, epochs=15, validation_split=0.2)

for i in range(10):
    mean= np.random.rand()
    predictions = model(np.random.normal(loc=mean, size=(1, len(indices), 1)))
    print("Model predicted %d, true label is %d" % (int(np.argmax(predictions)), mean))



