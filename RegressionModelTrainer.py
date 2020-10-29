import os
import sys
import logging
import numpy as np
import healpy as hp
import tensorflow as tf

from datetime import datetime
from utils import get_dataset
from DeepSphere.utils import extend_indices
from DeepSphere import healpy_networks as hp_nn
from DeepSphere import gnn_layers


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def is_test(x, y):
    return x % 5 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y

def preprocess_dataset(dset, batch_size, shuffle_size):
    dset = dset.shuffle(shuffle_size)

    test_dataset = dset.enumerate().filter(is_test).map(recover)
    train_dataset = dset.enumerate().filter(is_train).map(recover)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    return train_dataset, test_dataset


def mask_maker(n_side, vec, radius=1):
    n_pix = hp.nside2npix(n_side)
    hp_map = np.zeros(n_pix)

    # create a mask
    pix = hp.query_disc(nside=n_side, vec=vec, radius=radius)
    hp_map[pix] = 1.0

    hp_map_nest = hp.reorder(hp_map, r2n=True)
    indices = np.arange(n_pix)[hp_map_nest > 0.5]
    indices_ext = extend_indices(indices=indices, nside_in=n_side, nside_out=8)

    return indices, indices_ext


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    loss_object = tf.keras.losses.MeanAbsoluteError()
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def regression_model_trainer(data_path, batch_size, shuffle_size, epochs,
                             save_weights_dir, nside=256, vec=[-0.4, 1.0, 0.25],
                             radius=0.26, l_rate=0.008, HOME=True):
    NSIDE = nside
    NPIX = hp.nside2npix(NSIDE)
    vec = np.array(vec)

    indices, indices_ext = mask_maker(NSIDE, vec, radius=radius)

    # Define a masking array for tf.tensor
    bool_mask = np.zeros(NPIX)
    for index in range(NPIX):
        if index in indices_ext:
            bool_mask[index] = True

    # Define the layers of our model
    optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)

    layers = [hp_nn.HealpyChebyshev5(K=5, activation=tf.nn.elu),
              gnn_layers.HealpyPseudoConv(p=2, Fout=1, activation='relu'),
              hp_nn.HealpyMonomial(K=5, activation=tf.nn.elu),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(9)]

    tf.keras.backend.clear_session()

    model = hp_nn.HealpyGCNN(nside=NSIDE, indices=indices_ext, layers=layers)
    model.build(input_shape=(1,len(indices_ext), 1))

    scratch_path = os.path.expandvars("$SCRATCH")
    data_path = os.path.join(scratch_path, data_path)
    logger.info(f"Retrieving data from {data_path}")
    raw_dset = get_dataset(data_path)
    train_dset, test_dset = preprocess_dataset(raw_dset, batch_size, shuffle_size)

    # Keep results for plotting
    train_loss_results = []
    global_norm_results = []

    num_epochs = epochs

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_global_norm = []

        # Training loop - use batches of 500
        # Generate the data on the fly
        for set in train_dset:
            kappa_data = tf.expand_dims(tf.boolean_mask(set[0], bool_mask, axis=1), axis=-1)
            labels = set[1]

            # Optimize the model
            loss_value, grads = grad(model, kappa_data, labels)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            #epoch_accuracy.update_state(labels, model(kappa_data, training=True))
            epoch_global_norm.append(tf.linalg.global_norm(grads))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        global_norm_results.append((sum(epoch_global_norm) / len(epoch_global_norm)))

        if epoch % 10 == 0:
            logger.info("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

    if HOME:
        path_to_dir = os.path.join(os.path.expandvars("$HOME"), save_weights_dir)
    else:
        path_to_dir = os.path.join(os.path.expandvars("$SCRATCH"), save_weights_dir)
    os.makedirs(path_to_dir, exist_ok=True)
    weight_file_name = f"kappa_batch={batch_size}_shuffle={shuffle_size}_epoch={epochs}_"
    logger.info(f"Saving model weights to {save_weights_to}")
    model.save_weights(save_weights_to)



if __name__ == "__main__":
    print("Starting RegressionModelTrainer")
    args = sys.argv[1:]
    regression_model_trainer(args[0], args[1], args[2], args[3], args[4])