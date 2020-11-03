import os
import sys
import logging
import numpy as np
import healpy as hp
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from utils import get_dataset
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


recover = lambda x, y: y


def preprocess_dataset(dset, batch_size, shuffle_size):
    dset = dset.shuffle(shuffle_size)
    dset = dset.batch(batch_size, drop_remainder=True)

    test_dataset = dset.enumerate().filter(is_test).map(recover)
    train_dataset = dset.enumerate().filter(is_train).map(recover)

    test_counter = 0
    train_counter = 0
    for item in test_dataset:
        test_counter += 1
    for item in train_counter:
        train_counter += 1
    logger.info(
        f"Maps split into training={train_counter*batch_size} and test={test_counter*batch_size}"
    )
    return train_dataset, test_dataset


def mask_maker(dset):
    """
    ToDO: Find better name for function
    """
    iterator = iter(dset)
    bool_mask = hp.mask_good(iterator.get_next()[0].numpy())
    indices_ext = np.arange(len(bool_mask))[bool_mask > 0.5]

    return bool_mask, indices_ext


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


def regression_model_trainer(data_path,
                             batch_size,
                             shuffle_size,
                             epochs,
                             save_weights_dir,
                             nside=256,
                             l_rate=0.008,
                             HOME=True):
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    scratch_path = os.path.expandvars("$SCRATCH")
    data_path = os.path.join(scratch_path, data_path)
    logger.info(f"Retrieving data from {data_path}")
    raw_dset = get_dataset(data_path)
    bool_mask, indices_ext = mask_maker(raw_dset)

    train_dset, test_dset = preprocess_dataset(raw_dset, batch_size,
                                               shuffle_size)

    # Define the layers of our model
    optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)

    layers = [
        hp_nn.HealpyChebyshev5(K=5, activation=tf.nn.elu),
        gnn_layers.HealpyPseudoConv(p=2, Fout=1, activation='relu'),
        hp_nn.HealpyMonomial(K=5, activation=tf.nn.elu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(9)
    ]

    tf.keras.backend.clear_session()

    model = hp_nn.HealpyGCNN(nside=nside, indices=indices_ext, layers=layers)
    model.build(input_shape=(batch_size, len(indices_ext), 1))

    # Keep results for plotting
    train_loss_results = []
    global_norm_results = []

    num_epochs = epochs

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_global_norm = []

        for set in train_dset:
            kappa_data = tf.expand_dims(tf.boolean_mask(set[0],
                                                        bool_mask,
                                                        axis=1),
                                        axis=-1)
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
        global_norm_results.append(
            (sum(epoch_global_norm) / len(epoch_global_norm)))

        if epoch % 10 == 0:
            logger.info("Epoch {:03d}: Loss: {:.3f}".format(
                epoch, epoch_loss_avg.result()))

    if HOME:
        path_to_dir = os.path.join(os.path.expandvars("$HOME"),
                                   save_weights_dir)
    else:
        path_to_dir = os.path.join(os.path.expandvars("$SCRATCH"),
                                   save_weights_dir, date_time)
    os.makedirs(path_to_dir, exist_ok=True)
    weight_file_name = f"kappa_batch={batch_size}_shuffle={shuffle_size}_epoch={epochs}.tf"
    save_weights_to = os.path.join(path_to_dir, weight_file_name)
    logger.info(f"Saving model weights to {save_weights_to}")
    model.save_weights(save_weights_to)

    # Evaluate the model and plot the results
    omega_m = plt.figure(figsize=(12, 8))
    sigma_8 = plt.figure(figsize=(12, 8))

    omega_m_ax = omega_m.add_axes([0.1, 0.35, 0.8, 0.6],
                                  ylabel="Predictions",
                                  xlabel="Labels",
                                  title="OmegaM prediction compared to Label")
    true_line = np.linspace(0.1, 0.5, 100)
    omega_m_ax.plot(true_line, true_line, alpha=0.3, color="red")

    sigma_8_ax = omega_m.add_axes([0.1, 0.35, 0.8, 0.6],
                                  ylabel="Predictions",
                                  xlabel="Labels",
                                  title="Sigma8 prediction compared to Label")
    true_line = np.linspace(0.4, 1.3, 100)
    sigma_8_ax.plot(true_line, true_line, alpha=0.3, color="red")

    for index, set in test_dset.enumerate():
        kappa_data = tf.expand_dims(tf.boolean_mask(set[0], bool_mask, axis=1),
                                    axis=-1)
        labels = set[1]

        predictions = model(kappa_data)

        logger.info(f"Plotting predictions {index+1}")
        omega_m_ax.plot(labels[:, 0],
                        predictions[:, 0],
                        marker='o',
                        alpha=0.5,
                        ls='',
                        color="blue")
        sigma_8_ax.plot(labels[:, 6],
                        predictions[:, 6],
                        marker='o',
                        alpha=0.5,
                        ls='',
                        color="blue")

    path_to_plot_dir = os.path.join(os.path.expandvars("$HOME"), "Plots",
                                    date_time)
    os.makedirs(path_to_plot_dir, exist_ok=True)
    omega_m_name = f"OmegaM_comparison_batch={batch_size}_shuffle={shuffle_size}_epoch={epochs}.png"
    sigma_8_name = f"Sigma8_comparison_batch={batch_size}_shuffle={shuffle_size}_epoch={epochs}.png"
    omega_m_path = os.path.join(path_to_plot_dir, omega_m_name)
    sigma_8_path = os.path.join(path_to_plot_dir, sigma_8_name)
    omega_m.savefig(omega_m_path)
    sigma_8.savefig(sigma_8_path)


if __name__ == "__main__":
    print("Starting RegressionModelTrainer")
    args = sys.argv[1:]
    regression_model_trainer(args[0], int(args[1]), int(args[2]), int(args[3]),
                             args[4])
