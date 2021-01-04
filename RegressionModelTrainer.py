import os
import sys
import argparse
import logging
import collections
import numpy as np
import healpy as hp
import tensorflow as tf

from datetime import datetime
from utils import get_dataset
from DeepSphere import healpy_networks as hp_nn
from DeepSphere import gnn_layers
from Plotter import l2_color_plot, histo_plot, stats, S8plot, PredictionLabelComparisonPlot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def get_layers():
    layer = const_args["get_layer"]["layer"]
    if layer == "layer_1":
        layers = [
            hp_nn.HealpyChebyshev5(K=5, activation=tf.nn.elu),
            gnn_layers.HealpyPseudoConv(p=2, Fout=8, activation='relu'),
            hp_nn.HealpyMonomial(K=5, activation=tf.nn.elu),
            gnn_layers.HealpyPseudoConv(p=2, Fout=16, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2)
        ]
    if layer == "layer_2":
        layers = [
            gnn_layers.HealpyPseudoConv(p=1, Fout=64, activation=tf.nn.relu),
            gnn_layers.HealpyPseudoConv(p=1, Fout=128, activation=tf.nn.relu),
            hp_nn.HealpyChebyshev5(K=5, Fout=256, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(axis=-1),
            gnn_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
            hp_nn.HealpyChebyshev5(K=5, Fout=256, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(axis=-1),
            gnn_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
            hp_nn.Healpy_ResidualLayer("CHEBY",
                                       layer_kwargs={
                                           "K": 5,
                                           "activation": tf.nn.relu
                                       },
                                       use_bn=True,
                                       norm_type="layer_norm"),
            hp_nn.Healpy_ResidualLayer("CHEBY",
                                       layer_kwargs={
                                           "K": 5,
                                           "activation": tf.nn.relu
                                       },
                                       use_bn=True,
                                       norm_type="layer_norm"),
            hp_nn.Healpy_ResidualLayer("CHEBY",
                                       layer_kwargs={
                                           "K": 5,
                                           "activation": tf.nn.relu
                                       },
                                       use_bn=True,
                                       norm_type="layer_norm"),
            hp_nn.Healpy_ResidualLayer("CHEBY",
                                       layer_kwargs={
                                           "K": 5,
                                           "activation": tf.nn.relu
                                       },
                                       use_bn=True,
                                       norm_type="layer_norm"),
            hp_nn.Healpy_ResidualLayer("CHEBY",
                                       layer_kwargs={
                                           "K": 5,
                                           "activation": tf.nn.relu
                                       },
                                       use_bn=True,
                                       norm_type="layer_norm"),
            hp_nn.Healpy_ResidualLayer("CHEBY",
                                       layer_kwargs={
                                           "K": 5,
                                           "activation": tf.nn.relu
                                       },
                                       use_bn=True,
                                       norm_type="layer_norm"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.Dense(2)
        ]
    return layers


def is_test(x, y):
    return x % 10 == 0


def is_train(x, y):
    return not is_test(x, y)


recover = lambda x, y: y


def preprocess_dataset(dset):
    dset = dset.shuffle(const_args["preprocess_dataset"]["shuffle_size"])
    dset = dset.batch(const_args["preprocess_dataset"]["batch_size"], drop_remainder=True)

    if const_args["preprocess_dataset"]["split"]:
        test_dataset = dset.enumerate().filter(is_test).map(recover)
        train_dataset = dset.enumerate().filter(is_train).map(recover)

        test_counter = 0
        train_counter = 0
        for item in test_dataset:
            test_counter += 1
        for item in train_dataset:
            train_counter += 1
        logger.info(
            f"Maps split into training={train_counter * const_args['preprocess_dataset']['batch_size']} and test={test_counter * const_args['preprocess_dataset']['batch_size']}"
        )
        return train_dataset, test_dataset
    else:
        logger.info("Using all maps for training and evaluation")
        return dset


def mask_maker(dset):
    iterator = iter(dset)
    bool_mask = hp.mask_good(iterator.get_next()[0][0].numpy())
    indices_ext = np.arange(len(bool_mask))[bool_mask > 0.5]

    return bool_mask, indices_ext


@tf.function
def _make_pixel_noise(map):
    noises = []
    for tomo in range(const_args["_make_pixel_noise"]["tomo_num"]):
        try:
            noise_path = os.path.join(const_args["_make_pixel_noise"]["noise_dir"],
                                      f"PixelNoise_tomo={tomo + 1}.npz")
            noise_ctx = np.load(noise_path)
            mean_map = noise_ctx["mean_map"]
            variance_map = noise_ctx["variance_map"]
        except FileNotFoundError:
            logger.critical(
                "Are you trying to read PixelNoise_tomo=2x2.npz or PixelNoise_tomo=2.npz?"
            )
            logger.critical(
                "At the moment the noise is hardcoded to PixelNoise_tomo=2.npz. Please change this..."
            )
            sys.exit(0)

        mean = tf.convert_to_tensor(mean_map, dtype=tf.float32)
        stddev = tf.convert_to_tensor(variance_map, dtype=tf.float32)
        noise = tf.random.normal(map[:, :, tomo].shape, mean=0.0, stddev=1.0)
        noise = tf.math.multiply(noise, stddev)
        noise = tf.math.add(noise, mean)

        noises.append(noise)

    return tf.stack(noises, axis=-1)


@tf.function
def _make_noise(map):
    noises = []
    for tomo in range(const_args["_make_noise"]["tomo_num"]):
        noise = tf.random.normal(map[:, :, tomo].shape, mean=0.0, stddev=1.0)
        noise *= const_args["_make_noise"]["ctx"][tomo + 1][0]
        noise += const_args["_make_noise"]["ctx"][tomo + 1][1]
        noises.append(noise)
    return tf.stack(noises, axis=-1)


@tf.function
def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    loss_object = tf.keras.losses.MeanAbsoluteError()
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)


@tf.function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


@tf.function
def train_step(maps, labels, model, optimizer, epoch_loss_avg):
    # Add noise
    logger.debug("Adding noise")
    if const_args["noise_type"] == "pixel_noise":
        noise = _make_pixel_noise(maps)
    elif const_args["noise_type"] == "old_noise":
        noise = _make_noise(maps)
    logger.debug(f"Noise has shape {noise.shape}")
    kappa_data = tf.math.add(maps, noise)
    logger.debug(f"Noisy data has shape {kappa_data.shape}")

    # Optimize the model
    loss_value, grads = grad(model, kappa_data, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    epoch_loss_avg.update_state(loss_value)

    return tf.linalg.global_norm(grads)


def regression_model_trainer():
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    scratch_path = os.path.expandvars("$SCRATCH")
    data_path = os.path.join(scratch_path, const_args["data_dir"])
    logger.info(f"Retrieving data from {data_path}")
    raw_dset = get_dataset(data_path)
    bool_mask, indices_ext = mask_maker(raw_dset)

    # Use all the maps to train the model
    train_dset = preprocess_dataset(raw_dset)

    # Define the layers of our model
    optimizer = tf.keras.optimizers.Adam(learning_rate=const_args["l_rate"])

    tf.keras.backend.clear_session()

    layers = get_layers()

    model = hp_nn.HealpyGCNN(nside=const_args["nside"], indices=indices_ext, layers=layers)
    model.build(input_shape=(const_args["preprocess_dataset"]["batch_size"],
                             len(indices_ext),
                             const_args["_make_pixel_noise"]["tomo_num"]))

    # Keep results for plotting
    train_loss_results = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    global_norm_results = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

    for epoch in range(const_args["epochs"]):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_global_norm = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        for element in train_dset.enumerate():
            const_args["train_step"]["step"] = element[0]
            set = element[1]
            # Ensure that we have shape (batch_size, pex_len, 4)
            logger.debug("Setting input shape")
            kappa_data = tf.boolean_mask(tf.transpose(set[0], perm=[0, 2, 1]),
                                         bool_mask,
                                         axis=1)
            logger.debug(f"Input shape set to {kappa_data.shape}")
            labels = set[1][:, 0, :]

            # Optimize the model  --> Returns the loss average and the global norm of each epoch
            glob_norm = train_step(kappa_data, labels, model, optimizer, epoch_loss_avg)
            epoch_global_norm = epoch_global_norm.write(const_args["train_step"]["step"], glob_norm)

        # End epoch
        train_loss_results = train_loss_results.write(epoch, epoch_loss_avg.result())
        global_norm_results = global_norm_results.write(epoch,
                                                        (sum(epoch_global_norm.stack()) / len(
                                                            epoch_global_norm.stack())))

        if epoch % 10 == 0:
            logger.info("Epoch {:03d}: Loss: {:.3f}".format(
                epoch, epoch_loss_avg.result()))
        if epoch % int(const_args["epochs"] // 9) == 0:
            # Evaluate the model and plot the results
            epoch_non_zero = epoch + 1

            color_predictions = []
            color_labels = []

            om_histo = []
            s8_histo = []

            all_results = {}
            all_results["om"] = collections.OrderedDict()
            all_results["s8"] = collections.OrderedDict()

            om_pred_check = PredictionLabelComparisonPlot("Omega_m",
                                                          epoch=epoch_non_zero,
                                                          layer=const_args["get_layers"]["layer"])
            s8_pred_check = PredictionLabelComparisonPlot("Sigma_8",
                                                          epoch=epoch_non_zero,
                                                          layer=const_args["get_layers"]["layer"])

            test_dset = preprocess_dataset(raw_dset)
            for set in test_dset:
                kappa_data = tf.boolean_mask(tf.transpose(set[0],
                                                          perm=[0, 2, 1]),
                                             bool_mask,
                                             axis=1)
                labels = set[1][:, 0, :]
                labels = labels.numpy()

                # Add noise
                if const_args["noise_type"] == "pixel_noise":
                    noise = _make_pixel_noise(kappa_data)
                elif const_args["noise_type"] == "old_noise":
                    noise = _make_noise(kappa_data)
                kappa_data = tf.math.add(kappa_data, noise)

                predictions = model(kappa_data)

                for ii, prediction in enumerate(predictions.numpy()):
                    om_pred_check.add_to_plot(prediction[0], labels[ii, 0])
                    s8_pred_check.add_to_plot(prediction[1], labels[ii, 1])

                    color_predictions.append(prediction)
                    color_labels.append(labels[ii, :])

                    om_histo.append(prediction[0] - labels[ii, 0])
                    s8_histo.append(prediction[1] - labels[ii, 1])

                    try:
                        all_results["om"][(labels[ii][0],
                                           labels[ii][1])].append(
                            prediction[0])
                    except KeyError:
                        all_results["om"][(labels[ii][0],
                                           labels[ii][1])] = [prediction[0]]
                    try:
                        all_results["s8"][(labels[ii][0],
                                           labels[ii][1])].append(
                            prediction[1])
                    except KeyError:
                        all_results["s8"][(labels[ii][0],
                                           labels[ii][1])] = [prediction[1]]

            histo_plot(om_histo, "Om", epoch=epoch_non_zero, layer=const_args["get_layers"]["layer"])
            histo_plot(s8_histo, "S8", epoch=epoch_non_zero, layer=const_args["get_layers"]["layer"])
            l2_color_plot(np.asarray(color_predictions),
                          np.asarray(color_labels),
                          epoch=epoch_non_zero,
                          layer=const_args["get_layers"]["layer"])
            S8plot(all_results["om"], "Om", epoch=epoch_non_zero, layer=const_args["get_layers"]["layer"])
            S8plot(all_results["s8"],
                   "sigma8",
                   epoch=epoch_non_zero,
                   layer=const_args["get_layers"]["layer"])
            om_pred_check.save_plot()
            s8_pred_check.save_plot()
    stats(train_loss_results.stack().numpy(), "training_loss", layer=const_args["get_layers"]["layer"])
    stats(global_norm_results.stack().numpy(), "global_norm", layer=const_args["get_layers"]["layer"])

    if const_args["HOME"]:
        path_to_dir = os.path.join(os.path.expandvars("$HOME"),
                                   const_args["weights_dir"], const_args["get_layers"]["layer"], date_time)
    else:
        path_to_dir = os.path.join(os.path.expandvars("$SCRATCH"),
                                   const_args["weights_dir"], const_args["get_layers"]["layer"], date_time)
    os.makedirs(path_to_dir, exist_ok=True)
    weight_file_name = f"kappa_batch={const_args['preprocess_dataset']['batch_size']}" + \
                       f"_shuffle={const_args['preprocess_dataset']['shuffle_size']}_epoch={const_args['epochs']}.tf"
    save_weights_to = os.path.join(path_to_dir, weight_file_name)
    logger.info(f"Saving model weights to {save_weights_to}")
    model.save_weights(save_weights_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, action='store')
    parser.add_argument('--weights_dir', type=str, action='store')
    parser.add_argument('--noise_dir', type=str, action='store')
    parser.add_argument('--batch_size', type=int, action='store')
    parser.add_argument('--shuffle_size', type=int, action='store')
    parser.add_argument('--epochs', type=int, action='store')
    parser.add_argument('--layer', type=str, action='store')
    parser.add_argument('--noise_type', type=str, action='store', default='pixel_noise')
    parser.add_argument('--nside', type=int, action='store', default=512)
    parser.add_argument('--l_rate', type=float, action='store', default=0.008)
    parser.add_argument('--HOME', action='store_true', default=False)
    ARGS = parser.parse_args()

    print("Starting RegressionModelTrainer")

    # Define all constants used in helper functions
    # --> When tracing, we do not want to have any constant function arguments!
    const_args = {
        "get_layer": {"layer": ARGS.layer},
        "preprocess_dataset": {
            "batch_size": ARGS.batch_size,
            "shuffle_size": ARGS.shuffle_size,
            "split": False
        },
        "_make_pixel_noise": {
            "noise_dir": ARGS.noise_dir,
            "tomo_num": 4
        },
        "data_dir": ARGS.data_dir,
        "weights_dir": ARGS.weights_dir,
        "_make_noise": {"tomo_num": 4,
                        "ctx": {
                            1: [0.060280509803501296, 2.6956629531655215e-07],
                            2: [0.06124986702256547, -1.6575954273040043e-07],
                            3: [0.06110073383083452, -1.4452612096534303e-07],
                            4: [0.06125788725968831, 1.2850254404014072e-07]
                        }
                        },
        "train_step": {"step": 0},
        "noise_type": ARGS.noise_type,
        "epochs": ARGS.epochs,
        "nside": ARGS.nside,
        "l_rate": ARGS.l_rate
    }

    regression_model_trainer()
