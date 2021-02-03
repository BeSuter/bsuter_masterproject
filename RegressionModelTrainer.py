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
logger.setLevel(logging.DEBUG)

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
            tf.keras.layers.LayerNormalization(axis=1),
            gnn_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
            hp_nn.HealpyChebyshev5(K=5, Fout=256, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(axis=1),
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
            tf.keras.layers.LayerNormalization(axis=1),
            tf.keras.layers.Dense(2)
        ]
    return layers


def is_test(x, y):
    return x % 5 == 0


def is_train(x, y):
    return not is_test(x, y)


recover = lambda x, y: y


def preprocess_dataset(dset):
    dset = dset.shuffle(const_args["preprocess_dataset"]["shuffle_size"])
    dset = dset.batch(const_args["preprocess_dataset"]["batch_size"],
                      drop_remainder=True)

    if const_args["preprocess_dataset"]["split"]:
        logger.debug(f"Splitting data")
        test_dataset = dset.enumerate().filter(is_test).map(recover)
        train_dataset = dset.enumerate().filter(is_train).map(recover)
        test_dataset = test_dataset.prefetch(2)
        train_dataset = train_dataset.prefetch(2)

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
        dset = dset.prefetch(2)
        return dset


def preprocess_dataset(dset):
    dset = dset.shuffle(const_args["preprocess_dataset"]["batch_size"])
    dset = dset.batch(const_args["preprocess_dataset"]["batch_size"],
                      drop_remainder=True)

    logger.info("Using all noise maps.")
    dset = dset.prefetch(2)
    return dset


def mask_maker(dset):
    iterator = iter(dset)
    bool_mask = hp.mask_good(iterator.get_next()[0][0].numpy())
    indices_ext = np.arange(len(bool_mask))[bool_mask > 0.5]

    return bool_mask, indices_ext


@tf.function
def _make_noise():
    noises = []
    if const_args["noise_type"] == "pixel_noise":
        for tomo in range(const_args["_make_pixel_noise"]["tomo_num"]):
            try:
                noise_path = os.path.join(
                    const_args["_make_pixel_noise"]["noise_dir"],
                    f"NewPixelNoise_tomo={tomo + 1}.npz")
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
            stddev *= 38.798
            noise = tf.random.normal([
                const_args["preprocess_dataset"]["batch_size"],
                const_args["pixel_num"]
            ],
                                     mean=0.0,
                                     stddev=1.0)
            noise = tf.math.multiply(noise, stddev)
            noise = tf.math.add(noise, mean)

            noises.append(noise)
    elif const_args["noise_type"] == "old_noise":
        for tomo in range(const_args["_make_noise"]["tomo_num"]):
            noise = tf.random.normal([
                const_args["preprocess_dataset"]["batch_size"],
                const_args["pixel_num"]
            ],
                                     mean=0.0,
                                     stddev=1.0)
            noise *= const_args["_make_noise"]["ctx"][tomo + 1][0]
            noise += const_args["_make_noise"]["ctx"][tomo + 1][1]
            noises.append(noise)
    elif const_args["noise_type"] == "dominik_noise":
        data_path = "/scratch/snx3000/bsuter/TFRecordNoise"
        raw_noise_dset = get_dataset(data_path)
        noise_dset = preprocess_dataset(raw_noise_dset)
        iterator = iter(noise_dset)
        noise_element = iterator.get_next()
        # Ensure that we have shape (batch_size, pex_len, 4)
        noise = tf.boolean_mask(tf.transpose(noise_element, perm=[0, 2, 1]),
                                const_args["bool_mask"],
                                axis=1)
        logger.debug(f"Noise has shape={tf.shape(noise)}")

    if not const_args["noise_type"] == "dominik_noise":
        noise = tf.stack(noises, axis=-1)

    return noise


@tf.function
def loss(model, x, y):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    loss_object = tf.keras.losses.MeanAbsoluteError()
    y_ = model(x, training=True)

    return loss_object(y_true=y, y_pred=y_)


@tf.function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


@tf.function
def train_step(train_dset, model, optimizer):
    epoch_global_norm = tf.TensorArray(
        tf.float32,
        size=const_args["element_num"],
        dynamic_size=False,
        clear_after_read=False,
    )
    epoch_loss_avg = tf.TensorArray(
        tf.float32,
        size=const_args["element_num"],
        dynamic_size=False,
        clear_after_read=False,
    )
    for element in train_dset.enumerate():
        set = element[1]
        # Ensure that we have shape (batch_size, pex_len, 4)
        kappa_data = tf.boolean_mask(tf.transpose(set[0], perm=[0, 2, 1]),
                                     const_args["bool_mask"],
                                     axis=1)
        labels = set[1]
        # Add noise
        logger.debug("Adding noise")
        kappa_data = tf.math.add(kappa_data, _make_noise())

        # Optimize the model
        loss_value, grads = grad(model, kappa_data, labels)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg = epoch_loss_avg.write(
            tf.dtypes.cast(element[0], tf.int32), loss_value)
        epoch_global_norm = epoch_global_norm.write(
            tf.dtypes.cast(element[0], tf.int32), tf.linalg.global_norm(grads))

    return epoch_loss_avg.stack(), epoch_global_norm.stack()


@tf.function()
def count_elements(dset):
    num = 0
    for element in dset.enumerate():
        num = tf.dtypes.cast(element[0], tf.int32)
    return num + 1


def regression_model_trainer():
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    data_path = const_args["data_dir"]
    logger.info(f"Retrieving data from {data_path}")
    raw_dset = get_dataset(data_path)
    logger.debug(f"Making the mask")
    bool_mask, indices_ext = mask_maker(raw_dset)
    const_args["bool_mask"] = bool_mask
    const_args["pixel_num"] = len(indices_ext)

    # Use all the maps to train the model
    if const_args["preprocess_dataset"]["split"]:
        train_dset, test_dset = preprocess_dataset(raw_dset)
    else:
        train_dset = preprocess_dataset(raw_dset)
    num = count_elements(train_dset)
    const_args["element_num"] = tf.dtypes.cast(num, tf.int32)
    logger.info(f"Number of elements per epoch is {const_args['element_num']}")

    # Define the layers of our model
    optimizer = tf.keras.optimizers.Adam(learning_rate=const_args["l_rate"])

    tf.keras.backend.clear_session()

    layers = get_layers()

    model = hp_nn.HealpyGCNN(nside=const_args["nside"],
                             indices=indices_ext,
                             layers=layers)
    model.build(input_shape=(const_args["preprocess_dataset"]["batch_size"],
                             const_args["pixel_num"],
                             const_args["_make_pixel_noise"]["tomo_num"]))
    if const_args["continue_training"]:
        if const_args["checkpoint_dir"] == "undefined":
            logger.critical("Please define the directory within NGSFweights containing the desired weights")
            logger.critical("E.g. --checkpoint_dir=layer_2/pixel_noise/01-14-2021-18-38")
            sys.exit(0)
        else:
            path_to_weights = os.path.join(const_args["weights_dir"], const_args["checkpoint_dir"])
            logger.info(f"Loading weights from {path_to_weights}")
            model.load_weights(tf.train.latest_checkpoint(path_to_weights))
            const_args["weights_dir"] = os.path.join(const_args["weights_dir"], "RetrainedWeights")


    # Keep results for plotting
    train_loss_results = tf.TensorArray(tf.float32,
                                        size=0,
                                        dynamic_size=True,
                                        clear_after_read=False)
    global_norm_results = tf.TensorArray(tf.float32,
                                         size=0,
                                         dynamic_size=True,
                                         clear_after_read=False)
    if const_args["epochs"] < 9:
        # Prevent zeros division in line 331.
        modulo_epoch = 9
    else:
        modulo_epoch = const_args["epochs"]

    for epoch in range(const_args["epochs"]):
        logger.debug(f"Executing training step for epoch={epoch}")
        # Optimize the model  --> Returns the loss average and the global norm of each epoch
        epoch_loss_avg, epo_glob_norm = train_step(train_dset, model,
                                                   optimizer)

        # End epoch
        if epoch % 10 == 0:
            logger.info("Epoch {:03d}: Loss: {:.3f}".format(
                epoch,
                sum(epoch_loss_avg) / len(epoch_loss_avg)))
            train_loss_results = train_loss_results.write(
                epoch,
                sum(epoch_loss_avg) / len(epoch_loss_avg))
            global_norm_results = global_norm_results.write(
                epoch,
                sum(epo_glob_norm) / len(epo_glob_norm))
        if epoch % int(modulo_epoch // 9) == 0:
            # Evaluate the model and plot the results
            logger.info(
                f"Evaluating the model and plotting the results for epoch={epoch}"
            )
            if not const_args["debug"]:
                epoch_non_zero = epoch + 1

                color_predictions = []
                color_labels = []

                om_histo = []
                s8_histo = []

                all_results = {}
                all_results["om"] = collections.OrderedDict()
                all_results["s8"] = collections.OrderedDict()

                om_pred_check = PredictionLabelComparisonPlot(
                    "Omega_m",
                    epoch=epoch_non_zero,
                    layer=const_args["get_layer"]["layer"],
                    noise_type=const_args["noise_type"],
                    start_time=date_time)
                s8_pred_check = PredictionLabelComparisonPlot(
                    "Sigma_8",
                    epoch=epoch_non_zero,
                    layer=const_args["get_layer"]["layer"],
                    noise_type=const_args["noise_type"],
                    start_time=date_time)

                if const_args["preprocess_dataset"]["split"]:
                    pass
                else:
                    test_dset = train_dset
                for set in test_dset:
                    kappa_data = tf.boolean_mask(tf.transpose(set[0],
                                                              perm=[0, 2, 1]),
                                                 const_args["bool_mask"],
                                                 axis=1)
                    labels = set[1]
                    labels = labels.numpy()

                    # Add noise
                    kappa_data = tf.math.add(kappa_data, _make_noise())
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

                histo_plot(om_histo,
                           "Om",
                           epoch=epoch_non_zero,
                           layer=const_args["get_layer"]["layer"],
                           noise_type=const_args["noise_type"],
                           start_time=date_time)
                histo_plot(s8_histo,
                           "S8",
                           epoch=epoch_non_zero,
                           layer=const_args["get_layer"]["layer"],
                           noise_type=const_args["noise_type"],
                           start_time=date_time)
                l2_color_plot(np.asarray(color_predictions),
                              np.asarray(color_labels),
                              epoch=epoch_non_zero,
                              layer=const_args["get_layer"]["layer"],
                              noise_type=const_args["noise_type"],
                              start_time=date_time)
                S8plot(all_results["om"],
                       "Om",
                       epoch=epoch_non_zero,
                       layer=const_args["get_layer"]["layer"],
                       noise_type=const_args["noise_type"],
                       start_time=date_time)
                S8plot(all_results["s8"],
                       "sigma8",
                       epoch=epoch_non_zero,
                       layer=const_args["get_layer"]["layer"],
                       noise_type=const_args["noise_type"],
                       start_time=date_time)
                om_pred_check.save_plot()
                s8_pred_check.save_plot()
    if not const_args["debug"]:
        stats(train_loss_results.stack().numpy(),
              "training_loss",
              layer=const_args["get_layer"]["layer"],
              noise_type=const_args["noise_type"],
              start_time=date_time)
        stats(global_norm_results.stack().numpy(),
              "global_norm",
              layer=const_args["get_layer"]["layer"],
              noise_type=const_args["noise_type"],
              start_time=date_time)

    if const_args["HOME"]:
        path_to_dir = os.path.join(os.path.expandvars("$HOME"),
                                   const_args["weights_dir"],
                                   const_args["get_layer"]["layer"],
                                   const_args["noise_type"], date_time)
    else:
        path_to_dir = os.path.join(os.path.expandvars("$SCRATCH"),
                                   const_args["weights_dir"],
                                   const_args["get_layer"]["layer"],
                                   const_args["noise_type"], date_time)
    if not const_args["debug"]:
        os.makedirs(path_to_dir, exist_ok=True)
        weight_file_name = f"kappa_batch={const_args['preprocess_dataset']['batch_size']}" + \
                           f"_shuffle={const_args['preprocess_dataset']['shuffle_size']}_epoch={const_args['epochs']}.tf"
        save_weights_to = os.path.join(path_to_dir, weight_file_name)
        logger.info(f"Saving model weights to {save_weights_to}")
        model.save_weights(save_weights_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', type=str, action='store')
    parser.add_argument('--weights_dir', type=str, action='store')
    parser.add_argument('--noise_dir', type=str, action='store')
    parser.add_argument('--batch_size', type=int, action='store')
    parser.add_argument('--shuffle_size', type=int, action='store')
    parser.add_argument('--epochs', type=int, action='store')
    parser.add_argument('--layer', type=str, action='store')
    parser.add_argument('--noise_type',
                        type=str,
                        action='store',
                        default='pixel_noise')
    parser.add_argument('--split_data', action='store_true', default=False)
    parser.add_argument('--nside', type=int, action='store', default=512)
    parser.add_argument('--l_rate', type=float, action='store', default=0.008)
    parser.add_argument('--HOME', action='store_true', default=False)
    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--checkpoint_dir', type=str, action='store', default='undefined')
    ARGS = parser.parse_args()

    print("Starting RegressionModelTrainer")

    # Define all constants used in helper functions
    # --> When tracing, we do not want to have any constant function arguments!
    const_args = {
        "get_layer": {
            "layer": ARGS.layer
        },
        "preprocess_dataset": {
            "batch_size": ARGS.batch_size,
            "shuffle_size": ARGS.shuffle_size,
            "split": ARGS.split_data
        },
        "_make_pixel_noise": {
            "noise_dir": ARGS.noise_dir,
            "tomo_num": 4
        },
        "data_dir": ARGS.data_dirs,
        "weights_dir": ARGS.weights_dir,
        "_make_noise": {
            "tomo_num": 4,
            "ctx": {
                1: [0.060280509803501296, 2.6956629531655215e-07],
                2: [0.06124986702256547, -1.6575954273040043e-07],
                3: [0.06110073383083452, -1.4452612096534303e-07],
                4: [0.06125788725968831, 1.2850254404014072e-07]
            }
        },
        "train_step": {
            "step": 0
        },
        "noise_type": ARGS.noise_type,
        "epochs": ARGS.epochs,
        "nside": ARGS.nside,
        "l_rate": ARGS.l_rate,
        "HOME": ARGS.HOME,
        "continue_training": ARGS.continue_training,
        "checkpoint_dir": ARGS.checkpoint_dir,
        "debug": ARGS.debug
    }

    regression_model_trainer()
