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
from Plotter import l2_color_plot, histo_plot, S8plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def is_test(x, y):
    return x % 10 == 0


def is_train(x, y):
    return not is_test(x, y)


recover = lambda x, y: y


def preprocess_dataset(dset, batch_size, shuffle_size, split=False):
    dset = dset.shuffle(shuffle_size)
    dset = dset.batch(batch_size, drop_remainder=True)

    if split:
        test_dataset = dset.enumerate().filter(is_test).map(recover)
        train_dataset = dset.enumerate().filter(is_train).map(recover)

        test_counter = 0
        train_counter = 0
        for item in test_dataset:
            test_counter += 1
        for item in train_dataset:
            train_counter += 1
        logger.info(
            f"Maps split into training={train_counter*batch_size} and test={test_counter*batch_size}"
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
def _make_noise(map, ctx, tomo_num=4):
    noises = []
    for tomo in range(tomo_num):
        noise = tf.random.normal(map[0, :, tomo].shape, mean=0.0, stddev=1.0)
        noise *= ctx[tomo + 1][0]
        noise += ctx[tomo + 1][1]
        noises.append(noise)

    return tf.stack(noises, axis=1)


def regression_model_trainer(data_path,
                             batch_size,
                             shuffle_size,
                             weights_dir,
                             nside=512):
    noise_ctx = {
        1: [0.060280509803501296, 2.6956629531655215e-07],
        2: [0.06124986702256547, -1.6575954273040043e-07],
        3: [0.06110073383083452, -1.4452612096534303e-07],
        4: [0.06125788725968831, 1.2850254404014072e-07]
    }
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    scratch_path = os.path.expandvars("$SCRATCH")
    data_path = os.path.join(scratch_path, data_path)
    logger.info(f"Retrieving data from {data_path}")
    raw_dset = get_dataset(data_path)
    bool_mask, indices_ext = mask_maker(raw_dset)

    # Use all the maps to train the model
    test_dset = preprocess_dataset(raw_dset, batch_size, shuffle_size)

    layers = [
        hp_nn.HealpyChebyshev5(K=5, activation=tf.nn.elu),
        gnn_layers.HealpyPseudoConv(p=2, Fout=8, activation='relu'),
        hp_nn.HealpyMonomial(K=5, activation=tf.nn.elu),
        gnn_layers.HealpyPseudoConv(p=2, Fout=16, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2)
    ]

    tf.keras.backend.clear_session()

    model = hp_nn.HealpyGCNN(nside=nside, indices=indices_ext, layers=layers)
    model.build(input_shape=(batch_size, len(indices_ext), 4))
    model.load_weights(
        tf.train.latest_checkpoint(weights_dir))

    color_predictions = []
    color_labels = []

    om_histo = []
    s8_histo = []

    all_results = {}
    all_results["om"] = collections.OrderedDict()
    all_results["s8"] = collections.OrderedDict()


    for set in test_dset:
        kappa_data = tf.boolean_mask(tf.transpose(set[0],
                                                  perm=[0, 2, 1]),
                                     bool_mask,
                                     axis=1)
        labels = set[1][:, 0, :]
        labels = labels.numpy()

        # Add noise
        noise = _make_noise(kappa_data, noise_ctx)
        kappa_data = tf.math.add(kappa_data, noise)
        predictions = model(kappa_data)

        for ii, prediction in enumerate(predictions.numpy()):
            color_predictions.append(prediction)
            color_labels.append(labels[ii, :])

            om_histo.append(prediction[0] - labels[ii, 0])
            s8_histo.append(prediction[1] - labels[ii, 1])

            try:
                all_results["om"][(labels[ii][0], labels[ii][1])].append(
                    prediction[0]
                    )
            except KeyError:
                all_results["om"][(labels[ii][0], labels[ii][1])] = [
                    prediction[0]
                ]
            try:
                all_results["s8"][(labels[ii][0], labels[ii][1])].append(
                    prediction[1]
                    )
            except KeyError:
                all_results["s8"][(labels[ii][0], labels[ii][1])] = [
                    prediction[1]
                ]

    histo_plot(om_histo, "Om")
    histo_plot(s8_histo, "S8")
    l2_color_plot(np.asarray(color_predictions),
                  np.asarray(color_labels))
    S8plot(all_results["om"], "Om")
    S8plot(all_results["s8"], "sigma8")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, action='store')
    parser.add_argument('--weights_dir', type=str, action='store')
    parser.add_argument('--batch_size', type=int, action='store')
    parser.add_argument('--shuffle_size', type=int, action='store')
    ARGS = parser.parse_args()

    print("Starting Model Evaluation")
    regression_model_trainer(ARGS.data_dir, ARGS.batch_size, ARGS.shuffle_size,
                             ARGS.weights_dir)