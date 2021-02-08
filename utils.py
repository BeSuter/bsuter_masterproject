import os
import re
import sys
import logging
import tensorflow as tf

from DeepSphere import data
from DeepSphere import gnn_layers
from DeepSphere import healpy_networks as hp_nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=)\d+,\d+&\d+(?=_)", str).group(0)
    all_shapes = []
    for item in shapes.split("&"):
        try:
            all_shapes.append((int(item.split(",")[0]), int(item.split(",")[1])))
        except IndexError:
            all_shapes.append((int(item)))
    return all_shapes


def get_dataset(path=[]):
    if not isinstance(path, list):
        path = [path]
    all_files = []
    for pp in path:
        f_names = [
            os.path.join(pp, file) for file in os.listdir(pp)
            if file.endswith(".tfrecord")
        ]
        all_files.extend(f_names)
    shapes = _shape_finder(all_files[0])
    dset = tf.data.TFRecordDataset(all_files)
    decoded_dset = data.decode_labeled_dset(dset, shapes)
    return decoded_dset

def get_layers(layer):
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