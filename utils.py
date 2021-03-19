import os
import re
import sys
import logging
import tensorflow as tf

from deepsphere import healpy_layers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def decode_labeled_dset(dset, shapes, auto_tune=True):
    """
    Returns a dataset where the proto bufferes were decoded according to the prescription of serialize_labeled_example
    :param dset: the data set to decode
    :param shapes: a list of shapes [shape_sample, shape_label]
    :param auto_tune: use the experimental auto tune feature for the final mapping (dynamic CPU allocation)
    :return: the decoded dset having two elements, sample and label
    """

    # a function to decode a single proto buffer
    def decoder_func(record_bytes):
        scheme = {"sample": tf.io.FixedLenFeature(shapes[0], dtype=tf.float32),
                  "label": tf.io.FixedLenFeature(shapes[1], dtype=tf.float32)}

        example = tf.io.parse_single_example(
                    # Data
                    record_bytes,
                    # Schema
                    scheme
                  )
        return example["sample"], example["label"]

    # return the new dset
    if auto_tune:
        return dset.map(decoder_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # otherwise serial
    return dset.map(decoder_func)


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
    decoded_dset = decode_labeled_dset(dset, shapes)
    return decoded_dset

def get_layers(layer):
    if layer == "layer_1":
        layers = [
            healpy_layers.HealpyChebyshev(K=5, activation=tf.nn.elu),
            healpy_layers.HealpyPseudoConv(p=2, Fout=8, activation='relu'),
            healpy_layers.HealpyMonomial(K=5, activation=tf.nn.elu),
            healpy_layers.HealpyPseudoConv(p=2, Fout=16, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2)
        ]
    if layer == "layer_2":
        layers = [
            healpy_layers.HealpyPseudoConv(p=1, Fout=64, activation=tf.nn.relu),
            # healpy_layers.HealpyPseudoConv(p=1, Fout=128, activation=tf.nn.relu),
            healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(axis=1),
            healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
            healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(axis=1),
            healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
            healpy_layers.Healpy_ResidualLayer("CHEBY",
                                               layer_kwargs={
                                               "K": 5,
                                               "activation": tf.nn.relu
                                           },
                                           use_bn=True,
                                           norm_type="layer_norm"),
            # healpy_layers.HealpyPool(p=1, pool_type="AVG"),
            healpy_layers.Healpy_ResidualLayer("CHEBY",
                                               layer_kwargs={
                                                   "K": 5,
                                                   "activation": tf.nn.relu
                                               },
                                               use_bn=True,
                                               norm_type="layer_norm"),
            healpy_layers.Healpy_ResidualLayer("CHEBY",
                                               layer_kwargs={
                                                   "K": 5,
                                                   "activation": tf.nn.relu
                                               },
                                               use_bn=True,
                                               norm_type="layer_norm"),
            # healpy_layers.HealpyPool(p=1, pool_type="AVG"),
            healpy_layers.Healpy_ResidualLayer("CHEBY",
                                               layer_kwargs={
                                                   "K": 5,
                                                   "activation": tf.nn.relu
                                               },
                                               use_bn=True,
                                               norm_type="layer_norm"),
            healpy_layers.HealpyPool(p=1, pool_type="AVG"),
            healpy_layers.Healpy_ResidualLayer("CHEBY",
                                               layer_kwargs={
                                                   "K": 5,
                                                   "activation": tf.nn.relu
                                               },
                                               use_bn=True,
                                               norm_type="layer_norm"),
            healpy_layers.Healpy_ResidualLayer("CHEBY",
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