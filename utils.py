import os
import re
import sys
import uuid
import logging
import numpy as np
import tensorflow as tf

from DeepSphere import data

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=).+(?=\_total)", str).group(0)
    shapes = [(int(item), ) for item in shapes.split(",")]
    return shapes


def tfrecord_writer(path, target="TFRecords", file_count=32, SCRATCH=True):
    """
    Writes data in 'path' to a TFRecord file. The data will be batched such that there will
    be #file_count TFRecord files created.
    :param path: string, absolute path to data directory
    :param target: string, path to directory where TFRecord files will be written to
    :param file_count: int, number of TFRecord files created
    :param SCRATCH: bool, if True we have $SCRATCH/target
    """
    if SCRATCH:
        target = os.path.join(os.path.expandvars("$SCRATCH"), target)
    os.makedirs(target, exist_ok=True)

    map_count = len(
        [file for file in os.listdir(path) if not file.startswith(".")])
    batch_size = int(map_count / file_count)
    serialized_example_dump = []

    all_cosmologies = np.genfromtxt("cosmo.par")
    for cosmology_label in all_cosmologies:
        omega_m = cosmology_label[0]
        sigma_8 = cosmology_label[6]
        for num in range(5):
            file = f"kappa_map_cosmo_Om={omega_m}_num={num}_s8={sigma_8}_total.npy"
            file_path = os.path.join(path, file)
            try:
                kappa_map = np.load(file_path)
            except IOError as e:
                logger.critical(e)
                continue
            kappa_map = kappa_map - np.mean(kappa_map)

            serialized_example_dump.append(
                data.serialize_labeled_example(kappa_map, cosmology_label))

            if len(serialized_example_dump) % batch_size == 0 and not len(
                    serialized_example_dump) == 0:
                tfrecord_name = f"kappa_map_cosmo_shapes={len(kappa_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
                target_path = os.path.join(target, tfrecord_name)

                with tf.io.TFRecordWriter(target_path) as writer:
                    for serialized_example in serialized_example_dump:
                        writer.write(serialized_example)
                serialized_example_dump.clear()

    if not len(serialized_example_dump) == 0:
        tfrecord_name = f"kappa_map_cosmo_shapes={len(kappa_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
        target_path = os.path.join(target, tfrecord_name)

        with tf.io.TFRecordWriter(target_path) as writer:
            for serialized_example in serialized_example_dump:
                writer.write(serialized_example)


def get_dataset(path):
    f_names = [
        os.path.join(path, file) for file in os.listdir(path)
        if not file.startswith(".")
    ]
    shapes = _shape_finder(f_names[0])
    dset = tf.data.TFRecordDataset(f_names)
    decoded_dset = data.decode_labeled_dset(dset, shapes)
    return decoded_dset
