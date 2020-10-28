import os
import re
import sys
import uuid
import logging
import numpy as np
import healpy as hp
import tensorflow as tf

from DeepSphere import data

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def _is_power_of_2(n):
    """
    Allows zero to pass as a power of 2
    """
    return (n & (n - 1) == 0)


def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=).+(?=\_total)", str).group(0)
    shapes = [(int(item), ) for item in shapes.split(",")]
    return shapes


def tfrecord_writer(path,
                    target="TFRecords",
                    file_count=32,
                    downsampling=False,
                    SCRATCH=True):
    """
    Writes kappa_maps in 'path' to a TFRecord file. The kappa_maps are reordered to 'NESTED' an will be batched such
    that there will be #file_count TFRecord files created.
    :param path: string, absolute path to data directory
    :param target: string, path to directory where TFRecord files will be written to
    :param file_count: int, number of TFRecord files created
    :param downsampling: int, The desired nside of the output maps
    :param SCRATCH: bool, if True we have $SCRATCH/target
    """
    assert _is_power_of_2(
        downsampling
    ) == True, f"downsampling={downsampling} is not a power of two"

    if SCRATCH:
        target = os.path.join(os.path.expandvars("$SCRATCH"), target)
    os.makedirs(target, exist_ok=True)

    map_count = len(
        [file for file in os.listdir(path) if not file.startswith(".")])
    batch_size = int(map_count / file_count)
    serialized_example_dump = []

    all_cosmologies = np.genfromtxt(
        os.path.join(os.path.expandvars("$SCRATCH"), "cosmo.par"))
    for cosmology_label in all_cosmologies:
        omega_m = cosmology_label[0]
        sigma_8 = cosmology_label[6]
        for num in range(5):
            file = f"kappa_map_cosmo_Om={omega_m}_num={num}_s8={sigma_8}_total.npy"
            file_path = os.path.join(path, file)
            try:
                kappa_map = np.load(file_path)
                if downsampling:
                    kappa_map = hp.ud_grade(kappa_map,
                                            downsampling,
                                            order_out="NESTED")
                else:
                    kappa_map = hp.reorder(kappa_map, r2n=True)
            except IOError as e:
                logger.critical(e)
                continue
            kappa_map = kappa_map - np.mean(kappa_map)

            serialized_example_dump.append(
                data.serialize_labeled_example(kappa_map, cosmology_label))

            if len(serialized_example_dump) % batch_size == 0 and not len(
                    serialized_example_dump) == 0:
                logger.info("Dumping maps")
                res = int(np.sqrt((len(kappa_map) / 12.0)))
                tfrecord_name = f"kappa_map_cosmo_res={res}_shapes={len(kappa_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
                target_path = os.path.join(target, tfrecord_name)

                with tf.io.TFRecordWriter(target_path) as writer:
                    for index, serialized_example in enumerate(
                            serialized_example_dump):
                        logger.info(
                            f"Writing serialized_example {index+1}/{batch_size}"
                        )
                        writer.write(serialized_example)
                serialized_example_dump.clear()

    if not len(serialized_example_dump) == 0:
        logger.info("Dumping remaining maps")
        res = int(np.sqrt((len(kappa_map) / 12.0)))
        tfrecord_name = f"kappa_map_cosmo_res={res}_shapes={len(kappa_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
        target_path = os.path.join(target, tfrecord_name)

        with tf.io.TFRecordWriter(target_path) as writer:
            for index, serialized_example in enumerate(
                    serialized_example_dump):
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
