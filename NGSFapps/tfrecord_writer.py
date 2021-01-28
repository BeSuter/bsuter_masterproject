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


def _write_tfr(serialized_dump, target_path):
    batch_size = len(serialized_dump)
    with tf.io.TFRecordWriter(target_path) as writer:
        for index, serialized_example in enumerate(serialized_dump):
            logger.info(f"Writing serialized_example {index + 1}/{batch_size}")
            writer.write(serialized_example)


def _label_finder(str):
    Om_label = re.search(r"(?<=Om=).+(?=_eta)", str).group(0)
    s8_label = re.search(r"(?<=s8=).+(?=_tomo)", str).group(0)
    return (float(Om_label), float(s8_label))


def _noise_info(str):
    tomo = re.search(r"(?<=tomo=).+(?=_z)", str).group(0)
    scale = re.search(r"(?<=scale=).+(?=.n)", str).group(0)
    return (int(tomo), float(scale))


def collect_noise(noise_id):
    noise_dir = "/cluster/work/refregier/besuter/data/NoiseMaps"
    error_flag = False
    for cut in range(8):
        full_tomo_map = np.zeros(0)
        for tomo in range(1,5):
            try:
                tomo_map = np.load(os.path.join(
                    noise_dir,
                    f"Rotated_NOISE_mode=E_noise={noise_id}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy")
                )[cut]
            except (IOError, ValueError, IndexError) as e:
                logger.info(
                    f"Error while loading Rotated_NOISE_mode=E_noise={noise_id}_stat=" +
                    f"GetSmoothedMap_tomo={tomo}x{tomo}.npy " +
                    "excluding from TFRecord files \n")
                print(e)
                error_flag = True
                break
            full_tomo_map = np.append(full_tomo_map, tomo_map)
        if error_flag:
            break
        yield full_tomo_map


def collect_fiducial(idx):
    pass


def collect_grid(idx):
    pass


def cleanup_noise(noise_id):
    noise_dir = "/cluster/work/refregier/besuter/data/NoiseMaps"
    for tomo in range(1, 5):
        file_path = os.path.join(
            noise_dir,
            f"Rotated_NOISE_mode=E_noise={noise_id}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy")
        yield file_path


def cleanup_fiducial(idx):
    pass


def cleanup_grid(idx):
    pass


def LSF_tfrecord_writer(job_index,
                        MAP_TYPE="noise",
                        target="NGSFrecordsDomoNoise",
                        file_count=16):
    """
    ToDo: Find better way of generating the cosmology labels
    This function is meant to be used in the context of lsf job array.
    :param job_index: int, corresponds to the job index of the job array has to be [1-file_count].
    """
    # Job index starts at 1, array index starts at 0
    array_idx = int(job_index) - 1
    if os.getenv("DEBUG", False):
        work_dir = os.path.expandvars("$SCRATCH")
        target_path = os.path.join(work_dir, target)
        logger.debug(f"In debug mode. Target top_dir is {target_path}")
    else:
        target_path = "/cluster/work/refregier/besuter/data/NoiseMaps"
    os.makedirs(target_path, exist_ok=True)

    serialized_example_dump = []
    for idx in range(array_idx, 2001, file_count):
        if MAP_TYPE == "noise":
            label = np.array([0]) # We do not need a label for noise maps
            for full_tomo_map in collect_noise(idx):
                serialized_map = data.serialize_labeled_example(full_tomo_map, label)
                serialized_example_dump.append(serialized_map)
        elif MAP_TYPE == "kappa":
            for full_tomo_map, label in collect_fiducial(idx):
                serialized_map = data.serialize_labeled_example(full_tomo_map, label)
                serialized_example_dump.append(serialized_map)
            for full_tomo_map, label in collect_grid(idx):
                serialized_map = data.serialize_labeled_example(full_tomo_map, label)
                serialized_example_dump.append(serialized_map)
        if os.getenv("DEBUG", False):
            logger.debug("In debug mode. Stopping after 1 iteration")
            break
    logger.info(f"Dumping {MAP_TYPE} maps")
    map_shape = ','.join(map(str, np.shape(full_tomo_map)))
    label_shape = ','.join(map(str, np.shape(label)))
    del(full_tomo_map)
    del(label)
    tfrecord_name = f"{MAP_TYPE}_map_cosmo_shapes={map_shape}&{label_shape}_{uuid.uuid4().hex}.tfrecord"
    record_path = os.path.join(target_path, tfrecord_name)
    _write_tfr(serialized_example_dump, record_path)
    serialized_example_dump.clear()
    if not os.getenv("DEBUG", False):
        logger.info("Cleaning up")
        for idx in range(array_idx, 2001, file_count):
            if MAP_TYPE == "noise":
                for file in cleanup_noise(idx):
                    if os.isfile(file):
                        os.remove(file)
            if MAP_TYPE == "kappa":
                for file in cleanup_fiducial(idx):
                    if os.isfile(file):
                        os.remove(file)
                for file in cleanup_grid(idx):
                    if os.isfile(file):
                        os.remove(file)


if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(args[0])
    LSF_tfrecord_writer(job_index)
