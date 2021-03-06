import os
import re
import sys
import uuid
import logging
import numpy as np

import tensorflow as tf
from DeepSphere import data


if os.getenv("DEBUG", False):
    lvl = logging.DEBUG
else:
    lvl = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(lvl)

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


def collect_noise(noise_idx):
    noise_dir = "/cluster/work/refregier/besuter/data/NoiseMaps"
    error_flag = False
    for cut in range(8):
        full_tomo_map = []
        for tomo in range(1,5):
            try:
                tomo_map = np.load(os.path.join(
                    noise_dir,
                    f"Rotated_NOISE_mode=E_noise={noise_idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy")
                )[cut]
            except (IOError, ValueError, IndexError) as e:
                logger.info(
                    f"Error while loading Rotated_NOISE_mode=E_noise={noise_idx}_stat=" +
                    f"GetSmoothedMap_tomo={tomo}x{tomo}.npy " +
                    "excluding from TFRecord files")
                print(e)
                error_flag = True
                break
            full_tomo_map.append(tomo_map)
        if error_flag:
            break
        full_tomo_map = np.asarray(full_tomo_map)
        yield full_tomo_map


def collect_fiducial(noise_idx, real_idx):
    fiducial_dir = "/cluster/work/refregier/besuter/data/SmoothedMaps"
    error_flag = False
    for cut in range(8):
        full_tomo_map = []
        for tomo in range(1,5):
            try:
                tomo_map = np.load(os.path.join(
                    fiducial_dir,
                    f"Rotated_SIM_IA=0.0_Om=0.26_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8=0.84_stat=" + \
                    f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy")
                )[cut]
            except (IOError, ValueError, IndexError) as e:
                logger.debug(
                    f"Error while loading Rotated_SIM_IA=0.0_Om=0.26_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8=0.84_" +
                    f"stat=GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy " +
                    "excluding from TFRecord files")
                print(e)
                error_flag = True
                break
            full_tomo_map.append(tomo_map)
        if error_flag:
            break
        full_tomo_map = np.asarray(full_tomo_map)
        yield full_tomo_map


def collect_grid(noise_idx, real_idx, omega_m, sigma_8):
    grid_dir = "/cluster/work/refregier/besuter/data/SmoothedMaps"
    error_flag = False
    for cut in range(8):
        full_tomo_map = []
        for tomo in range(1, 5):
            try:
                tomo_map = np.load(os.path.join(
                    grid_dir,
                    f"Rotated_SIM_IA=0.0_Om={omega_m}_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8={sigma_8}_stat=" + \
                    f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy")
                )[cut]
            except (IOError, ValueError, IndexError) as e:
                logger.debug(
                    f"Error while loading Rotated_SIM_IA=0.0_Om={omega_m}_eta=0.0_m=0.0_mode=E_noise={noise_idx}" +
                    f"_s8={sigma_8}_stat=GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy " +
                    "excluding from TFRecord files")
                print(e)
                error_flag = True
                break
            full_tomo_map.append(tomo_map)
        if error_flag:
            break
        full_tomo_map = np.asarray(full_tomo_map)
        yield full_tomo_map


def cleanup_noise(noise_id):
    noise_dir = "/cluster/work/refregier/besuter/data/NoiseMaps"
    for tomo in range(1, 5):
        file_path = os.path.join(
            noise_dir,
            f"Rotated_NOISE_mode=E_noise={noise_id}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy")
        yield file_path


def cleanup_fiducial(noise_idx, real_idx):
    fiducial_dir = "/cluster/work/refregier/besuter/data/SmoothedMaps"
    for tomo in range(1, 5):
        file_path = os.path.join(
            fiducial_dir,
            f"Rotated_SIM_IA=0.0_Om=0.26_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8=0.84_stat=" +
            f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy")
        yield file_path


def cleanup_grid(noise_idx, real_idx, omega_m, sigma_8):
    grid_dir = "/cluster/work/refregier/besuter/data/SmoothedMaps"
    for tomo in range(1, 5):
        file_path = os.path.join(
            grid_dir,
            f"Rotated_SIM_IA=0.0_Om={omega_m}_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8={sigma_8}_stat=" +
            f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy")
        yield file_path


def LSF_tfrecord_writer(job_index,
                        MAP_TYPE,
                        target="NGSFrecordsDomoNoise",
                        file_count=16):
    # Job index starts at 1, array index starts at 0
    array_idx = int(job_index) - 1
    if os.getenv("DEBUG", False):
        work_dir = os.path.expandvars("$SCRATCH")
        target_path = os.path.join(work_dir, target)
        logger.debug(f"In debug mode. Target top_dir is {target_path}")
    else:
        if MAP_TYPE == "noise":
            target_path = "/cluster/work/refregier/besuter/data/NoiseMaps"
        if MAP_TYPE == "fiducial":
            target_path = "/cluster/work/refregier/besuter/data/SmoothedMaps"
        if MAP_TYPE == "grid":
            target_path = "/cluster/work/refregier/besuter/data/SmoothedMaps"
    os.makedirs(target_path, exist_ok=True)

    serialized_example_dump = []
    if MAP_TYPE == "noise":
        label = np.array([0])  # We do not need a label for noise maps
        for noise_id in range(array_idx, 2001, file_count):
            for full_tomo_map in collect_noise(noise_id):
                serialized_map = data.serialize_labeled_example(full_tomo_map, label)
                serialized_example_dump.append(serialized_map)
            if os.getenv("DEBUG", False):
                logger.debug("In debug mode. Stopping after 1 iteration")
                break
    elif MAP_TYPE == "fiducial":
        label = np.array([0.26, 0.84]) # Constant label for fiducial cosmology
        for noise_idx in range(10):
            for real_idx in range(array_idx, 50, file_count):
                for full_tomo_map in collect_fiducial(noise_idx, real_idx):
                    serialized_map = data.serialize_labeled_example(full_tomo_map, label)
                    serialized_example_dump.append(serialized_map)
                if os.getenv("DEBUG", False):
                    logger.debug("In debug mode. Stopping after 1 iteration")
                    break
            if os.getenv("DEBUG", False):
                break
    elif MAP_TYPE == "grid":
        file_count = 5
        assert int(job_index) < 6, "For grid cosmologies we parallelize over the realisations ranging from 1 to 5"
        cosmo_file = os.path.join("/cluster/work/refregier/besuter/data", "cosmo.par")
        all_cosmologies = np.genfromtxt(cosmo_file)
        for cosmology in all_cosmologies:
            omega_m = cosmology[0]
            sigma_8 = cosmology[6]
            label = np.array([omega_m, sigma_8])
            for noise_idx in range(10):
                for full_tomo_map in collect_grid(noise_idx, array_idx, omega_m, sigma_8):
                    serialized_map = data.serialize_labeled_example(full_tomo_map, label)
                    serialized_example_dump.append(serialized_map)
                if os.getenv("DEBUG", False):
                    logger.debug("In debug mode. Stopping after 1 iteration")
                    break
            if os.getenv("DEBUG", False):
                break
    logger.info(f"Dumping {MAP_TYPE} maps")
    map_shape = ','.join(map(str, np.shape(full_tomo_map)))
    label_shape = ','.join(map(str, np.shape(label)))
    logger.debug(f"Shapes are map_shape={map_shape} and label_shape={label_shape}")
    del full_tomo_map
    del label
    tfrecord_name = f"{MAP_TYPE}_map_cosmo_shapes={map_shape}&{label_shape}_" +\
                    f"start={array_idx}_step={file_count}_{uuid.uuid4().hex}.tfrecord"
    record_path = os.path.join(target_path, tfrecord_name)
    _write_tfr(serialized_example_dump, record_path)
    serialized_example_dump.clear()
    if not os.getenv("DEBUG", False):
        logger.info("Cleaning up")
        if MAP_TYPE == "noise":
            for idx in range(array_idx, 2001, file_count):
                for file in cleanup_noise(idx):
                    if os.path.isfile(file):
                        os.remove(file)
        if MAP_TYPE == "fiducial":
            for noise_idx in range(10):
                for real_idx in range(50):
                    for file in cleanup_fiducial(noise_idx, real_idx):
                        if os.path.isfile(file):
                            os.remove(file)
        if MAP_TYPE == "grid":
            cosmo_file = os.path.join("/cluster/work/refregier/besuter/data", "cosmo.par")
            all_cosmologies = np.genfromtxt(cosmo_file)
            for cosmology in all_cosmologies:
                omega_m = cosmology[0]
                sigma_8 = cosmology[6]
                for noise_idx in range(10):
                    for file in cleanup_grid(noise_idx, array_idx, omega_m, sigma_8):
                        if os.path.isfile(file):
                            os.remove(file)


if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(args[0])
    MAP_TYPE = str(args[1])
    LSF_tfrecord_writer(job_index, MAP_TYPE)
