import os
import re
import sys
import uuid
import logging
import numpy as np
import healpy as hp

from DeepSphere import data
from DeepSphere.utils import extend_indices
#from estats.catalog import catalog
from utils import _write_tfr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def _label_finder(str):
    Om_label = re.search(r"(?<=Om=).+(?=_eta)", str).group(0)
    s8_label = re.search(r"(?<=s8=).+(?=_tomo)", str).group(0)
    return (float(Om_label), float(s8_label))


def _noise_info(str):
    tomo = re.search(r"(?<=tomo=).+(?=_z)", str).group(0)
    scale = re.search(r"(?<=scale=).+(?=.n)", str).group(0)
    return float(tomo)


def LSF_tfrecord_writer(job_index,
                        path="NGSF_LIGHTCONES",
                        target="NGSFrecords",
                        file_count=32,
                        SCRATCH=True):
    """
    ToDo: Find better way of generating the cosmology labels
    This function is meant to be used in the context of lsf job array.
    :param job_index: int, corresponds to the job index of the job array has to be [1-file_count].
    """
    # Job index starts at 1, array index starts at 0
    array_idx = int(job_index) - 1
    if SCRATCH:
        map_path = os.path.join(os.path.expandvars("$SCRATCH"), path)
        target_path = os.path.join(os.path.expandvars("$SCRATCH"), target)
    else:
        map_path = path
        target_path = target
    f_names = [
        os.path.join(map_path, file) for file in os.listdir(map_path)
        if not file.startswith(".")
    ]
    batch_size = int(len(f_names) / file_count)
    start = int(array_idx * batch_size)
    if array_idx + 1 == file_count:
        name_batch = f_names[start:]
        logger.info(f"Serializing maps {start} to end")
    else:
        end = int(start + batch_size)
        name_batch = f_names[start:end]
        logger.info(f"Serializing maps {start} to {end}")
    serialized_example_dump = []
    for file in name_batch:
        try:
            k_map = np.load(file)
        except IOError as e:
            logger.critical(e)
            continue
        labels = _label_finder(file)
        noise_info = _noise_info(file)
        cosmo_labels = np.array([
            [labels[0], labels[1]], [noise_info[0], noise_info[1]]
        ])
        serialized_example_dump.append(
            data.serialize_labeled_example(k_map, cosmo_labels))
    logger.info("Dumping maps")
    tfrecord_name = f"kappa_map_cosmo={start}-{end-1}_shapes={np.shape(k_map)},{np.shape(cosmo_labels)}_total_{uuid.uuid4().hex}.tfrecord"
    target_path = os.path.join(target_path, tfrecord_name)
    _write_tfr(serialized_example_dump, target_path)
    serialized_example_dump.clear()

if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(os.environ[str(args[0])])
    LSF_tfrecord_writer(job_index)
