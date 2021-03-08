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


def _write_tfr(serialized_dump, target_path):
    batch_size = len(serialized_dump)
    with tf.io.TFRecordWriter(target_path) as writer:
        for index, serialized_example in enumerate(serialized_dump):
            logger.info(f"Writing serialized_example {index + 1}/{batch_size}")
            writer.write(serialized_example)


def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=)\d+,\d+&\d+(?=_)", str).group(0)
    all_shapes = []
    for item in shapes.split("&"):
        try:
            all_shapes.append((int(item.split(",")[0]), int(item.split(",")[1])))
        except IndexError:
            all_shapes.append((int(item)))
    return all_shapes


def _index_finder(str):
    index = re.search(r"(?<=start=)\d+(?=_step)", str).group(0)
    return index


def get_dataset(path=[]):
    if not isinstance(path, list):
        path = [path]
    for pp in path:
        f_names = [
            os.path.join(pp, file) for file in os.listdir(pp)
            if file.endswith(".tfrecord")
        ]
        for file in f_names:
            logger.info(f"Hitting on {file}")
            shapes = _shape_finder(file)
            dset = tf.data.TFRecordDataset([file])
            decoded_dset = data.decode_labeled_dset(dset, shapes)
            yield decoded_dset, file


def resize_tfr(path, MAP_TYPE, F_COUNT, target_path):
    for dset, file_name in get_dataset(path):
        if MAP_TYPE == "grid":
            index = _index_finder(file_name)
            if index in [0, 1]:
                logger.info(f"Index would have been {index} --> Already performed resizing for that index. Continuing...")
                continue
        serialized_example_dump = []
        for ddata in dset:
            full_map = ddata[0].numpy()
            label = ddata[1].numpy()

            resized_data = []
            for tomo in range(4):
                resized_data.append(full_map[tomo][full_map[tomo] > hp.UNSEEN])
            final_data = np.vstack(resized_data)
            if os.getenv("DEBUG", False):
                logger.debug(f"File Name is {file_name}")
                if not MAP_TYPE == "noise":
                    index = _index_finder(file_name)
                    logger.debug(f"Found index {index} in file name")
                logger.debug(f"Final Data has shape {np.shape(final_data)}")
                logger.debug(final_data)
                logger.debug(f"Stopping the script")
                sys.exit(0)
            serialized_map = data.serialize_labeled_example(final_data, label)
            serialized_example_dump.append(serialized_map)

        logger.info(f"Dumping {MAP_TYPE} maps")
        map_shape = ','.join(map(str, np.shape(final_data)))
        label_shape = ','.join(map(str, np.shape(label)))
        logger.debug(f"Shapes are map_shape={map_shape} and label_shape={label_shape}")
        del final_data
        del label
        if not MAP_TYPE == "noise":
            index = _index_finder(file_name)
            tfrecord_name = f"{MAP_TYPE}_map_cosmo_shapes={map_shape}&{label_shape}_" + \
                            f"start={index}_step={F_COUNT}_{uuid.uuid4().hex}.tfrecord"
        else:
            tfrecord_name = f"{MAP_TYPE}_map_cosmo_shapes={map_shape}&{label_shape}_" + \
                            f"{uuid.uuid4().hex}.tfrecord"
        record_path = os.path.join(target_path, tfrecord_name)
        _write_tfr(serialized_example_dump, record_path)
        serialized_example_dump.clear()


if __name__ == "__main__":
    args = sys.argv[1:]
    MAP_TYPE = str(args[0])

    if MAP_TYPE == "fiducial":
        path = "/cluster/work/refregier/besuter/All_TFR/TFRecordFiducial"
        target_path = "/cluster/work/refregier/besuter/Final_TFR/TFRecordFiducial"
        F_COUNT = 16
    elif MAP_TYPE == "grid":
        path = "/cluster/work/refregier/besuter/All_TFR/TFRecordGrid"
        target_path = "/cluster/work/refregier/besuter/Final_TFR/TFRecordGrid"
        F_COUNT = 5
    elif MAP_TYPE == "noise":
        path = "/cluster/work/refregier/besuter/All_TFR/TFRecordNoise"
        target_path = "/cluster/work/refregier/besuter/Final_TFR/TFRecordNoise"
        F_COUNT = None

    resize_tfr(path, MAP_TYPE, F_COUNT, target_path)


