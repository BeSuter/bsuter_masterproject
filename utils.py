import os
import re
import numpy as np
import tensorflow as tf

from DeepSphere import data


def _label_finder(str):
    Om_label = re.search(r"(?<=Om=).+(?=_num)", str).group(0)
    s8_label = re.search(r"(?<=s8=).+(?=_total)", str).group(0)
    return (float(Om_label), float(s8_label))


def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=).+(?=\.tfrecord)", str).group(0)
    shapes = [(int(item), ) for item in shapes.split(",")]
    return shapes


def tfrecord_writer(path):
    f_names = os.listdir(path)

    all_cosmologies = []
    for line in open("cosmo.par"):
        li = line.strip()
        if not li.startswith("#"):
            all_cosmologies.append([float(item) for item in li.split(" ")])

    for file in f_names:
        file_path = os.path.join(path, file)
        kappa_map = np.load(file_path)
        kappa_map = kappa_map - np.mean(kappa_map)
        labels = _label_finder(file)

        for index, line in enumerate(all_cosmologies):
            if labels[0] and labels[1] in line:
                cosmology_label = np.asarray(all_cosmologies[index])

        serialized_example_proto = data.serialize_labeled_example(
            kappa_map, cosmology_label)

        os.makedirs("TFRecords", exist_ok=True)
        tfrecord_name = f"kappa_map_cosmo_Om={labels[0]}_num=1_s8={labels[1]}_total_shapes={len(kappa_map)},{len(cosmology_label)}.tfrecord"
        target_path = os.path.join("TFRecords", tfrecord_name)

        with tf.io.TFRecordWriter(target_path) as writer:
            writer.write(serialized_example_proto)


def get_dataset(path):
    f_names = [os.path.join(path, file) for file in os.listdir(path) if not file.startswith(".")]
    shapes = _shape_finder(f_names[0])
    dset = tf.data.TFRecordDataset(f_names)
    decoded_dset = data.decode_labeled_dset(dset, shapes)
    return decoded_dset


if __name__ == "__main__":
#    tfrecord_writer("./kappa_maps")
#    dset = get_dataset("./TFRecords")

