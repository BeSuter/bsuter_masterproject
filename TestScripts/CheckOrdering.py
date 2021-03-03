import os
import re

import healpy as hp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    fid_dir = "/scratch/snx3000/bsuter/TFRecordFiducial"
    noise_dir = "/scratch/snx3000/bsuter/TFRecordNoise"

    map_save_dir = "/users/bsuter/maps_for_janis"
    os.makedirs(map_save_dir, exist_ok=True)

    count = 0
    for fid_maps, noise_maps in zip(get_dataset(fid_dir), get_dataset(noise_dir)):
        count += 1

        tomo = 1
        for tomo_map, tomo_noise in zip(fid_maps[0], noise_maps[0]):
            map_input = tomo_map[tomo_map > hp.UNSEEN]
            noise_input = tomo_noise[tomo_noise > hp.UNSEEN]
            map_indices = np.arange(len(tomo_map))[tomo_map > hp.UNSEEN]
            noise_indices = np.arange(len(tomo_noise))[tomo_noise > hp.UNSEEN]
            print(f"Map indices for count={count} and tomo={tomo}: ")
            print(map_indices)
            print(f"Noise indices for count={count} and tomo={tomo}: ")
            print(noise_indices)

            np.save(os.path.join(map_save_dir, f"Map_indices_{count}_tomo_{tomo}.npy"), map_indices)
            np.save(os.path.join(map_save_dir, f"noise_indices_{count}_tomo_{tomo}.npy"), noise_indices)

            np.save(os.path.join(map_save_dir, f"Map_input_{count}_tomo_{tomo}.npy"), map_input)
            np.save(os.path.join(map_save_dir, f"noise_input_{count}_tomo_{tomo}.npy"), noise_input)

            tomo += 1

        if count == 2:
            break
