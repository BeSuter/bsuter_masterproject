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

    fid_maps = iter(get_dataset(fid_dir)).get_next()
    noise_maps = iter(get_dataset(noise_dir)).get_next()

    fiducial_map_1 = fid_maps[0][0]
    noise_map_1 = noise_maps[0][0]

    fpp1 = hp.anafast(fiducial_map_1.numpy())
    npp1 = hp.anafast(noise_map_1.numpy())
    plt.figure()
    plt.loglog(fpp1, label="Only double smoothed fiducial")
    plt.loglog(npp1, label="Only double smoothed noise")
    plt.legend()
    plt.savefig("/users/bsuter/Compare_PP/Single_Double_Smoothed.png")
    
    hp.mollview(fiducial_map_1.numpy(), nest=True, title="Double Smoothed Fiducial Map")
    plt.savefig("/users/bsuter/Compare_PP/fiducial_map_1.png")

    hp.mollview(noise_map_1.numpy(), nest=True, title="Double Smoothed Noise Map")
    plt.savefig("/users/bsuter/Compare_PP/noise_map_1.png")

    full_double_smoothed_1 = np.full_like(fiducial_map_1.numpy(), hp.UNSEEN)
    for idx, val in enumerate(fiducial_map_1.numpy()):
        if val < -1e25:
            continue
        else:
            full_double_smoothed_1[idx] = fiducial_map_1.numpy()[idx] + noise_map_1.numpy()[idx]
    hp.mollview(full_double_smoothed_1, nest=True, title="Full Double Smoothed map")
    plt.savefig("/users/bsuter/Compare_PP/full_double_smoothed_1.png")

    full_double_smoothed_1 = hp.reorder(full_double_smoothed_1, n2r=True)
    pp_double_smoothed_1 = hp.anafast(full_double_smoothed_1)

    plt.figure()
    plt.loglog(pp_double_smoothed_1, label="Full Double Smoothed 1")
    plt.legend()
    plt.savefig("/users/bsuter/Compare_PP/only_DoubleSmoothedPP.png")

    fiducial_map_2 = fid_maps[0][1]
    noise_map_2 = noise_maps[0][1]

    full_double_smoothed_2 = tf.math.add(fiducial_map_2, noise_map_2).numpy()
    full_double_smoothed_2 = full_double_smoothed_2 - np.mean(full_double_smoothed_2)
    full_double_smoothed_2 = hp.reorder(full_double_smoothed_2, n2r=True)
    full_double_smoothed_2[full_double_smoothed_2 < -1.2e20] = 0
    pp_double_smoothed_2 = hp.anafast(full_double_smoothed_2)

    fiducial_map_3 = fid_maps[0][2]
    noise_map_3 = noise_maps[0][2]

    full_double_smoothed_3 = tf.math.add(fiducial_map_3, noise_map_3).numpy()
    full_double_smoothed_3 = full_double_smoothed_3 - np.mean(full_double_smoothed_3)
    full_double_smoothed_3 = hp.reorder(full_double_smoothed_3, n2r=True)
    full_double_smoothed_3[full_double_smoothed_3 < -1.2e20] = 0
    pp_double_smoothed_3 = hp.anafast(full_double_smoothed_3)

    fiducial_map_4 = fid_maps[0][3]
    noise_map_4 = noise_maps[0][3]

    full_double_smoothed_4 = tf.math.add(fiducial_map_4, noise_map_4).numpy()
    full_double_smoothed_4 = full_double_smoothed_4 - np.mean(full_double_smoothed_4)
    full_double_smoothed_4 = hp.reorder(full_double_smoothed_4, n2r=True)
    full_double_smoothed_4[full_double_smoothed_4 < -1.2e20] = 0
    pp_double_smoothed_4 = hp.anafast(full_double_smoothed_4)

    dir = "/scratch/snx3000/bsuter/Maps"
    all_ids = np.load(os.path.join(dir, "Map_ids.npy"))

    plt.figure()
    for id in all_ids[-1:]:
        for tomo in range(1, 5):
            try:
                map = np.load(os.path.join(dir, "FullMaps", f"Map_Om=0.26_s8=0.84_tomo={tomo}_id={id}.npy"))
                map = hp.reorder(map, n2r=True)
            except FileNotFoundError:
                continue
            ps = hp.anafast(map)
            plt.loglog(ps, label=f"Fiducial Map tomo={tomo}", alpha=0.4)
    plt.loglog(pp_double_smoothed_1, label="Double Smoothed Map tomo=1", alpha=0.4)
    plt.loglog(pp_double_smoothed_2, label="Double Smoothed Map tomo=2", alpha=0.4)
    plt.loglog(pp_double_smoothed_3, label="Double Smoothed Map tomo=3", alpha=0.4)
    plt.loglog(pp_double_smoothed_4, label="Double Smoothed Map tomo=4", alpha=0.4)
    plt.title("PowerSpectrum comparison: Pipeline Map vs. Double Smoothed Map")
    print("Saving Figure")
    plt.legend()
    plt.savefig("/users/bsuter/Compare_PP/PowerSpectrum_comparison.png")

    map = np.load(os.path.join(dir, "FullMaps", f"Map_Om=0.26_s8=0.84_tomo=1_id={id}.npy"))
    hp.mollview(map, nest=True, title="Fiducial Map")
    plt.savefig("/users/bsuter/Compare_PP/FiducialMap.png")
