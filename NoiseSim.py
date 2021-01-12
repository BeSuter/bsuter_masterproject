import os
import sys
import logging
import numpy as np
import healpy as hp


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def load_dict(filename_):
    ret_di = np.load(filename_)
    return ret_di


def save_dict(di_, filename_):
    np.savez(filename_, **di_)


def iterative_mean_and_var(map, prev_val):
    """
    """
    new_mean = np.zeros_like(map)
    new_variance = np.zeros_like(map)
    if prev_val["count"] == 1:
        for pixel, val in enumerate(map):
            new_mean[pixel] = np.mean([val, prev_val["mean_map"][pixel]])
            new_variance[pixel] = np.var([val, prev_val["mean_map"][pixel]])
    else:
        for pixel, val in enumerate(map):
            new_mean[pixel] = (val + (prev_val["count"] * prev_val["mean_map"][pixel])) / (prev_val["count"] + 1.)
            new_variance[pixel] = (
                    ((prev_val["count"] - 1.) / prev_val["count"]) * prev_val["variance_map"][pixel]
                    + (val - prev_val["mean_map"][pixel]) ** 2 / (prev_val["count"] + 1)
            )

    return new_mean, new_variance


def simulate_noise():
    """
    Iterative computation of the mean and variance for each pixel.
    Used to generate gaussian noise on the fly during HealpyGCNN model training.

    Return: Return a dummy statistic. Maybe possible to implement a pass statement.
    """
    noise_map_dir = "/cluster/work/refregier/besuter/master_branch/data/NoiseMaps/FullNoiseMaps"
    pixel_noise_dir = "/cluster/work/refregier/besuter/master_branch/data/NoiseMaps"

    noise_map_ids = np.load("/cluster/work/refregier/besuter/master_branch/data/NoiseMaps/NoiseMap_ids.npy")
    logger.info(f"Found {len(noise_map_ids)} noise map ids")
    for tomo in range(4):
        tomo += 1
        logger.info(f"Hitting on tomo={tomo}")
        filename_ = os.path.join(pixel_noise_dir, f"NewPixelNoise_tomo={tomo}.npz")

        first_id = noise_map_ids[0]
        remaining_ids = noise_map_ids[1::]

        first_map = np.load(os.path.join(noise_map_dir, f"NoiseMap_tomo={tomo}_id={first_id}.npy"))
        first_map = first_map[first_map > hp.UNSEEN]
        noise_data_map = {"mean_map": first_map,
                          "variance_map": np.zeros_like(first_map),
                          "count": 1}

        for id in remaining_ids:
            map = np.load(os.path.join(noise_map_dir, f"NoiseMap_tomo={tomo}_id={id}.npy"))
            noise_map = map[map > hp.UNSEEN]
            new_mean, new_variance = iterative_mean_and_var(noise_map, noise_data_map)
            noise_data_map["mean_map"] = new_mean
            noise_data_map["variance_map"] = new_variance
            noise_data_map["count"] = noise_data_map["count"] + 1

        logger.info(f"Saving NewPixelNoise_tomo={tomo}")
        save_dict(noise_data_map, filename_)