import os
import re
import sys
import uuid
import logging
import configparser
import numpy as np
import healpy as hp
import tensorflow as tf

from DeepSphere import data
from DeepSphere.utils import extend_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def _get_config(filename):
    parser = configparser.ConfigParser()
    parser.read(filename)
    return parser._sections


def _is_power_of_2(n):
    """
    Allows zero to pass as a power of 2
    """
    return n & (n - 1) == 0


def _label_finder(str):
    Om_label = re.search(r"(?<=Om=).+(?=_num)", str).group(0)
    s8_label = re.search(r"(?<=s8=).+(?=_total)", str).group(0)
    return (float(Om_label), float(s8_label))


def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=).+(?=_)", str).group(0)
    shapes = [(int(item.split(",")[0]), int(item.split(",")[1])) for item in shapes.split("&")]
    return shapes


def _get_euler_angles(configuration="Default"):
    all_angles = _get_config("./euler_angles.ini")[configuration]
    unpacked_angles = []
    for angles in all_angles.values():
        unpacked_angles.append(np.asarray(angles.split(",")).astype(int))

    return unpacked_angles


def _get_masked_map(map, mask_pixels, padded_pixels):
    masked_map = np.full_like(map, hp.UNSEEN)
    zero_padding = np.zeros_like(map)
    masked_map[padded_pixels] = zero_padding[padded_pixels]
    masked_map[mask_pixels] = map[mask_pixels]
    return masked_map


def _write_tfr(serialized_dump, target_path):
    batch_size = len(serialized_dump)
    with tf.io.TFRecordWriter(target_path) as writer:
        for index, serialized_example in enumerate(serialized_dump):
            logger.info(f"Writing serialized_example {index + 1}/{batch_size}")
            writer.write(serialized_example)


def _reorder_map(map, downsampling=False):
    if downsampling:
        map = hp.ud_grade(map, downsampling, order_out="NESTED")
    else:
        map = hp.reorder(map, r2n=True)
    return map


def get_mask(configuration="Default"):
    """
    At the moment only returns one shape...
    Also, the user is trusted to use correct nside depending on the maps
    """
    mask = _get_config("./mask.ini")[configuration]

    nside = int(mask["nside"])
    nside_out = int(mask["nside_out"])

    assert _is_power_of_2(
        nside) == True, f"downsampling={nside} is not a power of two"
    assert _is_power_of_2(
        nside_out) == True, f"downsampling={nside_out} is not a power of two"

    vec = np.asarray(mask["vec"].split(",")).astype(float)
    radius = float(mask["radius"])

    n_pix = hp.nside2npix(nside)
    hp_map = np.zeros(n_pix)

    # create a mask
    pix = hp.query_disc(nside=nside, vec=vec, radius=radius)
    hp_map[pix] = 1.0

    indices = np.arange(n_pix)[hp_map > 0.5]
    indices_ext = extend_indices(indices=indices,
                                 nside_in=nside,
                                 nside_out=nside_out,
                                 nest=False)

    return indices, indices_ext


def rotate_map(map, angles, eulertype="ZYX"):
    n_pix = len(map)
    nside = int(np.sqrt((n_pix / 12.0)))
    rotator = hp.rotator.Rotator(rot=angles, eulertype=eulertype)
    theta, phi = hp.pix2ang(ipix=np.arange(hp.nside2npix(nside=nside)),
                            nside=nside)
    theta_new, phi_new = rotator(theta, phi)
    interp_pix, interp_weights = hp.get_interp_weights(nside=nside,
                                                       theta=theta_new,
                                                       phi=phi_new)
    rotated_map = np.zeros(n_pix)
    for i in range(n_pix):
        rotated_map[i] += np.sum(interp_weights[:, i] * map[interp_pix[:, i]])

    return rotated_map


def tfrecord_writer(path,
                    target="TFRecords",
                    file_count=32,
                    euler_angles="Default",
                    mask="Default",
                    downsampling=False,
                    SCRATCH=True):
    """
    Writes kappa_maps in 'path' to a TFRecord file. The kappa_maps are reordered to 'NESTED' and will be batched such
    that there will be #file_count TFRecord files created.
    :param path: string, absolute path to data directory
    :param target: string, path to directory where TFRecord files will be written to
    :param file_count: int, number of TFRecord files created, default is 32
    :euler_angels: string, corresponding section in ./euler_angels.ini, set to False if not applicable
    :param mask: string, corresponding section in ./mask.ini, only applicable if euler_angles is not False
    :param downsampling: int, The desired nside of the output maps (optional)
    :param SCRATCH: bool, if True we have $SCRATCH/target (optional)
    """
    assert _is_power_of_2(
        downsampling
    ) == True, f"downsampling={downsampling} is not a power of two"

    if SCRATCH:
        target = os.path.join(os.path.expandvars("$SCRATCH"), target)
    os.makedirs(target, exist_ok=True)

    map_count = len(
        [file for file in os.listdir(path) if not file.startswith(".")])

    if euler_angles:
        all_angles = _get_euler_angles(euler_angles)
        mask_pixels, padded_pixels = get_mask(mask)
        batch_size = int(len(all_angles) * map_count / file_count)
    else:
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
                k_map = np.load(file_path)
            except IOError as e:
                logger.critical(e)
                continue
            k_map = k_map - np.mean(k_map)

            if euler_angles:
                for angles in all_angles:
                    rotated_k_map = rotate_map(k_map, angles)
                    masked_k_map = _get_masked_map(rotated_k_map, mask_pixels,
                                                   padded_pixels)

                    masked_k_map = _reorder_map(masked_k_map,
                                                downsampling=downsampling)

                    serialized_example_dump.append(
                        data.serialize_labeled_example(masked_k_map,
                                                       cosmology_label))

                    if len(serialized_example_dump
                           ) % batch_size == 0 and not len(
                               serialized_example_dump) == 0:
                        logger.info("Dumping maps")
                        tfrecord_name = f"kappa_map_cosmo_shapes={len(masked_k_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
                        target_path = os.path.join(target, tfrecord_name)
                        _write_tfr(serialized_example_dump, target_path)
                        serialized_example_dump.clear()
            else:
                k_map = _reorder_map(k_map, downsampling=downsampling)

                serialized_example_dump.append(
                    data.serialize_labeled_example(k_map, cosmology_label))

                if len(serialized_example_dump) % batch_size == 0 and not len(
                        serialized_example_dump) == 0:
                    logger.info("Dumping maps")
                    tfrecord_name = f"kappa_map_cosmo_shapes={len(k_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
                    target_path = os.path.join(target, tfrecord_name)
                    _write_tfr(serialized_example_dump, target_path)
                    serialized_example_dump.clear()

    if not len(serialized_example_dump) == 0:
        logger.info("Dumping remaining maps")
        tfrecord_name = f"kappa_map_cosmo_shapes={len(k_map)},{len(cosmology_label)}_total_{uuid.uuid4().hex}.tfrecord"
        target_path = os.path.join(target, tfrecord_name)
        _write_tfr(serialized_example_dump, target_path)


def LSF_map_rotator(job_index,
                    path="kappa_maps",
                    target=".cache",
                    downsampling=256,
                    cosmo_conf="Default",
                    euler_angles="Default",
                    mask="Default",
                    SCRATCH=True):
    """
    This function is meant to be used in the context of lsf job array.
    """
    cosmo_file = _get_config("./LSF_cosmo_conf.ini")[cosmo_conf][str(
        job_index)]
    if SCRATCH:
        tem_path = os.path.join(os.path.expandvars("$SCRATCH"), path)
        map_path = os.path.join(tem_path, cosmo_file)
        target_path = os.path.join(os.path.expandvars("$SCRATCH"), target)
    else:
        map_path = os.path.join(path, cosmo_file)
        target_path = target
    os.makedirs(target_path, exist_ok=True)
    try:
        k_map = np.load(map_path)
    except IOError as e:
        logger.critical(e)
        sys.exit(0)
    k_map = k_map - np.mean(k_map)

    all_angles = _get_euler_angles(euler_angles)
    mask_pixels, padded_pixels = get_mask(mask)

    for angles in all_angles:
        logger.info(
            f"Rotating map, euler_angle={angles[0]},{angles[1]},{angles[2]}")
        rotated_k_map = rotate_map(k_map, angles)
        masked_k_map = _get_masked_map(rotated_k_map, mask_pixels,
                                       padded_pixels)
        masked_k_map = _reorder_map(masked_k_map, downsampling=downsampling)
        outfile_name = cosmo_file[:-4] + f"_angles={angles[0]},{angles[1]},{angles[2]}.npy"
        final_target_path = os.path.join(target_path, outfile_name)
        np.save(final_target_path, masked_k_map)


def LSF_tfrecord_writer(job_index,
                        path=".cache",
                        target="TFRecordsMasked",
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
    if array_idx + 1 == file_count:
        start = int((array_idx - 1) * batch_size + 1)
        end = len(f_names) + 1
        name_batch = f_names[start:]
        logger.info(f"Serializing maps {start} to end")
    else:
        start = int(array_idx * batch_size)
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
        cosmo_labels = np.array([
            labels[0], -1.0, -1.0, 0.0, 0.0493, 0.6736, labels[1], 0.9649, 0.02
        ])
        serialized_example_dump.append(
            data.serialize_labeled_example(k_map, cosmo_labels))
    logger.info("Dumping maps")
    tfrecord_name = f"kappa_map_cosmo={start}-{end-1}_shapes={len(k_map)},{len(cosmo_labels)}_total_{uuid.uuid4().hex}.tfrecord"
    target_path = os.path.join(target_path, tfrecord_name)
    _write_tfr(serialized_example_dump, target_path)
    serialized_example_dump.clear()


def get_dataset(path=[]):
    if not isinstance(path, list):
        path = [path]
    all_files = []
    logger.debug(f"Adding TFRecord files from {path}")
    for pp in path:
        f_names = [
            os.path.join(pp, file) for file in os.listdir(pp)
            if file.endsswith(".tfrecord")
        ]
        all_files.extend(f_names)
    shapes = _shape_finder(all_files[0])
    dset = tf.data.TFRecordDataset(all_files)
    logger.debug(f"Created DataSet out of {len(all_files)} TFRecord files.")
    decoded_dset = data.decode_labeled_dset(dset, shapes)
    return decoded_dset
