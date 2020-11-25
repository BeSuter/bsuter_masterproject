import os
import sys
import logging
import numpy as np
import healpy as hp

from DeepSphere.utils import extend_indices
from estats.catalog import catalog

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


def _rotate_maps(map, ctx, downsampling=False):
    """
    Will rotate maps into position of MASK/TRIMMED-MASK_cut=0 and adjust for equal size
    :param map: A dictionary of Healpix convergence maps
    :param ctx: Context instance
    :returns dataset: tf.data.dataset containing the mask of the rotated maps
             rotated_indices_ext: np.array containing the healpy indices of interest in nested order.
    """
    indices = np.arange(len(map))[map > hp.UNSEEN]
    delta, alpha = hp.pix2ang(ctx["NSIDE"], indices)

    mock_cat = catalog(alphas=alpha,
                       deltas=delta,
                       degree=False,
                       colat=True)
    alpha, delta = mock_cat._rotate_coordinates(
        alpha_rot=2 * np.pi -
                  ctx["alpha_rotations"][ctx["index_counter"]],
        delta_rot=2 * np.pi -
                  ctx["dec_rotations"][ctx["index_counter"]],
        mirror=ctx["mirror"][ctx["index_counter"]])
    pix = mock_cat._pixelize(alpha, delta)
    rotated_indices_ext = extend_indices(
        indices=pix,
        nside_in=ctx["NSIDE"],
        nside_out=ctx["NSIDE_OUT"],
        nest=False)
    rotated_map = np.full_like(map, hp.UNSEEN)
    zero_padding = np.zeros_like(map)
    rotated_map[rotated_indices_ext] = zero_padding[
        rotated_indices_ext]
    rotated_map[pix] = map[indices]

    if downsampling:
        rotated_map = hp.ud_grade(rotated_map, downsampling, order_out="NESTED")
    else:
        rotated_map = hp.reorder(rotated_map, r2n=True)

    return rotated_map


def LSF_prep_map(job_index,
                    path="/cluster/work/refregier/besuter/data/LIGHTCONES",
                    mask_path = "/cluster/work/refregier/besuter/data/mask",
                    target="NGSF_LIGHTCONES",
                    downsampling=512,
                    cosmo_conf="z=0.0_0",
                    mask_count=8,
                    NSIDE=1024,
                    NSIDE_OUT=32,
                    SCRATCH=True):
    """
    This function is meant to be used in the context of lsf job array.
    """
    ctx = {
        "NSIDE": NSIDE,
        "NSIDE_OUT": NSIDE_OUT,
        "alpha_rotations": [0., 1.578, 3.142, 4.712, 0., 1.578, 3.142, 4.712],
        "dec_rotations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mirror": [False, False, False, False, True, True, True, True],
        "scales": [2.6]
    }
    # Change the cosmo_files!  --> We want to use the LIGHTCONES.
    cosmo_file = _get_config("/cluster/work/refregier/besuter/LIGHTCONE.ini")[cosmo_conf][str(
        job_index)]
    map_path = os.path.join(path, cosmo_file)
    if SCRATCH:
        target_path = os.path.join(os.path.expandvars("$SCRATCH"), target)
    else:
        target_path = target
    os.makedirs(target_path, exist_ok=True)
    try:
        k_map_unsmoothed = hp.read_map(map_path)
    except IOError as e:
        logger.critical(e)
        sys.exit(1)

    for cut in range(mask_count):
        ctx["index_counter"] = cut
        logger.info(
            f"Applying mask cut={cut} and rotating map back.")
        for scale in ctx["scales"]:
            k_map = hp.sphtfunc.smoothing(
                k_map_unsmoothed, fwhm=np.radians(float(scale) / 60.))
            mask = hp.read_map(os.path.join(mask_path, f"TRIMMED-MASK_cut={cut}_tomo=1.fits"))
            mask = np.logical_not(mask)
            k_map[mask] = hp.pixelfunc.UNSEEN
            k_map = _rotate_maps(k_map, ctx, downsampling=downsampling)

            outfile_name = cosmo_file[:-5] + f"_cut={cut}_scale={scale}.npy"
            final_target_path = os.path.join(target_path, outfile_name)
            np.save(final_target_path, k_map)

if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(args[0])
    LSF_prep_map(job_index)