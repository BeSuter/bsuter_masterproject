import os
import sys
import logging
import numpy as np
import healpy as hp

from DeepSphere.utils import extend_indices
from estats.catalog import catalog

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


def _rotate_map(map, ctx):
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
    del(mock_cat)

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

    if ctx["down_sample"]:
        rotated_map = hp.ud_grade(rotated_map,
                                  ctx["down_sample"],
                                  order_out="NESTED")
    else:
        rotated_map = hp.reorder(rotated_map, r2n=True)

    return rotated_map


def generate_rotated_fiducial_maps(noise_idx, tomo, real_idx, ctx):
    for mode in ["E"]:
        map_name = f"SIM_IA=0.0_Om=0.26_eta=0.0_m=0.0_mode={mode}_noise={noise_idx}_s8=0.84_stat=" + \
                   f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy"
        try:
            map = np.load(os.path.join(ctx["fiducial_dir"], map_name))
        except (IOError, ValueError) as e:
            logger.debug(
                f"Error while loading SIM_IA=0.0_Om=0.26_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8=0.84_stat=" +
                f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy " +
                "Adding to failed list")
            logger.debug(e)
            continue

        for cut in range(len(map)):
            ctx["index_counter"] = cut
            rotated_map = _rotate_map(map[cut], ctx)
            yield rotated_map


def generate_rotated_noise_maps(noise_idx, tomo, ctx):
    for mode in ["E"]:
        all_cuts_name = f"NOISE_mode={mode}_noise={noise_idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy"
        try:
            map = np.load(os.path.join(ctx["noise_dir"], all_cuts_name))
        except (IOError, ValueError) as e:
            logger.debug(
                f"Error while loading NOISE_mode={mode}_noise={noise_idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy " +
                "Adding to failed list")
            logger.debug(e)
            continue

        for cut in range(len(map)):
            ctx["index_counter"] = cut
            rotated_map = _rotate_map(map[cut], ctx)
            yield rotated_map


def generate_rotated_grid_maps(noise_idx, tomo, real_idx, omega_m, sigma_8, ctx):
    for mode in ["E"]:
        map_name = f"SIM_IA=0.0_Om={omega_m}_eta=0.0_m=0.0_mode={mode}_noise={noise_idx}_s8={sigma_8}_stat=" + \
                   f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy"
        try:
            map = np.load(os.path.join(ctx["grid_dir"], map_name))
        except (IOError, ValueError) as e:
            logger.debug(
                f"Error while loading SIM_IA=0.0_Om={omega_m}_eta=0.0_m=0.0_mode={mode}_noise={noise_idx}" +
                f"_s8={sigma_8}_stat=GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy")
            logger.debug(e)
            continue

        for cut in range(len(map)):
            ctx["index_counter"] = cut
            rotated_map = _rotate_map(map[cut], ctx)
            yield rotated_map


def map_manager(noise_idx, tomo, ctx, real_idx="Undefined", cosmo="Undefined"):
    if ctx["MAP_TYPE"] == "noise":
        map_generator = generate_rotated_noise_maps(noise_idx, tomo, ctx)
        map_name = f"NOISE_mode=E_noise={noise_idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy"
        logger_info = f"map with tomographic_bin={tomo}x{tomo} and noise_idx={noise_idx}"
        target = ctx["noise_dir"]
    elif ctx["MAP_TYPE"] == "fiducial":
        assert real_idx != "Undefined", "Pleas pass the realisation index of the map when using the fiducial mode"
        map_generator = generate_rotated_fiducial_maps(noise_idx, tomo, real_idx, ctx)
        map_name = f"SIM_IA=0.0_Om=0.26_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8=0.84_stat=" + \
                   f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy"
        logger_info = f"map with tomographic_bin={tomo}x{tomo}, noise={noise_idx} and realisation={real_idx}"
        target = ctx["fiducial_dir"]
    elif ctx["MAP_TYPE"] == "grid":
        assert real_idx != "Undefined", "Pleas pass the realisation index of the map when using the grid mode"
        assert cosmo != "Undefined", "Pleas pass the cosmology of the map when using the grid mode"
        omega_m = cosmo[0]
        sigma_8 = cosmo[1]
        map_generator = generate_rotated_grid_maps(noise_idx, tomo, real_idx, omega_m, sigma_8, ctx)
        map_name = f"SIM_IA=0.0_Om={omega_m}_eta=0.0_m=0.0_mode=E_noise={noise_idx}_s8={sigma_8}_stat=" + \
                   f"GetSmoothedMap_tomo={tomo}x{tomo}_z=0.0_{real_idx}.npy"
        logger_info = f"map with cosmology 0m={omega_m}, s8={sigma_8}, tomographic_bin={tomo}x{tomo}," + \
                      f"noise={noise_idx} and realisation={real_idx}"
        target = ctx["grid_dir"]

    tmp_cuts = []
    for rotated_map in map_generator:
        tmp_cuts.append(rotated_map)
    logger.info(f"Collected {len(tmp_cuts)} cuts from " + logger_info)
    all_cuts = np.asarray(tmp_cuts)
    del(tmp_cuts)

    if not os.getenv("DEBUG", False):
        np.save(os.path.join(target, "Rotated_" + map_name), all_cuts)
        if os.path.isfile(os.path.join(target, map_name)):
            os.remove(os.path.join(target, map_name))
    else:
        SCRATCH_path = os.path.join(os.path.expandvars("$SCRATCH"), "Rotated_" + map_name)
        logger.debug(f"Debug-Mode: Saving first rotated map file to {SCRATCH_path} then aborting.")
        np.save(SCRATCH_path, all_cuts)
        sys.exit(0)


def main(job_index, MAP_TYPE):
    tomo = int(job_index)
    ctx = {
        "NSIDE": 1024,
        "NSIDE_OUT": 32,
        "alpha_rotations": [0., 1.578, 3.142, 4.712, 0., 1.578, 3.142, 4.712],
        "dec_rotations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mirror": [False, False, False, False, True, True, True, True],
        "down_sample": 512,
        "noise_dir": "/cluster/work/refregier/besuter/data/NoiseMaps",
        "fiducial_dir": "/cluster/work/refregier/besuter/data/SmoothedMaps",
        "MAP_TYPE": MAP_TYPE
    }

    if ctx["MAP_TYPE"] == "noise":
        for noise_idx in range(2000):
            map_manager(noise_idx, tomo, ctx)
    elif ctx["MAP_TYPE"] == "fiducial":
        for noise_idx in range(10):
            for real_idx in range(50):
                map_manager(noise_idx, tomo, ctx, real_idx=real_idx)
    elif ctx["MAP_TYPE"] == "grid":
        cosmo_file = os.path.join("/cluster/work/refregier/besuter/data", "cosmo.par")
        all_cosmologies = np.genfromtxt(cosmo_file)
        for cosmology in all_cosmologies:
            cosmo = (cosmology[0], cosmology[6])
            for noise_idx in range(10):
                for real_idx in range(50):
                    map_manager(noise_idx, tomo, ctx, real_idx=real_idx, cosmo=cosmo)


if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(args[0])
    MAP_TYPE = str(args[1])

    main(job_index, MAP_TYPE)