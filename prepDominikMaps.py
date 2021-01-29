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


def _rotate_map(map, ctx):
    """
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


def map_manager(idx, tomo, ctx):
    for mode in ["E"]:
        all_cuts_name = f"NOISE_mode={mode}_noise={idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy"
        try:
            all_cuts = np.load(os.path.join(ctx["noise_dir"], all_cuts_name))
        except (IOError, ValueError) as e:
            logger.info(
                f"Error while loading NOISE_mode={mode}_noise={idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy " +
                "Adding to failed list")
            logger.info(e)
            continue

        tmp_cuts = []
        for cut in range(len(all_cuts)):
            ctx["index_counter"] = cut
            rotated_map = _rotate_map(all_cuts[cut], ctx)
            tmp_cuts.append(rotated_map)
            del(rotated_map)
        logger.info(f"Collected {len(tmp_cuts)}/{len(all_cuts)} cuts.")
        all_cuts = np.asarray(tmp_cuts)
        del(tmp_cuts)

        if not os.getenv("DEBUG", False):
            np.save(os.path.join(ctx["noise_dir"], "Rotated_" + all_cuts_name), all_cuts)
            if os.path.isfile(os.path.join(ctx["noise_dir"], all_cuts_name)):
                os.remove(os.path.join(ctx["noise_dir"], all_cuts_name))
        else:
            SCRATCH_path = os.path.join(os.path.expandvars("$SCRATCH"), "Rotated_" + all_cuts_name)
            logger.debug(f"Debug-Mode: Saving first noise file to {SCRATCH_path} then aborting.")
            np.save(SCRATCH_path, all_cuts)
            sys.exit(0)


def main(job_index):
    tomo = int(job_index)
    ctx = {
        "NSIDE": 1024,
        "NSIDE_OUT": 32,
        "alpha_rotations": [0., 1.578, 3.142, 4.712, 0., 1.578, 3.142, 4.712],
        "dec_rotations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mirror": [False, False, False, False, True, True, True, True],
        "down_sample": 512,
        "noise_dir": "/cluster/work/refregier/besuter/data/NoiseMaps"
    }

    for idx in range(2000):
        map_manager(idx, tomo, ctx)


if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(args[0])

    main(job_index)