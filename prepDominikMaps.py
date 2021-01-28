import os
import sys
import logging
import argparse
import numpy as np
import healpy as hp

from memory_profiler import profile
from DeepSphere.utils import extend_indices
from estats.catalog import catalog

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

@profile
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

@profile
def main(job_index, debug=False):
    tomo = int(job_index)
    ctx = {
        "NSIDE": 1024,
        "NSIDE_OUT": 32,
        "alpha_rotations": [0., 1.578, 3.142, 4.712, 0., 1.578, 3.142, 4.712],
        "dec_rotations": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mirror": [False, False, False, False, True, True, True, True],
        "down_sample": 512
    }
    noise_dir = "/cluster/work/refregier/besuter/data/NoiseMaps"

    # Load file containing noise_id of corrupted files
    try:
        corrupted = np.load(os.path.join(noise_dir, f"corrupted_files_tomo={tomo}.npy"))
    except IOError:
        corrupted = np.zeros(0, dtype=np.int32)

    for idx in range(2000):
        for mode in ["E"]:
            all_cuts_name = f"NOISE_mode={mode}_noise={idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy"
            try:
                all_cuts = np.load(os.path.join(noise_dir, all_cuts_name))
            except IOError:
                logger.info(f"Error while loading NOISE_mode={mode}_noise={idx}_stat=GetSmoothedMap_tomo={tomo}x{tomo}.npy " +\
                            "Adding to failed list")
                corrupted = np.append(corrupted, np.array([idx]))
                continue

            tmp_cuts = []
            for cut in range(len(all_cuts)):
                ctx["index_counter"] = cut
                tmp_cuts.append(_rotate_map(all_cuts[cut], ctx))
            all_cuts = np.asarray(tmp_cuts)

            if not debug:
                np.save(os.path.join(noise_dir, "Rotated_" + all_cuts_name), all_cuts)
                if os.path.isfile(os.path.join(noise_dir, all_cuts_name)):
                    os.remove(os.path.join(noise_dir, all_cuts_name))
            else:
                SCRATCH_path = os.path.join(os.path.expandvars("$SCRATCH"), "Rotated_" + all_cuts_name)
                logger.debug(f"Debug-Mode: Saving first noise file to {SCRATCH_path} then aborting.")
                np.save(SCRATCH_path, all_cuts)
                sys.exit(0)
    np.save(os.path.join(noise_dir, f"corrupted_files_tomo={tomo}.npy"), corrupted)


if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(args[0])

    main(job_index)