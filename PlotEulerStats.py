import os
import re
import sys
import argparse
import logging
import collections
import numpy as np
import healpy as hp

from datetime import datetime
from Plotter import PredictionLabelComparisonPlot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def plot_euler_stats():

    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    STATS_DIR = "/cluster/work/refregier/besuter/data/STATS"
    cosmo_file = os.path.join("/cluster/work/refregier/besuter/data", "cosmo.par")
    all_cosmologies = np.genfromtxt(cosmo_file)

    om_pred_check = PredictionLabelComparisonPlot(
        "Omega_m",
        layer="STATS",
        noise_type="Pipeline",
        start_time=date_time,
        evaluation="Evaluation")
    s8_pred_check = PredictionLabelComparisonPlot(
        "Sigma_8",
        layer="STATS",
        noise_type="Pipeline",
        start_time=date_time,
        evaluation="Evaluation")

    for cosmology in all_cosmologies:
        cosmo = [cosmology[0], cosmology[6]]
        logger.info(f"Looking at cosmology {cosmo}")
        f_name = f"SIM_IA=0.0_Om={cosmo[0]}_eta=0.0_m=0.0_mode=E_s8={cosmo[1]}_stat=FullHealpyGCNN_tomo=1x1_z=0.0.npy"
        predictions = np.load(os.path.join(STATS_DIR, f_name))

        for ii, prediction in enumerate(predictions.numpy()):
            om_pred_check.add_to_plot(prediction[0], cosmo[0])
            s8_pred_check.add_to_plot(prediction[1], cosmo[1])

    logger.info("Saving plots")
    om_pred_check.save_plot()
    s8_pred_check.save_plot()


if __name__ == '__main__':
    plot_euler_stats()
