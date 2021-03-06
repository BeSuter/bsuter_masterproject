import os
import sys
import logging
import numpy as np
import pandas as pd

from datetime import datetime
from Plotter import PredictionLabelComparisonPlot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def plot_euler_stats(STATS_DIR="/cluster/work/refregier/besuter/data/STATS", stat="FullHealpyGCNN_DS2", tomo="0x0"):

    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    cosmo_file = os.path.join("/cluster/work/refregier/besuter/data", "cosmo.par")
    all_cosmologies = np.genfromtxt(cosmo_file)

    om_pred_check = PredictionLabelComparisonPlot(
        "Omega_m",
        layer="STATS",
        noise_type="Pipeline",
        start_time=date_time,
        evaluation="Evaluation",
        evaluation_mode="average")
    s8_pred_check = PredictionLabelComparisonPlot(
        "Sigma_8",
        layer="STATS",
        noise_type="Pipeline",
        start_time=date_time,
        evaluation="Evaluation",
        evaluation_mode="average")

    for cosmology in all_cosmologies:
        cosmo = [cosmology[0], cosmology[6]]
        logger.info(f"\n Looking at cosmology {cosmo}")
        for real in range(5):
            try:
                f_name = f"SIM_IA=0.0_Om={cosmo[0]}_eta=0.0_m=0.0_mode=E_s8={cosmo[1]}_stat={stat}_tomo={tomo}_z=0.0_{real}.npy"
                predictions = np.load(os.path.join(STATS_DIR, f_name))
                logger.info(f"Found {f_name}")

                for ii, prediction in enumerate(predictions):
                    om_pred_check.add_to_plot(prediction[0], cosmo[0])
                    s8_pred_check.add_to_plot(prediction[1], cosmo[1])
            except FileNotFoundError:
                pass

    cosmo = [0.26, 0.84]
    logger.info(f"\n Looking at cosmology {cosmo}")
    for real in range(50):
        try:
            f_name = f"SIM_IA=0.0_Om={cosmo[0]}_eta=0.0_m=0.0_mode=E_s8={cosmo[1]}_stat={stat}_tomo={tomo}_z=0.0_{real}.npy"
            predictions = np.load(os.path.join(STATS_DIR, f_name))
            logger.info(f"Found {f_name}")

            for ii, prediction in enumerate(predictions):
                om_pred_check.add_to_plot(prediction[0], cosmo[0])
                s8_pred_check.add_to_plot(prediction[1], cosmo[1])
        except FileNotFoundError:
            pass

    logger.info("Saving plots")
    om_pred_check.save_plot()
    s8_pred_check.save_plot()


def generate_debug_file_names():
    path = "/cluster/work/refregier/besuter/DebugFolder"
    for real in range(50):
        for noise in range(5):
            tmp_name = f"meta_data_om=0.26_s8=0.84_noise={noise}_real={real}.csv"
            f_name = os.path.join(path, tmp_name)
            yield f_name


def plot_debug_meta_data():
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")

    om_pred_check = PredictionLabelComparisonPlot(
        "Omega_m",
        layer="DEBUG",
        noise_type="Pipeline",
        start_time=date_time,
        evaluation="Evaluation",
        evaluation_mode="average")
    s8_pred_check = PredictionLabelComparisonPlot(
        "Sigma_8",
        layer="DEBUG",
        noise_type="Pipeline",
        start_time=date_time,
        evaluation="Evaluation",
        evaluation_mode="average")

    for f_name in generate_debug_file_names():
        df = pd.read_csv(f_name)

        om_pred = df['OmPrediction']
        om_label = df['OmLabel']

        s8_pred = df['S8Prediction']
        s8_label = df['S8Label']

        om_pred_check.add_to_plot(om_pred, om_label)
        s8_pred_check.add_to_plot(s8_pred, s8_label)

    logger.info("Saving plots")
    om_pred_check.save_plot()
    s8_pred_check.save_plot()


if __name__ == '__main__':
    plot_euler_stats()
    # plot_debug_meta_data()
