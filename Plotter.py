import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

def _l2_norm(x, y):
    """
    :params x: ndarray
    :params y: ndarray
    """
    l2 = []
    for i in range(len(x)):
        l2.append(np.sum(np.power((x[i] - y[i]), 2)))
    return np.asarray(l2)

def l2_color_plot(predictions, labels, target=None, HOME=True):
    """
    :params predictions: ndarray, array containing (Omega_M, Sigma_8) ndarray for all predictions
    :params labels: ndarray, array containing (Omega_M, Sigma_8) ndarray for labels corresponding to the predictions

    returns:
    """
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
    if HOME:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", "L2_color_plot")
        os.makedirs(tmp_path, exist_ok=True)
        file_name = f"date_time={date_time}_predictions={len(predictions)}.png"
        file_path = os.path.join(tmp_path, file_name)
    else:
        os.makedirs(target, exist_ok=True)
        file_name = f"date_time={date_time}_predictions={len(predictions)}.png"
        file_path = os.path.join(target, file_name)
    l2_values = _l2_norm(predictions, labels)

    fig = plt.figure(figsize=(12, 8))
    fig_ax = fig.add_axes([0.1, 0.35, 0.8, 0.6],
                          ylabel="Sigma_8",
                          xlabel="Omega_M",
                          title="L2 Norm prediction to label")
    cm = plt.cm.get_cmap("magma")
    sc = plt.scatter(predictions[:,0], predictions[:,1], s=100, c=l2_values, cmap=cm, edgecolors='black')
    fig.colorbar(sc)
    fig.savefig(file_path)

