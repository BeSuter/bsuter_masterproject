import os
import collections
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


def _l2_norm(x, y):
    """
    :params x: ndarray
    :params y: ndarray
    """
    l2 = collections.OrderedDict()
    for i in range(len(x)):
        try:
            l2[(y[i][0], y[i][1])].append(np.sum(np.power((x[i] - y[i]), 2)))
        except KeyError:
            l2[(y[i][0], y[i][1])] = [np.sum(np.power((x[i] - y[i]), 2))]
    l2_mean = []
    for key in l2.keys():
        l2_mean.append(sum(l2[key]) / len(l2[key]))

    return np.asarray(l2_mean)


class L2ColorPlot:
    def __init__(self, target=None, HOME=True):
        """
        ToDo: Implement error handling if no target is provided
        """
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.add_axes([0.1, 0.35, 0.8, 0.6],
                          ylabel="Sigma_8",
                          xlabel="Omega_M",
                          title="L2 Norm prediction to label")

        if HOME:
            tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots",
                                    "L2_color_plot")
            os.makedirs(tmp_path, exist_ok=True)
            self.file_path = os.path.join(
                tmp_path, f"L2ColorPlot_date_time={date_time}.png")
        elif target:
            os.makedirs(target, exist_ok=True)
            self.file_path = os.path.join(
                target, f"L2ColorPlot_date_time={date_time}.png")

    def add_to_plot(self, predictions, labels):
        """
            :params predictions: ndarray, array containing (Omega_M, Sigma_8) ndarray for all predictions
            :params labels: ndarray, array containing (Omega_M, Sigma_8) ndarray for labels corresponding to the predictions

            returns:
            """
        print("Plotting")
        l2_values = _l2_norm(predictions, labels)

        cm = plt.cm.get_cmap("magma")
        sc = plt.scatter(np.unique(labels[:, 0]),
                         np.unique(labels[:, 1]),
                         s=100,
                         c=l2_values,
                         cmap=cm,
                         edgecolors='black')
        self.fig.colorbar(sc)

    def save_plot(self):
        print("Saving")
        self.fig.savefig(self.file_path)