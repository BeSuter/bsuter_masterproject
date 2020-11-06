import os
import collections
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


def _l2_norm(predictions, labels):
    """
    :params x: ndarray
    :params y: ndarray
    """
    l2 = collections.OrderedDict()
    for i in range(len(predictions)):
        try:
            l2[(labels[i][0], labels[i][1])].append(
                np.sum(np.power((predictions[i] - labels[i]), 2)))
        except KeyError:
            l2[(labels[i][0], labels[i][1])] = [
                np.sum(np.power((predictions[i] - labels[i]), 2))
            ]
    l2_mean = []
    for key in l2.keys():
        l2_mean.append(sum(l2[key]) / len(l2[key]))

    return np.asarray(l2_mean)


def _stupid_way_of_preserving_tuple_order(labels):
    no_duplicate_labels = collections.OrderedDict()
    for i in range(len(labels)):
        no_duplicate_labels[(labels[i][0], labels[i][1])] = None
    labels = []
    for key in no_duplicate_labels.keys():
        labels.append(key)
    return np.asarray(labels)


def l2_color_plot(predictions, labels, target=None):
    """Plots mean L2 Norm between all predictions and unique labels"""
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
    fig = plt.figure(figsize=(12, 8))
    fig.add_axes([0.1, 0.35, 0.8, 0.6],
                 ylabel="Sigma_8",
                 xlabel="Omega_M",
                 title="L2 Norm prediction to label")

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"L2ColorPlot_date_time={date_time}.png")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots",
                                "L2_color_plot")
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path,
                                 f"L2ColorPlot_date_time={date_time}.png")

    l2_values = _l2_norm(predictions, labels)

    cm = plt.cm.get_cmap("magma")
    no_duplicate_labels = _stupid_way_of_preserving_tuple_order(labels)
    print(np.asarray(no_duplicate_labels))
    sc = plt.scatter(no_duplicate_labels[:, 0],
                     no_duplicate_labels[:, 1],
                     s=100,
                     c=l2_values,
                     cmap=cm,
                     edgecolors='black')
    fig.colorbar(sc)
    fig.savefig(file_path)


class PredictionLabelComparisonPlot:
    def __init__(self, topic, target=None, **kwargs):
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.fig = plt.figure(figsize=(12, 8))
        self.fig_ax = self.fig.add_axes(
            [0.1, 0.35, 0.8, 0.6],
            ylabel="Predictions",
            xlabel="Labels",
            title=f"{topic} prediction compared to Label")
        plot_name = f"{topic}_comparison"
        batch_size = kwargs.pop("batch_size", None)
        if batch_size:
            plot_name += f"_batch={batch_size}"
        shuffle_size = kwargs.pop("shuffle_size", None)
        if shuffle_size:
            plot_name += f"_shuffle={shuffle_size}"
        epochs = kwargs.pop("epochs", None)
        if epochs:
            plot_name += f"_epochs={epochs}"
        plot_name += ".png"

        if target:
            os.makedirs(target, exist_ok=True)
            self.file_path = os.path.join(target, plot_name)
        else:
            tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots",
                                    date_time)
            os.makedirs(tmp_path, exist_ok=True)
            self.file_path = os.path.join(tmp_path, plot_name)

    def add_to_plot(self, predictions, labels):
        self.fig_ax.plot(labels,
                         predictions,
                         marker='o',
                         alpha=0.5,
                         ls='',
                         color="blue")

    def save_plot(self):
        xmin, xmax = self.fig_ax.axis()[:2]
        true_line = np.linspace(xmin, xmax, 100)
        self.fig_ax.plot(true_line, true_line, alpha=0.3, color="red")

        self.fig.savefig(self.file_path)
