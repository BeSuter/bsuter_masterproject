import os
import collections
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from datetime import datetime


def noise_plotter(noise, indices_ext, nside, tomo_num=4, target=None):
    date_time = datetime.now().strftime("%m-%d-%Y")
    total_noise_map = np.full(hp.nside2npix(nside), hp.UNSEEN)
    for tomo in range(tomo_num):
        for idx, pixel in enumerate(indices_ext):
            total_noise_map[pixel] = noise[0, idx, tomo]

        hp.mollview(total_noise_map, nest=True, title=f"Noise only, tomographic bin {tomo}")

        if target:
            os.makedirs(target, exist_ok=True)
            file_path = os.path.join(target,
                                     f"Noise_tomo={tomo}_date_time={date_time}.plt")
        else:
            tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", "Noise")
            os.makedirs(tmp_path, exist_ok=True)
            file_path = os.path.join(tmp_path,
                                     f"Noise_tomo={tomo}_date_time={date_time}.plt")
        plt.savefig(file_path)


def S8plot(data, label, target=None, epoch=None):
    """
    :params data: OrderedDict:
    :params label: str
    """
    date_time = datetime.now().strftime("%m-%d-%Y")

    S8_values = []
    prediction_values = []
    error_values = []
    for key, value in data.items():
        S8_values.append(key[1]*np.sqrt(key[0]/0.3))
        error_values.append(np.std(value, axis=0))
        prediction_values.append(np.mean(value, axis=0))

    plt.figure(figsize=(12,8))
    plt.errorbar(S8_values, prediction_values, yerr=error_values, marker='o', linestyle='')
    plt.xlabel("sigma8*sqrt(Om/0.3)")
    plt.ylabel(label)

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"S8plot_{label}_date_time={date_time}")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", "S8plot",
                                label)
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path,
                                 f"S8plot_date_time={date_time}")
    if epoch:
        epoch -= 1
        file_path += f"_epoch={epoch}.png"
    else:
        file_path += ".png"
    plt.savefig(file_path)


def stats(data, label, target=None, epoch=None):
    """
    :params data: ndarray
    :params label: str
    """
    date_time = datetime.now().strftime("%m-%d-%Y")

    plt.figure(figsize=(12,8))
    plt.plot(data, label=label)
    plt.title(f"Monitoring {label}")
    plt.legend()

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"Monitoring_{label}_date_time={date_time}")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", "Monitoring",
                                label)
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path,
                                 f"Monitoring_date_time={date_time}")
    if epoch:
        epoch -= 1
        file_path += f"_epoch={epoch}.png"
    else:
        file_path += ".png"
    plt.savefig(file_path)


def histo_plot(data, label, target=None, epoch=None):
    """
    :params data: ndarray
    :params label: str
    """
    date_time = datetime.now().strftime("%m-%d-%Y")

    plt.figure(figsize=(12,8))
    plt.hist(data, bins=75, label=label, range=(-0.275,0.275), density=True)
    plt.title("Histogram of prediction - label")
    plt.legend()

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"HistoPlot_for_{label}_date_time={date_time}")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots",
                                label)
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path,
                                 f"HistoPlot_{label}")
    if epoch:
        epoch -=1
        file_path += f"_epoch={epoch}.png"
    else:
        file_path += ".png"
    plt.savefig(file_path)


def _l2_norm_and_labels_ordered(predictions, labels):
    """
    :params predictions: ndarray
    :params labels: ndarray
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
    labels = []
    for key in l2.keys():
        labels.append(key)
        l2_mean.append(sum(l2[key]) / len(l2[key]))

    return np.asarray(l2_mean), np.asarray(labels)


def l2_color_plot(predictions, labels, target=None, epoch=None):
    """Plots mean L2 Norm between all predictions and unique labels"""
    date_time = datetime.now().strftime("%m-%d-%Y")
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
        if epoch:
            epoch -= 1
            file_path = os.path.join(tmp_path,
                                     f"L2ColorPlot_epoch={epoch}.png")
        else:
            file_path = os.path.join(tmp_path,
                                     f"L2ColorPlot.png")

    l2_values, no_duplicate_labels = _l2_norm_and_labels_ordered(
        predictions, labels)

    cm = plt.cm.get_cmap("magma")
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
            tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots")
            os.makedirs(tmp_path, exist_ok=True)
            self.file_path = os.path.join(tmp_path, plot_name + f"_date_time={date_time}")

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
