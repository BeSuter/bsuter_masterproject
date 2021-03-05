import os
import collections
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime


def noise_plotter(noise,
                  indices_ext,
                  nside,
                  tomo_num=4,
                  target=None,
                  layer="Undefined_Layer",
                  noise_type="Undefinded_Noise_Type",
                  start_time="Undefined_Time",
                  evaluation=""):
    date_time = datetime.now().strftime("%m-%d-%Y")
    total_noise_map = np.full(hp.nside2npix(nside), hp.UNSEEN)
    for tomo in range(tomo_num):
        for idx, pixel in enumerate(indices_ext):
            total_noise_map[pixel] = noise[0, idx, tomo]

        hp.mollview(total_noise_map,
                    nest=True,
                    title=f"Noise only, tomographic bin {tomo + 1}")

        if target:
            os.makedirs(target, exist_ok=True)
            file_path = os.path.join(
                target, f"Noise_tomo={tomo + 1}_date_time={date_time}.plt")
        else:
            tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", evaluation,
                                    "Noise", layer, noise_type, start_time)
            os.makedirs(tmp_path, exist_ok=True)
            file_path = os.path.join(
                tmp_path, f"Noise_tomo={tomo + 1}_date_time={date_time}.pdf")
        plt.savefig(file_path)


def S8plot(data,
           label,
           target=None,
           epoch=None,
           layer="Undefined_Layer",
           noise_type="Undefinded_Noise_Type",
           start_time="Undefined_Time",
           evaluation=""):
    """
    :params data: OrderedDict:
    :params label: str
    """
    date_time = datetime.now().strftime("%m-%d-%Y")

    S8_values = []
    prediction_values = []
    error_values = []
    for key, value in data.items():
        S8_values.append(key[1] * np.sqrt(key[0] / 0.3))
        error_values.append(np.std(value, axis=0))
        prediction_values.append(np.mean(value, axis=0))

    plt.figure(num="S8plot", figsize=(12, 8))
    plt.errorbar(S8_values,
                 prediction_values,
                 yerr=error_values,
                 marker='o',
                 linestyle='')
    plt.title(f"S8plot for epoch={epoch}")
    plt.xlabel("sigma8*sqrt(Om/0.3)")
    plt.ylabel(label)

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"S8plot_{label}_date_time={date_time}")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", evaluation, "S8plot",
                                label, layer, noise_type, start_time)
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path, f"S8plot_date_time={date_time}")
    if epoch:
        epoch -= 1
        file_path += f"_epoch={epoch}.png"
    else:
        file_path += ".png"
    plt.savefig(file_path)
    plt.close("S8plot")


def stats(data,
          label,
          target=None,
          epoch=None,
          layer="Undefined_Layer",
          noise_type="Undefinded_Noise_Type",
          start_time="Undefined_Time",
          evaluation=""):
    """
    :params data: ndarray
    :params label: str
    """
    date_time = datetime.now().strftime("%m-%d-%Y")

    plt.figure(num="stats", figsize=(12, 8))
    plt.plot(data, label=label)
    plt.ylim(0, 5.)
    plt.title(f"Monitoring {label}, epoch={epoch}")
    plt.legend()

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"Monitoring_{label}_date_time={date_time}")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", evaluation,
                                "Monitoring", label, layer, noise_type,
                                start_time)
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path, f"Monitoring_date_time={date_time}")
    if epoch:
        epoch -= 1
        file_path += f"_epoch={epoch}.png"
    else:
        file_path += ".png"
    plt.savefig(file_path)
    plt.close("stats")


def histo_plot(data,
               label,
               target=None,
               epoch=None,
               layer="Undefined_Layer",
               noise_type="Undefinded_Noise_Type",
               start_time="Undefined_Time",
               evaluation=""):
    """
    :params data: ndarray
    :params label: str
    """
    date_time = datetime.now().strftime("%m-%d-%Y")

    plt.figure(num="histo_plot", figsize=(12, 8))
    plt.hist(data, bins=75, label=label, range=(-0.275, 0.275), density=True)
    plt.title(f"Histogram of prediction - label, epoch={epoch}")
    plt.legend()

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(
            target, f"HistoPlot_for_{label}_date_time={date_time}")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", evaluation, label,
                                layer, noise_type, start_time)
        os.makedirs(tmp_path, exist_ok=True)
        file_path = os.path.join(tmp_path, f"HistoPlot_{label}")
    if epoch:
        epoch -= 1
        file_path += f"_epoch={epoch}.png"
    else:
        file_path += ".png"
    plt.savefig(file_path)
    plt.close("histo_plot")


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


def l2_color_plot(predictions,
                  labels,
                  target=None,
                  epoch=None,
                  layer="Undefined_Layer",
                  noise_type="Undefinded_Noise_Type",
                  start_time="Undefined_Time",
                  evaluation=""):
    """Plots mean L2 Norm between all predictions and unique labels"""
    date_time = datetime.now().strftime("%m-%d-%Y")
    fig = plt.figure(figsize=(12, 8))
    fig.add_axes([0.1, 0.35, 0.8, 0.6],
                 ylabel="Sigma_8",
                 xlabel="Omega_M",
                 title=f"L2 Norm prediction to label for epoch {epoch}")

    if target:
        os.makedirs(target, exist_ok=True)
        file_path = os.path.join(target,
                                 f"L2ColorPlot_date_time={date_time}.png")
    else:
        tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", evaluation,
                                "L2_color_plot", layer, noise_type, start_time)
        os.makedirs(tmp_path, exist_ok=True)
        if epoch:
            epoch -= 1
            file_path = os.path.join(tmp_path,
                                     f"L2ColorPlot_epoch={epoch}.png")
        else:
            file_path = os.path.join(tmp_path, f"L2ColorPlot.png")

    l2_values, no_duplicate_labels = _l2_norm_and_labels_ordered(
        predictions, labels)

    cm = plt.cm.get_cmap("magma")
    sc = plt.scatter(no_duplicate_labels[:, 0],
                     no_duplicate_labels[:, 1],
                     s=100,
                     c=l2_values,
                     cmap=cm,
                     edgecolors='black')
    fig.colorbar(sc)
    fig.savefig(file_path)
    plt.close(fig)


class PredictionLabelComparisonPlot:
    def __init__(self, topic, target=None, **kwargs):
        epoch = kwargs.pop("epochs", None)
        if epoch:
            title = f"{topic} prediction compared to Label for epoch {epoch}"
        else:
            title = f"{topic} prediction compared to Label"
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.fig = plt.figure(figsize=(12, 8))
        self.fig_ax = self.fig.add_axes(
            [0.1, 0.35, 0.8, 0.6],
            ylabel="Predictions",
            xlabel="Labels",
            title=title)
        plot_name = f"{topic}_comparison"
        batch_size = kwargs.pop("batch_size", None)
        if batch_size:
            plot_name += f"_batch={batch_size}"
        shuffle_size = kwargs.pop("shuffle_size", None)
        if shuffle_size:
            plot_name += f"_shuffle={shuffle_size}"
        epoch = kwargs.pop("epochs", None)
        if epoch:
            plot_name += f"_epochs={epoch}"
        layer = kwargs.pop("layer", "Undefined_Layer")
        noise_type = kwargs.pop("noise_type", "Undefinded_Noise_Type")
        start_time = kwargs.pop("start_time", "Undefined_Time")
        evaluation = kwargs.pop("evaluation", "")

        self.evaluation_mode = kwargs.pop("evaluation_mode", None)
        if self.evaluation_mode:
            assert self.evaluation_mode == "average", "Currently only plotting the average of all predictions is " \
                                                      "implemented as an alternative to plotting all predictions. \n" \
                                                      "Use evaluation_mode=average to make use of this feature."
            self.all_values = {}

        if target:
            os.makedirs(target, exist_ok=True)
            self.file_path = os.path.join(target, plot_name + ".png")
        else:
            tmp_path = os.path.join(os.path.expandvars("$HOME"), "Plots", evaluation,
                                    layer, noise_type, start_time)
            os.makedirs(tmp_path, exist_ok=True)
            self.file_path = os.path.join(
                tmp_path, plot_name + f"_date_time={date_time}" + ".png")

    def add_to_plot(self, predictions, labels):
        if self.evaluation_mode == "average":
            if isinstance(labels, pd.core.series.Series):
                labels = labels.tolist()
            if isinstance(predictions, pd.core.series.Series):
                predictions = predictions.tolist()
            if isinstance(labels, (list, np.ndarray)) and len(labels) > 1:
                print("part 1")
                print(type(labels))
                print(labels)
                print(predictions)
                for idx, label in enumerate(labels):
                    try:
                        self.all_values[label].append(predictions[idx])
                    except KeyError:
                        self.all_values[label] = [predictions[idx]]
            else:
                print("Part 2")
                if isinstance(labels, (list, np.ndarray)) and len(labels) == 1:
                    labels = int(labels[0])
                if not isinstance(predictions, (list, np.ndarray)):
                    predictions = [predictions]
                try:
                    print(type(labels))
                    print(labels)
                    self.all_values[labels].extend(predictions)
                except KeyError:
                    self.all_values[labels] = predictions

        self.fig_ax.plot(labels,
                         predictions,
                         marker='o',
                         alpha=0.5,
                         ls='',
                         color="blue")

    def save_plot(self):
        if self.evaluation_mode == "average":
            for key, values in self.all_values.items():
                mean = np.mean(values)
                stddev = np.std(values)
                self.fig_ax.errorbar(key, mean, yerr=stddev, marker='o', alpha=0.5, linestyle='', color="blue")
        xmin, xmax = self.fig_ax.axis()[:2]
        true_line = np.linspace(xmin, xmax, 100)
        self.fig_ax.plot(true_line, true_line, alpha=0.3, color="red")

        self.fig.savefig(self.file_path)
        plt.close(self.fig)
