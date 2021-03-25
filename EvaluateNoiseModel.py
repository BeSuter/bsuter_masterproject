import os
import re
import sys
import utils
import argparse
import logging
import collections
import numpy as np
import healpy as hp
import tensorflow as tf

from datetime import datetime
from deepsphere import healpy_networks as hp_nn
from Plotter import l2_color_plot, histo_plot, S8plot, PredictionLabelComparisonPlot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


class Evaluator:

    def __init__(self, params):
        self.date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.params = params

        self.worker_id = ""
        self.is_root_worker = True

        self.indices_ext_initialized = None
        self.pixel_num = None

        self._train_preprint()

        self._set_dataloader()

        if self.params['noise']['noise_type'] == "dominik_noise":
            self._set_noise_dataloader()

        self._set_model()

    def _train_preprint(self):
        print('')
        print(' -------------- Starting Evaluation    ({})'.format(self.date_time))

        if self.params['model']['debug']:
            print('')
            print(' ######################################################')
            print('          !!!!   RUNNING IN DEBUG MODE   !!!!')
            print('            Saving of Plots will be skipped')
            print(' ######################################################')
            print('')

        print('')
        print(' DATALOADER ')
        print(' ---------- ')
        print(f"- Loaded Data from {self.params['dataloader']['data_dirs']}")
        print(f"- Batch Size is  {self.params['dataloader']['batch_size']}")
        print(f"- Shuffle Size is {self.params['dataloader']['shuffle_size']}")
        print(f"- Prefetch Size is {self.params['dataloader']['prefetch_batch']}")
        print(f"- Number of Tomographic Bins is {self.params['dataloader']['tomographic_bin_number']}")
        if self.params['dataloader']['pipeline_data']:
            print(" !!!!! Using Maps generated vrom Dominik's Pipeline !!!!")
            print(f" Random ordering of Maps is set to {self.params['dataloader']['random']}")
        if self.params['dataloader']['split_data']:
            print(" !!!!! DATA WILL BE SPLIT INTO TRAINING AND EVALUATION DATA !!!!! ")
        else:
            print(" !!!!! USING ALL MAPS FOR EVALUATION !!!!! ")

        print('')
        print(' NOISE ')
        print(' ----- ')
        print(f"- Noise Type is {self.params['noise']['noise_type']}")
        if self.params['noise']['noise_type'] == 'dominik_noise':
            print(f" -- Loaded Noise from {self.params['noise']['noise_dataloader']['data_dirs']}")
            print(f" -- Noise Shuffle Size is {self.params['noise']['noise_dataloader']['shuffle_size']}")
            print(f" -- Noise Repeat Count is {self.params['noise']['noise_dataloader']['repeat_count']}")

        print('')
        print(' MODEL ')
        print(' ----- ')
        print(f"- Layer Name is {self.params['model']['layer']}")
        print(f"- NSIDE set to {self.params['model']['nside']}")
        print(
            f"- Number of neighbors considered when building the graph is set to {self.params['model']['n_neighbors']}"
        )
        path = os.path.join(self.params['model']['weights_dir'], self.params['model']['checkpoint_dir'])
        print(f"- Loading Weights from {path}")
        print('')

    @staticmethod
    def _mask_maker(raw_dset):
        iterator = iter(raw_dset)
        bool_mask = hp.mask_good(iterator.get_next()[0][0].numpy())
        indices_ext = np.arange(len(bool_mask))[bool_mask > 0.5]
        logger.debug(f"Extended indices are {indices_ext}")

        return bool_mask, indices_ext

    @staticmethod
    def _label_finder(str):
        Om_label = re.search(r"(?<=Om=).+(?=_s8=)", str).group(0)
        s8_label = re.search(r"(?<=s8=).+(?=_tomo)", str).group(0)
        return float(Om_label), float(s8_label)

    def import_pipeline_maps(self, index, random=False):
        full_tomo_map = []
        logger.debug(f"Loading pipeline maps")
        path_to_map_ids = os.path.join("/scratch/snx3000/bsuter/Maps", "Map_ids.npy")
        all_map_paths = os.listdir("/scratch/snx3000/bsuter/Maps/FullMaps")

        all_ids = np.load(path_to_map_ids)
        choosen_labels = []
        # np.random.seed(self.params['dataloader']['seed'])
        if random:
            random_ids = np.random.randint(0, high=len(all_ids), size=1)
        else:
            random_ids = [index]
            logger.info(f"Using Id number {index}")
        for id in random_ids:
            for file_name in all_map_paths:
                if file_name.endswith(f"_id={all_ids[id]}.npy"):
                    choosen_labels.append(self._label_finder(file_name))
                    break

        for tomo in range(self.params['dataloader']['tomographic_bin_number']):
            single_tomo_maps = []
            for idx, id_num in enumerate(random_ids):
                map_name = os.path.join("/scratch/snx3000/bsuter/Maps",
                                        "FullMaps",
                                        f"Map_Om={choosen_labels[idx][0]}_s8={choosen_labels[idx][1]}_tomo={tomo + 1}_id={all_ids[id_num]}.npy")
                full_map = np.load(map_name)
                if not self.indices_ext_initialized:
                    logger.info(f"Initializing extended indices")
                    self.indices_ext = np.arange(len(full_map))[full_map > hp.UNSEEN]
                    self.pixel_num = len(self.indices_ext)
                    logger.debug(f"Extended indices are {self.indices_ext}")
                    self.indices_ext_initialized = True
                map = tf.convert_to_tensor(full_map[full_map > hp.UNSEEN], dtype=tf.float32)
                single_tomo_maps.append(map)
            full_tomo_map.append(tf.stack(single_tomo_maps, axis=0))

        stacked_maps = tf.stack(full_tomo_map, axis=-1)
        logger.debug(f"Stacked Maps are {stacked_maps}")
        return stacked_maps, choosen_labels

    def _set_dataloader(self):
        def is_test(index, value):
            return index % 5 == 0

        def is_train(index, value):
            return not is_test(index, value)

        def recover(index, value):
            return value

        if self.params['dataloader']['pipeline_data']:
            self.test_dataset = []
            for i in range(self.params['dataloader']['map_count']):
                self.test_dataset.append(self.import_pipeline_maps(i, random=False))
            logger.debug(f"Shape of Pipeline Data is {np.shape(self.test_dataset)}")
            logger.debug(f"Pixel Number is {self.pixel_num}")

        else:
            data_dirs = self.params['dataloader']['data_dirs']
            batch_size = self.params['dataloader']['batch_size']
            shuffle_size = self.params['dataloader']['shuffle_size']
            prefetch_batch = self.params['dataloader']['prefetch_batch']

            total_dataset = utils.get_dataset(data_dirs)

            bool_mask, indices_ext = Evaluator._mask_maker(total_dataset)
            self.bool_mask = bool_mask
            self.indices_ext = indices_ext
            self.pixel_num = len(indices_ext)

            total_dataset = total_dataset.shuffle(shuffle_size)
            total_dataset = total_dataset.batch(batch_size, drop_remainder=True)

            if self.params['dataloader']['split_data']:
                test_dataset = total_dataset.enumerate().filter(is_test).map(
                    recover)
                train_dataset = total_dataset.enumerate().filter(is_train).map(
                    recover)
                self.test_dataset = test_dataset.prefetch(prefetch_batch)
                self.train_dataset = train_dataset.prefetch(prefetch_batch)
            else:
                total_dataset = total_dataset.prefetch(prefetch_batch)
                self.test_dataset = total_dataset
                self.train_dataset = total_dataset

    def _set_noise_dataloader(self):
        """ Only used if we intend to use noise maps directly from the NGSF pipeline """
        data_dirs = self.params['noise']['noise_dataloader']['data_dirs']
        shuffle_size = self.params['noise']['noise_dataloader']['shuffle_size']
        repeat_count = self.params['noise']['noise_dataloader']['repeat_count']
        batch_size = self.params['dataloader']['batch_size']
        prefetch_batch = self.params['dataloader']['prefetch_batch']

        total_noise_dataset = utils.get_dataset(data_dirs)

        total_noise_dataset = total_noise_dataset.shuffle(shuffle_size).repeat(-1)
        total_noise_dataset = total_noise_dataset.batch(batch_size,
                                                        drop_remainder=True)
        total_noise_dataset = total_noise_dataset.prefetch(prefetch_batch)
        iterator = iter(total_noise_dataset)
        self.noise_dataset_iterator = iterator

    def _set_model(self):
        tf.keras.backend.clear_session()
        self.layers = utils.get_layers(self.params['model']['layer'])

        self.model = hp_nn.HealpyGCNN(nside=self.params['model']['nside'],
                                      indices=self.indices_ext,
                                      layers=self.layers,
                                      n_neighbors=self.params['model']['n_neighbors'])
        self.model.build(
            input_shape=(self.params['dataloader']['batch_size'],
                         self.pixel_num,
                         self.params['dataloader']['tomographic_bin_number']))
        logger.debug(f"Building model with input shape ({self.params['dataloader']['batch_size'], self.pixel_num, self.params['dataloader']['tomographic_bin_number']})")

        if self.params['model']['checkpoint_dir'] == "undefined":
            logger.critical(
                "Please define the directory within NGSFweights containing the desired weights " +
                "E.g. --checkpoint_dir=layer_2/pixel_noise/01-14-2021-18-38" + self.worker_id
            )
            sys.exit(0)
        else:
            path_to_weights = os.path.join(
                self.params['model']['weights_dir'],
                self.params['model']['checkpoint_dir'])
            # self.model.load_weights(
            #    tf.train.latest_checkpoint(path_to_weights))
            self.model.load_weights(tf.train.latest_checkpoint(path_to_weights))

    def _make_noise(self):
        noises = []
        if self.params['noise']['noise_type'] == "pixel_noise":
            for tomo in range(
                    self.params['dataloader']['tomographic_bin_number']):
                try:
                    noise_path = os.path.join(
                        self.params['noise']['noise_dir'],
                        f"NewPixelNoise_tomo={tomo + 1}.npz")
                    noise_ctx = np.load(noise_path)
                    mean_map = noise_ctx["mean_map"]
                    variance_map = noise_ctx["variance_map"]
                except FileNotFoundError:
                    logger.critical(
                        "Are you trying to read PixelNoise_tomo=2x2.npz or PixelNoise_tomo=2.npz?\n" +
                        "At the moment the noise is hardcoded to PixelNoise_tomo=2.npz. Please change this..." +
                        self.worker_id
                    )
                    sys.exit(0)
                mean = tf.convert_to_tensor(mean_map, dtype=tf.float32)
                stddev = tf.convert_to_tensor(variance_map, dtype=tf.float32)
                stddev *= 38.798  # Fix this, recalculate the standard deviation of the noise maps
                noise = tf.random.normal(
                    [self.params['dataloader']['batch_size'], self.pixel_num],
                    mean=0.0,
                    stddev=1.0)
                noise = tf.math.multiply(noise, stddev)
                noise = tf.math.add(noise, mean)
                noises.append(noise)
            noise = tf.stack(noises, axis=-1)
        elif self.params['noise']['noise_type'] == "old_noise":
            for tomo in range(
                    self.params['dataloader']['tomographic_bin_number']):
                noise = tf.random.normal(
                    [self.params['dataloader']['batch_size'], self.pixel_num],
                    mean=0.0,
                    stddev=1.0)
                noise *= self.params['noise']['tomographic_context'][tomo +
                                                                     1][0]
                noise += self.params['noise']['tomographic_context'][tomo +
                                                                     1][1]
                noises.append(noise)
            noise = tf.stack(noises, axis=-1)
        elif self.params['noise']['noise_type'] == "dominik_noise":
            noise_element = self.noise_dataset_iterator.get_next()[0]
            noise = tf.boolean_mask(tf.transpose(noise_element, perm=[0, 2, 1]),
                                    self.bool_mask,
                                    axis=1)
        elif self.params['noise']['noise_type'] == "noise_free":
            for tomo in range(self.params['dataloader']['tomographic_bin_number']):
                noise = np.zeros((self.params['dataloader']['batch_size'], self.pixel_num))
                noise = tf.convert_to_tensor(noise, dtype=tf.float32)
                noises.append(noise)
            noise = tf.stack(noises, axis=-1)

        return noise

    def evaluate(self):
        print('')
        logger.info(" ----   STARTING EVALUATION   ---- ")
        print('')
        color_predictions = []
        color_labels = []
        om_histo = []
        s8_histo = []

        all_results = {"om": collections.OrderedDict(), "s8": collections.OrderedDict()}
        om_pred_check = PredictionLabelComparisonPlot(
            "Omega_m",
            layer=self.params['model']['layer'],
            noise_type=self.params['noise']['noise_type'],
            start_time=self.date_time,
            evaluation="Evaluation",
            evaluation_mode=self.params['plots']['PredictionLabelComparisonPlot']['evaluation_mode'])
        s8_pred_check = PredictionLabelComparisonPlot(
            "Sigma_8",
            layer=self.params['model']['layer'],
            noise_type=self.params['noise']['noise_type'],
            start_time=self.date_time,
            evaluation="Evaluation",
            evaluation_mode=self.params['plots']['PredictionLabelComparisonPlot']['evaluation_mode'])

        for set in self.test_dataset:
            if not self.params['dataloader']['pipeline_data']:
                shape = [self.params['dataloader']['batch_size'],
                         self.pixel_num,
                         self.params['dataloader']['tomographic_bin_number']]
                kappa_data = tf.boolean_mask(tf.transpose(
                    set[0], perm=[0, 2, 1]),
                    self.bool_mask,
                    axis=1)
                kappa_data = tf.ensure_shape(kappa_data, shape)
                labels = set[1]
                labels = labels.numpy()

                # Add noise
                if not self.params['noise']['noise_free']:
                    noise = tf.ensure_shape(self._make_noise(), shape)
                    kappa_data = tf.math.add(kappa_data, noise)
            else:
                kappa_data = set[0]
                labels = np.asarray(set[1])

                logger.debug(f"Kappa Data has shape {kappa_data.shape}")
                logger.debug(f"Labels have shape {np.shape(labels)}")

            predictions = self.model.__call__(kappa_data, training=False)

            for ii, prediction in enumerate(predictions.numpy()):
                om_pred_check.add_to_plot(prediction[0], labels[ii, 0])
                s8_pred_check.add_to_plot(prediction[1], labels[ii, 1])

                color_predictions.append(prediction)
                color_labels.append(labels[ii, :])

                om_histo.append(prediction[0] - labels[ii, 0])
                s8_histo.append(prediction[1] - labels[ii, 1])

                try:
                    all_results["om"][(labels[ii][0], labels[ii][1])].append(prediction[0])
                except KeyError:
                    all_results["om"][(labels[ii][0], labels[ii][1])] = [prediction[0]]
                try:
                    all_results["s8"][(labels[ii][0], labels[ii][1])].append(prediction[1])
                except KeyError:
                    all_results["s8"][(labels[ii][0], labels[ii][1])] = [prediction[1]]

        histo_plot(om_histo,
                   "Om",
                   layer=self.params['model']['layer'],
                   noise_type=self.params['noise']['noise_type'],
                   start_time=self.date_time,
                   evaluation="Evaluation")
        histo_plot(s8_histo,
                   "S8",
                   layer=self.params['model']['layer'],
                   noise_type=self.params['noise']['noise_type'],
                   start_time=self.date_time,
                   evaluation="Evaluation")
        l2_color_plot(
            np.asarray(color_predictions),
            np.asarray(color_labels),
            layer=self.params['model']['layer'],
            noise_type=self.params['noise']['noise_type'],
            start_time=self.date_time,
            evaluation="Evaluation")
        S8plot(all_results["om"],
               "Om",
               layer=self.params['model']['layer'],
               noise_type=self.params['noise']['noise_type'],
               start_time=self.date_time,
               evaluation="Evaluation")
        S8plot(all_results["s8"],
               "sigma8",
               layer=self.params['model']['layer'],
               noise_type=self.params['noise']['noise_type'],
               start_time=self.date_time,
               evaluation="Evaluation")
        om_pred_check.save_plot()
        s8_pred_check.save_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', type=str, action='store')
    parser.add_argument('--weights_dir', type=str, action='store')
    parser.add_argument('--noise_dir', type=str, action='store')
    parser.add_argument('--batch_size', type=int, action='store', default=1)
    parser.add_argument('--shuffle_size', type=int, action='store')
    parser.add_argument('--epochs', type=int, action='store')
    parser.add_argument('--layer', type=str, action='store')
    parser.add_argument('--noise_free', action='store_true', default=False)
    parser.add_argument('--noise_type',
                        type=str,
                        action='store',
                        default='pixel_noise')
    parser.add_argument('--split_data', action='store_true', default=False)
    parser.add_argument('--nside', type=int, action='store', default=512)
    parser.add_argument('--n_neighbors', type=int, action='store', default=20)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        action='store',
                        default='undefined')
    parser.add_argument('--prefetch_batch',
                        type=int,
                        action='store',
                        default=2)
    parser.add_argument('--tomographic_bin_number',
                        type=int,
                        action='store',
                        default=4)
    parser.add_argument('--noise_shuffle',
                        type=int,
                        action='store',
                        default=75)
    parser.add_argument('--repeat_count', type=int, action='store', default=2)
    parser.add_argument('--pipeline_data', action='store_true', default=False)
    parser.add_argument('--map_count', type=int, action='store', default=0)
    parser.add_argument('--seed', type=int, action='store', default=0)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--evaluation_mode', action='store', default=None)
    ARGS = parser.parse_args()

    if ARGS.debug:
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

    parameters = {
        'dataloader': {
            'data_dirs': ARGS.data_dirs,
            'batch_size': ARGS.batch_size,
            'shuffle_size': ARGS.shuffle_size,
            'prefetch_batch': ARGS.prefetch_batch,
            'tomographic_bin_number': ARGS.tomographic_bin_number,
            'split_data': ARGS.split_data,
            'pipeline_data': ARGS.pipeline_data,
            'map_count': ARGS.map_count,
            'seed': ARGS.seed,
            'random': ARGS.random
        },
        'noise': {
            'noise_free': ARGS.noise_free,
            'noise_type': ARGS.noise_type,
            'noise_dir': ARGS.noise_dir,
            'noise_dataloader': {
                'data_dirs': "/scratch/snx3000/bsuter/Final_TFR/TFRecordNoise",
                'shuffle_size': ARGS.noise_shuffle,
                'repeat_count': ARGS.repeat_count
            },
            'tomographic_context': {
                1: [0.060280509803501296, 2.6956629531655215e-07],
                2: [0.06124986702256547, -1.6575954273040043e-07],
                3: [0.06110073383083452, -1.4452612096534303e-07],
                4: [0.06125788725968831, 1.2850254404014072e-07]
            }
        },
        'model': {
            'debug': ARGS.debug,
            'layer': ARGS.layer,
            'nside': ARGS.nside,
            'n_neighbors': ARGS.n_neighbors,
            'weights_dir': ARGS.weights_dir,
            'checkpoint_dir': ARGS.checkpoint_dir,
            },
        'plots': {
            'PredictionLabelComparisonPlot': {
                'evaluation_mode': ARGS.evaluation_mode
            }
        }
        }
    evaluator = Evaluator(parameters)
    evaluator.evaluate()
