import os
import sys
import utils
import argparse
import logging
import collections
import numpy as np
import healpy as hp
import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    IMPORTED_HVD = True
except ImportError:
    IMPORTED_HVD = False

from datetime import datetime
from DeepSphere import healpy_networks as hp_nn
from Plotter import l2_color_plot, histo_plot, stats, S8plot, PredictionLabelComparisonPlot


class Trainer:

    def __init__(self, params):
        self.date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.params = params

        self.worker_id = ""
        self.is_root_worker = True

        if self.params['training']['distributed'] and IMPORTED_HVD:
            hvd.init()
            self._configure_worker()
        elif self.params['training']['distributed']:
            logger.critical("Failed to import horovod.tensorflow. Proceeding with non distributed training")
            self.params['training']['distributed'] = False

        self._train_preprint()

        self._set_dataloader()

        if self.params['noise']['noise_type'] == "dominik_noise":
            self._set_noise_dataloader()

        self._set_model()

    def _train_preprint(self):
        print('')
        print(' -------------- Starting training    ({})'.format(self.date_time))

        if self.params['training']['distributed'] and IMPORTED_HVD:
            print('')
            print(' IN DISTRIBUTED TRAINING MODE ')
            print(' ---------------------------- ')
            print(self.worker_id)

        print('')
        print(' DATALOADER ')
        print(' ---------- ')
        print(f"- Loaded Data from {self.params['dataloader']['data_dirs']}")
        print(f"- Batch Size is  {self.params['dataloader']['batch_size']}")
        print(f"- Shuffle Size is {self.params['dataloader']['shuffle_size']}")
        print(f"- Prefetch Size is {self.params['dataloader']['prefetch_batch']}")
        print(f"- Number of Tomographic Bins is {self.params['dataloader']['tomographic_bin_number']}")
        if self.params['dataloader']['split_data']:
            print(" !!!!! DATA WILL BE SPLIT INTO TRAINING AND EVALUATION DATA !!!!! ")
        else:
            print(" !!!!! USING ALL MAPS FOR TRAINING AND EVALUATION !!!!! ")

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
        print(f"- Learning Rate is set to {self.params['model']['l_rate']}")
        if self.params['model']['profiler']['profile']:
            print(f" -- Profiling Training for Epochs {self.params['model']['profiler']['epochs']}")
            print(f" -- Saving Profile Log to {self.params['model']['profiler']['log_dir']}")
        if self.params['model']['continue_training']:
            print('')
            print(f" !! CONTINUING TRAINING !! ")
            path = os.path.join(self.params['model']['weights_dir'], self.params['model']['checkpoint_dir'])
            print(f"- Loading Weights from {path}")
        print('')

    def _configure_worker(self):
        self.worker_id = f" -- Worker ID is {hvd.rank()}/{hvd.size()}"
        if hvd.rank() != 0:
            self.is_root_worker = False

    @staticmethod
    def _mask_maker(raw_dset):
        iterator = iter(raw_dset)
        bool_mask = hp.mask_good(iterator.get_next()[0][0].numpy())
        indices_ext = np.arange(len(bool_mask))[bool_mask > 0.5]
        logger.debug(f"Extended indices are {indices_ext}")

        return bool_mask, indices_ext

    #@tf.function()
    def count_elements(self):
        num = 0
        for element in self.train_dataset.enumerate():
            num += 1
        self.params['dataloader']['number_of_elements'] = int(num)
        logger.debug(f"self.params['dataloader']['number_of_elements'] has type {type(self.params['dataloader']['number_of_elements'])}")

    def _set_dataloader(self):
        def is_test(index, value):
            return index % 5 == 0

        def is_train(index, value):
            return not is_test(index, value)

        def recover(index, value):
            return value

        data_dirs = self.params['dataloader']['data_dirs']
        batch_size = self.params['dataloader']['batch_size']
        shuffle_size = self.params['dataloader']['shuffle_size']
        prefetch_batch = self.params['dataloader']['prefetch_batch']
        distributed_training = self.params['training']['distributed']

        total_dataset = utils.get_dataset(data_dirs)

        if distributed_training:
            total_dataset = total_dataset.shard(hvd.size(), hvd.rank())

        bool_mask, indices_ext = Trainer._mask_maker(total_dataset)
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

        self.count_elements()

    def _set_noise_dataloader(self):
        """ Only used if we intend to use noise maps directly from the NGSF pipeline """
        data_dirs = self.params['noise']['noise_dataloader']['data_dirs']
        shuffle_size = self.params['noise']['noise_dataloader']['shuffle_size']
        repeat_count = self.params['noise']['noise_dataloader']['repeat_count']
        batch_size = self.params['dataloader']['batch_size']
        prefetch_batch = self.params['dataloader']['prefetch_batch']
        distributed_training = self.params['training']['distributed']

        total_noise_dataset = utils.get_dataset(data_dirs)

        if distributed_training:
            total_noise_dataset = total_noise_dataset.shard(hvd.size(), hvd.rank())

        total_noise_dataset = total_noise_dataset.shuffle(shuffle_size).repeat(
            repeat_count)
        total_noise_dataset = total_noise_dataset.batch(batch_size,
                                                        drop_remainder=True)
        total_noise_dataset = total_noise_dataset.prefetch(prefetch_batch)
        self.noise_dataset = total_noise_dataset

    def _init_noise_iteration(self):
        iterator = iter(self.noise_dataset)
        self.noise_dataset_iterator = iterator

    def _set_model(self):
        tf.keras.backend.clear_session()
        self.layers = utils.get_layers(self.params['model']['layer'])
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.params['model']['l_rate'])

        self.model = hp_nn.HealpyGCNN(nside=self.params['model']['nside'],
                                      indices=self.indices_ext,
                                      layers=self.layers)
        self.model.build(
            input_shape=(self.params['dataloader']['batch_size'],
                         self.pixel_num,
                         self.params['dataloader']['tomographic_bin_number']))
        if self.params['model']['continue_training']:
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
                self.model.load_weights(
                    tf.train.latest_checkpoint(path_to_weights))
                self.params['model']['weights_dir'] = os.path.join(
                    self.params['model']['weights_dir'], "RetrainedWeights")

    def _save_model(self, epoch):
        path_to_dir = os.path.join(os.path.expandvars("$HOME"),
                                   self.params['model']['weights_dir'],
                                   self.params['model']['layer'],
                                   self.params['noise']['noise_type'],
                                   self.date_time)
        os.makedirs(path_to_dir, exist_ok=True)
        weight_file_name = f"kappa_batch={self.params['dataloader']['batch_size']}" + \
                           f"_shuffle={self.params['noise']['noise_dataloader']['shuffle_size']}" + \
                           f"_epoch={epoch}.tf"
        save_weights_to = os.path.join(path_to_dir, weight_file_name)
        logger.info(
            f"Saving model weights to {save_weights_to} for epoch {epoch}" + self.worker_id)
        self.model.save_weights(save_weights_to)

    def _make_log_dir(self, epoch):
        path_to_dir = os.path.join(os.path.expandvars("$HOME"),
                                   self.params['model']['profiler']['log_dir'])
        os.makedirs(path_to_dir, exist_ok=True)

        log_dir = os.path.join(path_to_dir, f"layer={self.params['model']['layer']}" +
                               f"_noise={self.params['dataloader']['noise_type']}" +
                               f"_epoch={epoch}_time={self.date_time}")
        return log_dir

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
        elif self.params['noise']['noise_type'] == "dominik_noise":
            noise_element = self.noise_dataset_iterator.get_next()[0]
            noise = tf.boolean_mask(tf.transpose(noise_element, perm=[0, 2,
                                                                      1]),
                                    self.bool_mask,
                                    axis=1)
        if not self.params['noise']['noise_type'] == "dominik_noise":
            noise = tf.stack(noises, axis=-1)

        return noise

    @tf.function
    def train_step(self, first_epoch):
        for element in self.train_dataset.enumerate():
            index = tf.dtypes.cast(element[0], tf.int32)
            set = element[1]
            kappa_data = tf.boolean_mask(tf.transpose(set[0], perm=[0, 2, 1]),
                                         self.bool_mask,
                                         axis=1)
            labels = set[1]
            # Add noise
            kappa_data = tf.math.add(kappa_data, self._make_noise())

            # Optimize the model
            with tf.GradientTape() as tape:
                loss_object = tf.keras.losses.MeanAbsoluteError()
                y_ = self.model.__call__(kappa_data, training=True)
                loss_value = loss_object(y_true=labels, y_pred=y_)
            if self.params['training']['distributed']:
                tape = hvd.DistributedGradientTape(tape)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.params['training']['distributed'] and index == 0 and first_epoch:
                hvd.broadcast_variables(self.model.variables, root_rank=0)
                hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

            self.epoch_loss_avg = self.epoch_loss_avg.write(index, loss_value)
            self.epoch_global_norm = self.epoch_global_norm.write(index, tf.linalg.global_norm(grads))

    def train(self):
        # Keep results for plotting
        train_loss_results = tf.TensorArray(tf.float32,
                                            size=0,
                                            dynamic_size=True,
                                            clear_after_read=False)
        global_norm_results = tf.TensorArray(tf.float32,
                                             size=0,
                                             dynamic_size=True,
                                             clear_after_read=False)
        if self.params['model']['epochs'] < self.params['model']['number_of_epochs_eval']:
            # Defaults to evaluating the last epoch
            self.params['model']['number_of_epochs_eval'] = self.params['model']['epochs'] - 1

        for epoch in range(self.params['model']['epochs']):
            logger.debug(f"Executing training step for epoch={epoch}" + self.worker_id)

            if self.params['noise']['noise_type'] == "dominik_noise":
                self._init_noise_iteration()

            self.epoch_global_norm = tf.TensorArray(
                tf.float32,
                size=self.params['dataloader']["number_of_elements"],
                dynamic_size=False,
                clear_after_read=False,
            )
            self.epoch_loss_avg = tf.TensorArray(
                tf.float32,
                size=self.params['dataloader']["number_of_elements"],
                dynamic_size=False,
                clear_after_read=False,
            )

            epoch_cond = epoch in self.params['model']['profiler']['epochs']
            if self.params['model']['profiler']['profile'] and epoch_cond and self.is_root_worker:
                log_dir = self._make_log_dir(epoch)
                logger.info("Starting profiling" + self.worker_id + "\n")
                with tf.profiler.experimental.Profile(log_dir):
                    self.train_step(epoch == 0)
            else:
                self.train_step(epoch == 0)
            self.epoch_loss_avg = self.epoch_loss_avg.stack()
            self.epoch_global_norm = self.epoch_global_norm.stack()

            # End epoch
            if epoch % 10 == 0:
                loss = sum(self.epoch_loss_avg) / len(self.epoch_loss_avg)
                glob_norm = sum(self.epoch_global_norm) / len(self.epoch_global_norm)

                logger.info(f"Finished epoch {epoch}. Loss was {loss}" + self.worker_id)

                train_loss_results = train_loss_results.write(epoch, loss)
                global_norm_results = global_norm_results.write(epoch, glob_norm)

            epoch_cond = epoch % self.params['model']['epochs_save'] == 0
            if epoch > 0 and epoch_cond and self.is_root_worker and not self.params['model']['debug']:
                self._save_model(epoch + 1)

            eval_cond = (epoch % (self.params['model']['epochs'] // self.params['model']['number_of_epochs_eval']) == 0)
            epoch_cond = (epoch + 1 == self.params['model']['epochs'])
            if (epoch > 0 and eval_cond) or epoch_cond and self.is_root_worker and not self.params['model']['debug']:
                # Evaluate the model and plot the results
                logger.info(f"Evaluating the model and plotting results for epoch={epoch}" + self.worker_id)
                epoch_non_zero = epoch + 1

                color_predictions = []
                color_labels = []
                om_histo = []
                s8_histo = []

                all_results = {"om": collections.OrderedDict(), "s8": collections.OrderedDict()}
                om_pred_check = PredictionLabelComparisonPlot(
                    "Omega_m",
                    epoch=epoch_non_zero,
                    layer=self.params['model']['layer'],
                    noise_type=self.params['noise']['noise_type'],
                    start_time=self.date_time)
                s8_pred_check = PredictionLabelComparisonPlot(
                    "Sigma_8",
                    epoch=epoch_non_zero,
                    layer=self.params['model']['layer'],
                    noise_type=self.params['noise']['noise_type'],
                    start_time=self.date_time)

                for set in self.test_dataset:
                    kappa_data = tf.boolean_mask(tf.transpose(
                        set[0], perm=[0, 2, 1]),
                        self.bool_mask,
                        axis=1)
                    labels = set[1]
                    labels = labels.numpy()

                    # Add noise
                    kappa_data = tf.math.add(kappa_data,
                                             self._make_noise())
                    predictions = self.model.__call__(kappa_data)

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
                           epoch=epoch_non_zero,
                           layer=self.params['model']['layer'],
                           noise_type=self.params['noise']['noise_type'],
                           start_time=self.date_time)
                histo_plot(s8_histo,
                           "S8",
                           epoch=epoch_non_zero,
                           layer=self.params['model']['layer'],
                           noise_type=self.params['noise']['noise_type'],
                           start_time=self.date_time)
                l2_color_plot(
                    np.asarray(color_predictions),
                    np.asarray(color_labels),
                    epoch=epoch_non_zero,
                    layer=self.params['model']['layer'],
                    noise_type=self.params['noise']['noise_type'],
                    start_time=self.date_time)
                S8plot(all_results["om"],
                       "Om",
                       epoch=epoch_non_zero,
                       layer=self.params['model']['layer'],
                       noise_type=self.params['noise']['noise_type'],
                       start_time=self.date_time)
                S8plot(all_results["s8"],
                       "sigma8",
                       epoch=epoch_non_zero,
                       layer=self.params['model']['layer'],
                       noise_type=self.params['noise']['noise_type'],
                       start_time=self.date_time)
                om_pred_check.save_plot()
                s8_pred_check.save_plot()

        if not self.params['model']['debug'] and self.is_root_worker:
            stats(train_loss_results.stack().numpy(),
                  "training_loss",
                  layer=self.params['model']['layer'],
                  noise_type=self.params['noise']['noise_type'],
                  start_time=self.date_time)
            stats(global_norm_results.stack().numpy(),
                  "global_norm",
                  layer=self.params['model']['layer'],
                  noise_type=self.params['noise']['noise_type'],
                  start_time=self.date_time)

            self._save_model(epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', type=str, action='store')
    parser.add_argument('--weights_dir', type=str, action='store')
    parser.add_argument('--noise_dir', type=str, action='store')
    parser.add_argument('--batch_size', type=int, action='store')
    parser.add_argument('--shuffle_size', type=int, action='store')
    parser.add_argument('--epochs', type=int, action='store')
    parser.add_argument('--layer', type=str, action='store')
    parser.add_argument('--noise_type',
                        type=str,
                        action='store',
                        default='pixel_noise')
    parser.add_argument('--split_data', action='store_true', default=False)
    parser.add_argument('--nside', type=int, action='store', default=512)
    parser.add_argument('--l_rate', type=float, action='store', default=0.008)
    parser.add_argument('--continue_training',
                        action='store_true',
                        default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        action='store',
                        default='undefined')
    parser.add_argument('--profile', action='store_true', default=False)
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
    parser.add_argument('--log_dir',
                        type=str,
                        action='store',
                        default="model_profiles")
    parser.add_argument('--epochs_save', type=int, action='store', default=30)
    parser.add_argument('--number_of_epochs_eval',
                        type=int,
                        action='store',
                        default=4)
    parser.add_argument('--distributed_training', action='store_true', default=False)
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
            'split_data': ARGS.split_data
        },
        'noise': {
            'noise_type': ARGS.noise_type,
            'noise_dir': ARGS.noise_dir,
            'noise_dataloader': {
                'data_dirs': "/scratch/snx3000/bsuter/TFRecordNoise",
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
            'layer': ARGS.layer,
            'epochs': ARGS.epochs,
            'l_rate': ARGS.l_rate,
            'nside': ARGS.nside,
            'continue_training': ARGS.continue_training,
            'weights_dir': ARGS.weights_dir,
            'checkpoint_dir': ARGS.checkpoint_dir,
            'epochs_save': ARGS.epochs_save,
            'number_of_epochs_eval': ARGS.number_of_epochs_eval,
            'profiler': {
                'log_dir': ARGS.log_dir,
                'profile': ARGS.profile,
                'epochs': [1, 10]
            }
        },
        'training': {
            'distributed': ARGS.distributed_training
        }
    }
    #tf.config.experimental_run_functions_eagerly(True)
    trainer = Trainer(parameters)
    trainer.train()
