import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from utils import create_nonexistent_directory


class GenericLogger:

    def __init__(
            self,
            log_dir: str,
            log_interval=50,
            val_interval=200,
            metrics=None,
            plot_fn=None
    ):

        """
        Generic logger class. 

        :param log_dir: 
            Root directory for the log files
        :type log_dir : str
    
        :param log_interval: 
            Interval between logging, measured in batches
        :type log_interval : int

        :param val_interval: 
            Interval between validation, measured in batches
        :type log_interval : int

        :param metrics (default=None): 
            List of different metrics to log 
        :type metrics: 
            List of :class:`BaseMetric` or None 

        :param generate_interval: 
            Interval between generating samples, measured in batches
        :type generate_interval: int

        :param plot_fn:
            A function to plot samples. 
        :type plot_fn: callable
        """

        if not os.path.isabs(log_dir):
            log_dir = os.path.join(os.getcwd(), log_dir)

        # if check_if_dir_exists(path):
        #     warnings.warn(f"{path} already exists as non-empty directory. Log data may be overwritten.")

        self.log_dir = log_dir
        self.log_interval = log_interval
        self.val_interval = val_interval

        # array to accumulate loss function values

        self.loss_accum = []

        self.metrics = metrics if metrics is not None else []

        # map to store history of all metrics and losses

        self.metrics_history = dict()

        # create subdirectory for loss and each metrics

        self.metrics_subdirs = dict()

        self.metrics_subdirs['loss'] = os.path.join(log_dir, 'loss')
        if not check_if_dir_exists(self.metrics_subdirs['loss']):
            os.makedirs(self.metrics_subdirs['loss'])

        for m in self.metrics:
            self.metrics_subdirs[m.name] = os.path.join(log_dir, m.name)
            if not check_if_dir_exists(self.metrics_subdirs[m.name]):
                os.mkdir(self.metrics_subdirs[m.name])

    def log_metrics(self, name, value, plot=True, plot_fn=None):

        save_path = os.path.join(self.metrics_subdirs[name], name)

        if name not in self.metrics_history:
            self.metrics_history[name] = [value]
        else:
            self.metrics_history[name].append(value)

        np.savetxt(save_path + '.txt', np.array(self.metrics_history[name]), delimiter=',')

        if plot:
            plt.figure(figsize=(10, 10))
            plt.plot(self.metrics_history[name], label=f'{name}')
            plt.legend()
            plt.savefig(save_path + '.png', dpi=150)
            plt.close()

    def log(self, trainer, current_step, loss, training_disc):

        self.g_loss_accum.append(loss)

        if current_step % self.log_interval == 0:
            self.log_metrics('loss', np.mean(np.array(self.loss_accum)))
            self.loss_accum = []

        if current_step % self.val_interval == 0:

            val_metrics = trainer.validate(self.metrics)
            for name, value in val_metrics.items(): self.log_metrics(name, value)
