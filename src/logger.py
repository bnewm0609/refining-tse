"""Defines helper object for logging."""


from matplotlib.figure import Figure
import numpy as np
import os
import pickle
from shutil import copyfile


class Logger(object):
    """Helper object for logging.

    Supports writing for multiple metrics and can write text, figures, `.npz`s, or `.pkl`s.

    Attributes:
        path (str): Path to directory to log to.
        general_log_file: Main logging file object.
        npz_path (str): Path to directory to log `.npz` files to.
        pickle_path (str): Path to directory to log `.pkl` files to.
        fig_path (str): Path to directory to log figures to.
        print_to_stdout (bool): Whether to print to stdout.
        print_metrics_to_general_log (bool): Whether to print results from metrics to
            the general log, instead of just to metric-specific logs.
    """
    def __init__(self, config, run_name, config_file):
        """Initializes Logger. Must initialize `self.metric_writers` later when metrics are known.

        Args:
            config (dict): Logger-level config dict.
            run_name (str): Name of this run.
            config_file (str): Path to overall config dict.
        """
        # Path to write to.
        self.path = os.path.join(config["path"], run_name)
        os.makedirs(self.path, exist_ok=True)

        # Copy config file ASAP.
        copyfile(config_file, os.path.join(self.path, "config.yaml"))

        # Places to write to.
        self.general_log_file = open(os.path.join(self.path, "log.txt"), "w")
        self.npz_path = os.path.join(self.path, "npzs")
        os.makedirs(self.npz_path, exist_ok=True)
        self.pickle_path = os.path.join(self.path, "pickles")
        os.makedirs(self.pickle_path, exist_ok=True)
        self.fig_path = os.path.join(self.path, "figs")
        os.makedirs(self.fig_path, exist_ok=True)

        # Print options.
        self.print_to_stdout = config["print_to_stdout"]
        self.print_metrics_to_general_log = config["print_metrics_to_general_log"]
        self.p(f"Writing output to: {self.path}")

        # This can only be initialized when we know which metrics to use.
        self.metric_writers = None

    @staticmethod
    def write_to_file(msg, file):
        """Writes a string to a file and appends a newline.

        Args:
            msg (str): Message to write.
            file: Open file object to write to.
        """
        file.write(msg)
        file.write("\n")

    def p(self, msg, file=None, allow_stdout=True):
        """Writes a string to a file (if any) and possibly to the main log file or stdout.

        If there is no file or `self.print_metrics_to_general_log`, then write to the general log,
        and also write to stdout of allowed to.

        Args:
            msg (str): Message to write.
            file: Open file object to write to. If None, we might write to the main log file.
            allow_stdout (bool): Whether writing to stdout is allowed.
        """
        if file is None or self.print_metrics_to_general_log:
            if self.print_to_stdout and allow_stdout:
                print(msg)
            Logger.write_to_file(msg, self.general_log_file)
        if file:
            Logger.write_to_file(msg, file)

    def p_npz(self, name, data):
        """Writes data to an `.npz` file.

        Args:
            name (str): Filename.
            data: Data to write.
        """
        np.savez(os.path.join(self.npz_path, f"{name}.npz"), data)

    def p_pickle(self, name, data):
        """Writes data to a `.pkl` file.

        Args:
            name (str): Filename.
            data: Data to write.
        """
        with open(os.path.join(self.pickle_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def p_fig(self, name, fig):
        """Writes a `matplotlib.figure.Figure`.

        Args:
            name (str): Filename.
            fig (Figure): Figure to write.
        """
        fig.savefig(os.path.join(self.fig_path, name))

    def p_dict(self, msg_dict, file=None, allow_stdout=True):
        """Writes contents of a dict, inferring its contents.

        For each key-value pair, checks if the value is an `np.ndarray` or a `matplotlib.figure.Figure`.
        Writes an `.npz` or figure appropriately. If the value is a scalar, prints it out as text.
        Otherwise, pickles the value. Generally, keys are used as filenames.

        Args:
            msg_dict (dict): Dict of things to write.
            file: Open file object to write to, or None.
            allow_stdout (bool): Whether writing to stdout is allowed.
        """
        for key, val in msg_dict.items():
            if isinstance(val, np.ndarray):
                self.p(f"Writing npz with name {key}.", file, allow_stdout)
                self.p_npz(key, val)
            elif isinstance(val, Figure):
                self.p(f"Writing figure with name {key}.", file, allow_stdout)
                self.p_fig(key, val)
            elif np.isscalar(val):
                self.p(f"{key}: {val}", file, allow_stdout)
            else:
                self.p(f"Writing pickle with name {key}.", file, allow_stdout)
                self.p_pickle(key, val)

    def p_general(self, msg, file=None, allow_stdout=True):
        """Writes stuff in an inferred manner.

        If `msg` is a string, just prints the string.
        If `msg` is a `dict`, uses `p_dict` to write.
        Currently doesn't support other types of `msg`.

        Args:
            msg: Content to write.
            file: Open file object to write to, or None.
            allow_stdout (bool): Whether writing to stdout is allowed.
        """
        if isinstance(msg, str):
            self.p(msg, file, allow_stdout)
        elif isinstance(msg, dict):
            self.p_dict(msg, file, allow_stdout)
        else:
            raise TypeError(f"We don't know how to record results for a message of type {type(msg)}.")

    def initialize_metric_writers(self, metric_names):
        """Initializes metric writers based on metric names. Each metric gets one text file.

        Args:
            metric_names (List[str]): List of metric names.
        """
        # Check that it's appropriate to initialize these.
        if hasattr(self, "metric_writers") and self.metric_writers is not None:
            raise ValueError("Tried to initialize metric writers but already had them!"
                             " Clear old ones first using `logger.close()`.")

        self.metric_writers = {}
        metrics_path = os.path.join(self.path, "metrics")
        os.makedirs(metrics_path, exist_ok=True)

        for metric_name in metric_names:
            metric_writer = open(os.path.join(metrics_path, f"{metric_name}.txt"), "w")
            self.metric_writers[metric_name] = metric_writer
            self.p(f"Metric: {metric_name}", metric_writer)

    def close(self):
        """Closes all open file objects. Resets `self.metric_writers` to None."""
        for metric_writer in self.metric_writers.values():
            metric_writer.close()
        self.metric_writers = None
        self.general_log_file.close()

    def write_metrics(self, min_pair, metrics_dict):
        """Writes that an example was run, and writes associated metrics.

        Args:
            min_pair (.datasets.MPE_Dataset.Min_Pair): A minimal pair example.
            metrics_dict (dict): Maps metric names to contents to write.
        """
        # Record this example.
        self.p(f"Ran example {min_pair.example_id} '{min_pair.left_context} [MASK] {min_pair.right_context}' "
               f"from template {min_pair.template_id} with label {min_pair.label.name}.")

        # Record each metric.
        for metric_name, results in metrics_dict.items():
            metric_writer = self.metric_writers[metric_name]
            self.p_general(f"Template {min_pair.template_id}, Example {min_pair.example_id}:", metric_writer)
            self.p_general(results, metric_writer)

    def summarize(self, summary_dict):
        """Writes summary results at the end of an experiment.

        Args:
            summary_dict (dict): Maps metric names to contents to write.
        """
        self.p("\n--------Summary Results--------\n")
        for metric_name, summary in summary_dict.items():
            self.p(f"For metric {metric_name}:")
            self.p_general(summary)
            self.p_general(summary, self.metric_writers[metric_name], False)
            self.p("-" * 30)
