"""Routes experiment to instances of the specified metrics to compute.

Attributes:
    metric_name_to_MetricComputer_class (dict): Maps metric name to MetricComputer subclass.
"""


from .main_metric import MainMetricComputer
from .metric_computer import MetricComputer
from .ML_metric import MLMetricComputer


metric_name_to_MetricComputer_class = {
    "ML": MLMetricComputer, # computes TSE
    "main": MainMetricComputer, # computes EW and MW
}


def load_metrics_computers(config):
    """Loads instances of metrics specified in config.

    Args:
        config (dict): Metrics-level config dict with keys specifying metric names and values being config dicts
            for each MetricComputer.

    Returns:
        Dict[MetricComputer]: Instances of loaded metrics, specified by name.
    """
    metrics_computers = {}
    for metric_name, metric_computer_args in config.items():
        class_of_MetricComputer = metric_name_to_MetricComputer_class.get(metric_name)
        if class_of_MetricComputer:
            assert issubclass(class_of_MetricComputer, MetricComputer)
            metrics_computers[metric_name] = class_of_MetricComputer(metric_computer_args)
        else:
            raise ValueError(f"Unrecognized metric name {metric_name}.")
    return metrics_computers
