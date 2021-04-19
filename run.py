"""Entry point for running an experiment:

```
python3 run.py [path_to_config]
```

"""


from argparse import ArgumentParser
from datetime import datetime
import re
import os
import yaml

from src.datasets.datasets import load_dataset
from src.experiments import run_experiment
from src.logger import Logger
from src.metrics.metrics import load_metrics_computers
from src.models.models import load_model


def get_run_name(config):
    """Returns timestamped name for this run based on config file.

    Args:
        config (dict): Overall config dict.

    Returns:
        str: Timestamped name for this run.
    """
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    metric_names = "-".join(list(config["metrics"].keys()))
    if config.get("unique"):
        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        return "_".join([dataset_name, model_name, metric_names, timestamp])
    return "_".join([dataset_name, model_name, metric_names])


def main():
    """Gets config file, loads specified components, and runs experiment."""
    # Get overall config dict.
    parser = ArgumentParser()
    parser.add_argument("config")
    config_file = parser.parse_args().config
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Initialize logger.
    run_name = get_run_name(config)
    print(f"Running {run_name}")
    logger = Logger(config["logger"], run_name, config_file)
    logger.p(f"Using config file {config_file}.")

    logger.p("Loading model...")
    model = load_model(config["model"])

    logger.p("Loading dataset...")
    dataset = load_dataset(config["dataset"])

    logger.p("Loading metrics...")
    metric_computers = load_metrics_computers(config["metrics"])

    logger.p("Running experiment...")
    run_experiment(model, dataset, logger, metric_computers, config["experiment"])

    logger.p("Done! Closing logger...")
    logger.close()

    print("Have a nice day!")


if __name__ == "__main__":
    main()
