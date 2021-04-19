"""Defines core logic for an experiment."""

import numpy as np
import torch
from .constants import BATCH_SIZE

def iter_batch(dataset, batch_size):
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_experiment(model, dataset, logger, metric_computers, config):
    """Runs and logs an experiment given initialized components.

    Args:
        model (.models.MPEModel): Initialized model.
        dataset (.datasets.MPEDataset): Initialized dataset.
        logger (.logger.Logger): Helper for logging text and other results.
        metric_computers (List[.metrics.MetricComputer]): List of metrics to compute.
        config (dict): Experiment-level config file.
    """
    logger.p("Initializing file writers for metric computers...")
    logger.initialize_metric_writers(list(metric_computers.keys()))

    logger.p("Iterating over dataset...")
    max_examples = config.get("max_examples")
    previous_examples = set()
    all_valid_min_pairs = {metric_name: [] for metric_name in metric_computers.keys()}
    all_valid_min_pairs_idxs = {metric_name: [] for metric_name in metric_computers.keys()}
    for i, min_pairs in enumerate(iter_batch(dataset, BATCH_SIZE)):
        if max_examples is not None and i * BATCH_SIZE >= max_examples:
            break

        # get logits from model
        logits_batch = model.predict(
                [min_pair.left_context for min_pair in min_pairs],
                [min_pair.right_context for min_pair in min_pairs]
                )

        if logits_batch is None:
            continue

        if model.bidirectional:
            min_pair_exs = [
                    f'{" ".join(min_pair.left_context)} <verb> {" ".join(min_pair.right_context)}'
                    for min_pair in min_pairs
                    ]
        else:
            min_pair_exs = [
                    f'{" ".join(min_pair.left_context)} <verb>'
                    for min_pair in min_pairs
                    ]


        metrics_dict = {}
        for metric_name, metric_computer in metric_computers.items():

            valid_idxs = []
            valid_min_pairs = []
            local_previous_examples = set()
            for min_pair_i, min_pair in enumerate(min_pairs):
                if metric_name == "ML" or (not min_pair_exs[min_pair_i] in previous_examples.union(local_previous_examples)):
                    valid_idxs.append(min_pair_i)
                    valid_min_pairs.append(min_pair)
                    local_previous_examples.add(min_pair_exs[min_pair_i])

            metrics_dict[metric_name] = metric_computer.compute(
                    logits_batch[torch.tensor(valid_idxs, device=logits_batch.device).long()],
                    [min_pair.label for min_pair in valid_min_pairs],
                    [min_pair.template_id for min_pair in valid_min_pairs],
                    model.word_to_index
                    )

            # track minimal pairs that look the same to the model to avoid overcounting
            # (only an issue for MW and EW)
            all_valid_min_pairs[metric_name].extend(valid_min_pairs)
            if "valid_idxs" in metrics_dict[metric_name]:
                all_valid_min_pairs_idxs[metric_name].extend([vi + i*BATCH_SIZE for vi in metrics_dict[metric_name]["valid_idxs"]])
            else:
                all_valid_min_pairs_idxs[metric_name].extend([vi + i*BATCH_SIZE for vi in valid_idxs])

        del logits_batch
        previous_examples = previous_examples.union(local_previous_examples)

    logger.p("Summarizing results...")
    summary_dict = {}
    for metric_name, metric_computer in metric_computers.items():
        summary_dict[metric_name] = metric_computer.summarize()
        summary_dict[metric_name].update({
            f"{metric_name}-valid_min_pairs": all_valid_min_pairs[metric_name],
            f"{metric_name}-valid_min_pairs_idxs": np.array(all_valid_min_pairs_idxs[metric_name]),
            })
    logger.summarize(summary_dict)
