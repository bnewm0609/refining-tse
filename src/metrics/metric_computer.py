"""Defines general structure of a metric computer, as well as some general utility functions.

Attributes:
    example_aggregator_name_to_function: Maps name of an example aggregator to the corresponding function.
"""


import numpy as np

from ..constants import INVALID_SCORE_VALUE, BATCH_SIZE


def mean_example_aggregator(score_per_example, extra_info_per_example):
    """Aggregates example scores via mean. Returns invalid score if 0 scores are given."""
    if len(score_per_example) == 0:
        return INVALID_SCORE_VALUE, {}
    else:
        return np.mean(score_per_example), {}


example_aggregator_name_to_function = {
    "mean": mean_example_aggregator,
}


class MetricComputer(object):
    """Abstract class that computes a metric for each example, records some information, and later produces a summary.

    Functions to override:
        get_example_aggregator_from_name
        _compute

    Attributes:
        extra_info_per_example (list): List of per-example info needed for later aggregation, e.g. example weight.
        example_aggregator: Function that aggregates information.
            Args:
                score_per_example (List[float]): List of scores for each example.
                extra_info_per_example (list): List of extra information for each example.
            Returns:
                Tuple([float, dict]): Overall score and any supplementary information.
    """
    def __init__(self, config):
        """Initializes metric computer.

        Args:
            config: Metric-level config dict.
        """
        self._score_per_example = []
        self.extra_info_per_example = []  # for more complicated aggregation functions, e.g. weights per example
        self.example_aggregator = self.get_example_aggregator_from_name(config["example_aggregator"])
        self._summary_functions = self.initialize_summary_functions()
        self.use_custom_dataset = config.get("use_custom_dataset", False)

    @property
    def summary_functions(self):
        """list: List of functions to run during summarization.
            Returns:
                list: Update output `summary_dict` with more information to give to the `Logger`.

        Children should include `self.summary_functions.extend([func1, func2])` in their `__init__`.
        """
        return self._summary_functions

    @property
    def score_per_example(self):
        """List[float]: List of scores for each example."""
        return self._score_per_example

    def initialize_summary_functions(self):
        """Returns list of functions to run during summarization.

        Returns:
            list: List of functions to run.
                Returns:
                    dict: Information to give to `Logger`.
        """
        return []

    def get_example_aggregator_from_name(self, aggregator_name):
        """Returns the example aggregator function given the aggregator name. Override me.

        Args:
            aggregator_name (str): Name of example aggregator.

        Returns:
            Function that aggregates information, as described in `MetricComputer` docstring.
        """
        return example_aggregator_name_to_function[aggregator_name]

    def _compute(self, logits, label, template_id, word_to_index):
        """Computes score for a single example. Override me.

        Same Args as `compute`.

        Returns:
            Tuple[float, Any, dict]: The example score, any extra info for later example aggregation (e.g., weight),
                and a dict to update this example's `metrics_dict` with.
        """
        raise NotImplementedError

    def compute(self, logits, label, template_id, word_to_index):
        """Computes metrics for a single example, and keeps some internal notes.

        Records whether the score was invalid. If not, tracks them.

        Args:
            logits (torch.Tensor): Predicted logits with shape (1, vocab_size) or just (vocab_size).
            label (..constants.Number): Correct singular/plural label of this example.
            template_id: Template ID of this example.
            word_to_index: Dict-like indexer object mapping a word to an index.

        Returns:
            dict: Information for the `Logger` to record for this example. Includes "score" key.
        """
        score, extra_info, metrics_dict = self._compute(logits, label, template_id, word_to_index)

        if score == INVALID_SCORE_VALUE:
            metrics_dict["score"] = "INVALID_SCORE_VALUE"
        else:
            metrics_dict["score"] = score
            if isinstance(score, list):
                # used ony by ML metric
                valid_idxs, valid_scores, valid_extra_infos = zip(*[(i, s, ei) for i, (s, ei) in enumerate(zip(score, extra_info)) if s != INVALID_SCORE_VALUE])
                metrics_dict["valid_idxs"] = valid_idxs
                self.score_per_example.extend(valid_scores)
                self.extra_info_per_example.extend(valid_extra_infos)
            else:
                # used by main metric, but main metric tracks its own state so
                # `score` is not meaningful
                self.score_per_example.append(score)
                self.extra_info_per_example.append(score)
        return metrics_dict

    def summarize(self):
        """Summarizes the model's score on the dataset by producing information for the `Logger`.

        Returns:
            dict: Summary information for the `Logger`, including an "Overall model score" key.
        """
        overall_score, summary_dict = self.example_aggregator(self.score_per_example, self.extra_info_per_example)

        if overall_score == INVALID_SCORE_VALUE:
            summary_dict["Overall model score"] = "INVALID_SCORE_VALUE"
        else:
            summary_dict["Overall model score"] = overall_score
            summary_dict["Number of examples"] = len(self.score_per_example)

            # Iterate through other summary functions.
            for summary_function in self.summary_functions:
                summary_dict.update(summary_function())
        return summary_dict
