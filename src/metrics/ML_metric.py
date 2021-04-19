"""Metric for looking at one specific lemma per example, like Marvin and Linzen."""


from ..constants import INVALID_SCORE_VALUE
from .metric_computer import MetricComputer

import numpy as np



class MLMetricComputer(MetricComputer):
    """Mimics Marvin and Linzen style evaluation (TSE), which looks at one specific lemma per example.

    Attributes:
        use_custom_dataset (bool): Whether we're using a `CustomDataset`.
    """
    def __init__(self, config):
        super().__init__(config)
        self.use_custom_dataset = config.get("use_custom_dataset", False)
        self.summary_functions.append(self.report_state_summary)

    def _compute(self, logits, label, template_id_batch, word_to_index):
        # Extract example' associated lemma
        assert self.use_custom_dataset, "Can only compute TSE with custom datasets. Make sure `use_custom_dataset` is True in metrics section of the config file."
        words_of_interest = [template_id.lemma_inflections for template_id in template_id_batch] 

        scores = []
        for batch_i, word_of_interest in enumerate(words_of_interest):
            # Ensure both inflections are in the vocab.
            if not word_of_interest[0] in word_to_index or not word_of_interest[1] in word_to_index:
                scores.append((INVALID_SCORE_VALUE, None, {}))
                continue

            # Only look at logit entries for those two inflections.
            logit_values = (logits[batch_i, word_to_index[word_of_interest[label[batch_i].value]]],
                            logits[batch_i, word_to_index[word_of_interest[1 - label[batch_i].value]]])
            scores.append(
                    (int(logit_values[label[batch_i].value] > logit_values[1 - label[batch_i].value]), None, {})
                    )

        scores, others, things = zip(*scores)
        return list(scores), list(others), {}

    def report_state_summary(self):
        return {
                "ML_scores": np.array(self.score_per_example),
                }
