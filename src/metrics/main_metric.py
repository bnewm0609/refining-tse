"""Defines the EW and MW metrics."""


import csv
from collections import defaultdict, namedtuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .metric_computer import MetricComputer


k_mass_scores = namedtuple("k_mass_scores", ["scores", "prob_counted", "natural_cutoffs"])

def array_to_str(arr, sep_char=", "):
    return sep_char.join(str(x) for x in arr)


def extract(tensor, indices, maxval=1.0, minval=0.0):  # torch tensors
    answer = tensor.gather(1, torch.clamp(indices, 0, tensor.shape[1] - 1))
    answer[indices < 0] = minval  # negative indices map to minval
    answer[indices >= tensor.shape[1]] = maxval  # indices past the length of the tensor map to maxval
    return answer

@dataclass
class PrePost:
    pre: torch.Tensor
    post: torch.Tensor


class MainMetricComputer(MetricComputer):
    """Metric computer that calculates lemma-based score for top-k or bottom-k portion of
    probability mass with binary(sum of singular, sum of plural) or average(binary(singular form, plural form)).
    """
    def __init__(self, config):
        """Initializes with a particular set of known lemmas, a lemma scorer, and a lemma aggregator.

        Args:
            config (dict): Metric-level config dict.
        """
        super().__init__(config)
        self.path = config["lemma_inflections_path"]
        self.ew = config["use_equal_verb_voting"]  # Using EW
        self.mw = not self.ew # Using MW
        self.cutoffs_top = np.array(config["cutoffs_top"])
        self.cutoffs_bot = np.array(config["cutoffs_bot"])
        self.cutoff_scores_by_example_top = []
        self.prob_counted_by_example_top = []
        self.cutoff_scores_by_example_bot = []
        self.prob_counted_by_example_bot = []

        # Load lemma information once we know model vocab.
        self.has_initialized_lemma_info = False
        self.verb_to_index = None
        self.index_to_inflections = None
        self.num_lemmas_in_model_vocab = -1

        # Internal stats to track.
        self.lemma_stats = {}

        # Update summary functions.
        self.summary_functions.append(self.main_summary)
        self.summary_functions.append(self.report_state_summary)

        # Store config dict.
        self._config = config

    def load_lemma_info(self, word_to_index):
        """Loads known lemmas from a `.csv` file, checking if they're in the model's vocabulary.

        The `.csv` file should container a header row, then rows each with a lemma index, singular inflection,
        plural inflection, and frequency.

        Args:
            word_to_index: Dict-like object that maps words to their indices in the model's logits.
        """
        skipped_index_to_inflections = {}  # Currently unused. Represents lemmas with an inflection not in the vocab.
        with open(self.path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            self.verb_to_index = {}
            self.index_to_inflections = {}
            for row in reader:
                index, singular_form, plural_form, _ = row
                if singular_form not in word_to_index or plural_form not in word_to_index:
                    skipped_index_to_inflections[index] = (singular_form, plural_form)
                else:
                    self.verb_to_index[singular_form] = index
                    self.verb_to_index[plural_form] = index
                    self.index_to_inflections[index] = (singular_form, plural_form)

    def initialize_lemma_info(self, word_to_index):
        """Loads known lemmas and checks how many are in the model's vocabulary.

        Args:
            word_to_index (dict): Maps words to their lemma index.
        """
        self.load_lemma_info(word_to_index)
        self.num_lemmas_in_model_vocab = len(self.index_to_inflections)
        self.has_initialized_lemma_info = True
        self.lemma_stats = {
                verb : {"ranks": [], "probs": [], "cum_probs":[]}
                for verb in self.verb_to_index
                }

    def _compute(self, logits, label, template_id, word_to_index):
        """Computes per-example scores based on aggregated per-lemma scores."""
        # `word_to_index` specifies model vocab, so we can initialize lemma info.
        if not self.has_initialized_lemma_info:
            self.initialize_lemma_info(word_to_index)

        # Sort logits from most to least probable.
        sorted_logits_top, sorted_indices_top = torch.sort(logits, descending=True)
        scores_top, prob_counted_top = self.compute_interpolated_scores_for_first_k_mass(
            sorted_indices_top, sorted_logits_top, self.cutoffs_top, label, word_to_index, False
        )
        self.cutoff_scores_by_example_top.extend(scores_top.tolist())
        self.prob_counted_by_example_top.extend(prob_counted_top.tolist())

        # Sort logits from least to most probable.
        sorted_logits_bot = torch.flip(sorted_logits_top, dims=[-1])
        sorted_indices_bot = torch.flip(sorted_indices_top, dims=[-1])
        scores_bot, prob_counted_bot = self.compute_interpolated_scores_for_first_k_mass(
            sorted_indices_bot, sorted_logits_bot, self.cutoffs_bot, label, word_to_index, True
        )
        self.cutoff_scores_by_example_bot.extend(scores_bot.tolist())
        self.prob_counted_by_example_bot.extend(prob_counted_bot.tolist())

        # Arbitrarily return score for first cutoff for top-k probability
        # This is just for logging, all the scores are harvested later
        return scores_top[0, 0], None, {}

    def compute_interpolated_scores_for_first_k_mass(self, sorted_indices, sorted_logits, cutoffs, label, word_to_index,
                                                     must_have_both_inflections):
        pre, post = self.compute_scores_for_first_k_mass(
                sorted_indices, sorted_logits, cutoffs, label, word_to_index, must_have_both_inflections, True)
        scores_pre, prob_counted_pre, natural_cutoff_pre = pre
        scores_post, prob_counted_post, natural_cutoff_post = post
        assert np.all(cutoffs <= natural_cutoff_post) and np.all(natural_cutoff_pre <= cutoffs)
        # Ensure that {prob_mass,scores}_{pre,post} have nans in the same locations
        for pre, post in [(scores_pre, scores_post), (prob_counted_pre, prob_counted_post)]:
            for scores_one_side, scores_other_side in [(pre, post), (post, pre)]:
                nan_idxs = np.nonzero(np.isnan(scores_one_side))
                scores_one_side[nan_idxs] = scores_other_side[nan_idxs]

        assert np.all(np.argwhere(np.isnan(scores_pre)) == np.argwhere(np.isnan(scores_post)))
        # We assume that the mass on the cutoff is enough to make this not blow up...
        alpha = (cutoffs - natural_cutoff_pre) / (natural_cutoff_post - natural_cutoff_pre)
        assert (np.all((0 <= alpha) | (np.isclose(alpha, 0))) and np.all((alpha <= 1) | (np.isclose(alpha, 1))))
        scores = (1 - alpha) * scores_pre + alpha * scores_post
        prob_counted = (1 - alpha) * prob_counted_pre + alpha * prob_counted_post
        return scores, prob_counted

    def compute_scores_for_first_k_mass(self, sorted_indices, sorted_logits, cutoffs, label, word_to_index,
                                        must_have_both_inflections, include_cutoff):
        """Computes the scores by taking the first predictions from a list to gather a certain amount of mass."""
        del include_cutoff
        device = sorted_logits.device
        num_cutoffs = len(cutoffs)

        rank_to_index_batch = [dict(list(enumerate(si.tolist()))) for si in sorted_indices.cpu()]
        index_to_rank_batch = defaultdict(list)
        for rti in rank_to_index_batch:
            for rank, index in rti.items():
                index_to_rank_batch[index].append(rank)

        # Get probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.min(torch.cumsum(sorted_probs, dim=-1), torch.tensor(1.0))  # min for rounding error

        # Which are in top-k? Shape: num_cutoffs
        # We want first i that gets at least v mass, i.e. a[i-1] < v <= a[i]
        last_accepted_rank = torch.tensor([np.searchsorted(cp, cutoffs) for cp in cum_probs.cpu().numpy()], device=device)

        # Track the cutoff we're representing. Shape: num_cutoffs
        natural_cutoff = PrePost(extract(cum_probs, last_accepted_rank - 1),
                                 extract(cum_probs, last_accepted_rank))

        # Set-up for scoring
        if self.ew:
            total_count = PrePost(torch.zeros(sorted_logits.shape[0], num_cutoffs), torch.zeros(sorted_logits.shape[0], num_cutoffs))
            right_count = PrePost(torch.zeros(sorted_logits.shape[0], num_cutoffs), torch.zeros(sorted_logits.shape[0], num_cutoffs))
        else:
            total_prob_right = PrePost(torch.zeros(sorted_logits.shape[0], num_cutoffs), torch.zeros(sorted_logits.shape[0], num_cutoffs))
            total_prob_wrong = PrePost(torch.zeros(sorted_logits.shape[0], num_cutoffs), torch.zeros(sorted_logits.shape[0], num_cutoffs))

        # Track probability actually counted
        prob_counted = PrePost(torch.zeros(sorted_logits.shape[0], num_cutoffs), torch.zeros(sorted_logits.shape[0], num_cutoffs))

        # Iterate through all known lemmas.
        label_values_batch = torch.tensor([l.value for l in label], device=device).reshape(-1, 1)
        for lemma_index, inflections in self.index_to_inflections.items():
            ranks_batch = torch.tensor(
                    [
                        tuple(index_to_rank_batch[word_to_index[word]][batch_i]
                        for word in inflections)
                        for batch_i in range(sorted_logits.shape[0])
                    ], device=device)


            prob_right_batch = sorted_probs.gather(1, ranks_batch.gather(1, label_values_batch))
            prob_wrong_batch = sorted_probs.gather(1, ranks_batch.gather(1, 1 - label_values_batch))

            is_acceptable_0 = PrePost(
                    ranks_batch[:, 0].unsqueeze(1) < last_accepted_rank,
                    ranks_batch[:, 0].unsqueeze(1) <= last_accepted_rank
                    )
            is_acceptable_1 = PrePost(
                    ranks_batch[:, 1].unsqueeze(1) < last_accepted_rank,
                    ranks_batch[:, 1].unsqueeze(1) <= last_accepted_rank
                    )

            # Shape: num_cutoffs
            if must_have_both_inflections:
                should_lemma_count = PrePost(
                    (is_acceptable_0.pre & is_acceptable_1.pre).int(),
                    (is_acceptable_0.post & is_acceptable_1.post).int(),
                )
            else:
                should_lemma_count = PrePost(
                    (is_acceptable_0.pre | is_acceptable_1.pre).int(),
                    (is_acceptable_0.post | is_acceptable_1.post).int(),
                )

            if self.ew:
                right_count.pre += (prob_right_batch > prob_wrong_batch).int() * should_lemma_count.pre
                right_count.post += (prob_right_batch > prob_wrong_batch).int() * should_lemma_count.post
                total_count.pre += should_lemma_count.pre
                total_count.post += should_lemma_count.post
            else:
                if must_have_both_inflections:
                    total_prob_right.pre += prob_right_batch * should_lemma_count.pre
                    total_prob_right.post += prob_right_batch * should_lemma_count.post
                    total_prob_wrong.pre += prob_wrong_batch * should_lemma_count.pre
                    total_prob_wrong.post += prob_wrong_batch * should_lemma_count.post
                else: # MW, top-k
                    is_acceptable_right = PrePost(
                            ranks_batch.gather(1, label_values_batch) < last_accepted_rank,
                            ranks_batch.gather(1, label_values_batch) <= last_accepted_rank,
                            )
                    is_acceptable_wrong = PrePost(
                            ranks_batch.gather(1, 1 - label_values_batch) < last_accepted_rank,
                            ranks_batch.gather(1, 1 - label_values_batch) <= last_accepted_rank,
                            )
                    total_prob_right.pre += prob_right_batch * is_acceptable_right.pre.int()
                    total_prob_right.post += prob_right_batch * is_acceptable_right.post.int()
                    total_prob_wrong.pre += prob_wrong_batch * is_acceptable_wrong.pre.int()
                    total_prob_wrong.post += prob_wrong_batch * is_acceptable_wrong.post.int()

            prob_counted.pre += (prob_right_batch + prob_wrong_batch) * should_lemma_count.pre
            prob_counted.post += (prob_right_batch + prob_wrong_batch) * should_lemma_count.post

            # save some lemma stats - only once for top or bottom!
            if must_have_both_inflections:
                inf_s, inf_p = inflections
                self.lemma_stats[inf_s]["ranks"] = np.concatenate((self.lemma_stats[inf_s]["ranks"], ranks_batch[:, 0].cpu().numpy()))
                self.lemma_stats[inf_p]["ranks"] = np.concatenate((self.lemma_stats[inf_p]["ranks"], ranks_batch[:, 1].cpu().numpy()))
                self.lemma_stats[inf_s]["probs"] = np.concatenate((self.lemma_stats[inf_s]["probs"], sorted_probs.gather(1, ranks_batch[:, 0].unsqueeze(1)).squeeze(1).cpu().numpy()))
                self.lemma_stats[inf_p]["probs"] = np.concatenate((self.lemma_stats[inf_p]["probs"], sorted_probs.gather(1, ranks_batch[:, 1].unsqueeze(1)).squeeze(1).cpu().numpy()))
                self.lemma_stats[inf_s]["cum_probs"] = np.concatenate((self.lemma_stats[inf_s]["cum_probs"], cum_probs.gather(1, ranks_batch[:, 0].unsqueeze(1)).squeeze(1).cpu().numpy()))
                self.lemma_stats[inf_p]["cum_probs"] = np.concatenate((self.lemma_stats[inf_p]["cum_probs"], cum_probs.gather(1, ranks_batch[:, 1].unsqueeze(1)).squeeze(1).cpu().numpy()))

        if self.ew:
            scores = PrePost(
                    (right_count.pre / total_count.pre).cpu().numpy(),
                    (right_count.post / total_count.post).cpu().numpy()
                )
            invalid = PrePost(total_count.pre == 0, total_count.post == 0)
        else:
            scores = PrePost(
                    (total_prob_right.pre / (total_prob_right.pre + total_prob_wrong.pre)).cpu().numpy(),
                    (total_prob_right.post / (total_prob_right.post + total_prob_wrong.post)).cpu().numpy()
                )
            invalid = PrePost(
                    (total_prob_right.pre + total_prob_wrong.pre) == 0.0,
                    (total_prob_right.post + total_prob_wrong.post) == 0.0
                )

        scores.pre[invalid.pre] = np.NaN
        scores.post[invalid.post] = np.NaN
        return (k_mass_scores(scores.pre, prob_counted.pre.cpu().numpy(), natural_cutoff.pre.cpu().numpy()),
                 k_mass_scores(scores.post, prob_counted.post.cpu().numpy(), natural_cutoff.post.cpu().numpy()))

    def main_summary(self):
        """Creates the k vs score graph, with k being top-k or bottom-k proportion."""
        overall_scores_top = self.get_scores(
            self.cutoff_scores_by_example_top, self.cutoffs_top)
        overall_scores_bot = self.get_scores(
            self.cutoff_scores_by_example_bot, self.cutoffs_bot)
        return {
            "Cutoffs (top-k)": array_to_str(self.cutoffs_top),
            "Cutoffs (bottom-k)": array_to_str(self.cutoffs_bot),
            "Overall scores (top-k)": array_to_str(overall_scores_top),
            "Overall scores (bottom-k)": array_to_str(overall_scores_bot),
            "Prob counted (top-k)": array_to_str(np.mean(np.array(self.prob_counted_by_example_top), axis=0)),
            "Prob counted (bottom-k)": array_to_str(np.mean(np.array(self.prob_counted_by_example_bot), axis=0)),
            "% examples invalid (top-k)": array_to_str(np.mean(np.isnan(self.cutoff_scores_by_example_top), axis=0)),
            "% examples invalid (bottom-k)": array_to_str(np.mean(np.isnan(self.cutoff_scores_by_example_bot), axis=0)),
        }

    def get_scores(self, data, cutoffs):
        """Returns these computed overall scores."""
        scores = np.array(data)  # (examples, cutoffs)
        overall_scores = np.nanmean(scores, axis=0)  # (cutoffs)
        return overall_scores

    def report_state_summary(self):
        return {
                "Cutoff scores (top-k)": np.array(self.cutoff_scores_by_example_top),
                "Cutoff scores (bottom-k)": np.array(self.cutoff_scores_by_example_bot),
                "Prob counted (top-k)": np.array(self.prob_counted_by_example_top),
                "Prob counted (bottom-k)": np.array(self.prob_counted_by_example_bot),
                "Lemma Stats": self.lemma_stats,
                }
