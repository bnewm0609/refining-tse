"""Defines a dataset with explicitly listed examples.

Attributes:
    TemplateWithSpecificLemma (namedtuple): The namedtuple for a template that also has lemma information.
"""


from collections import namedtuple
import json
import os
import string

from ..constants import MaskedVerb_to_Number
from .MPE_dataset import MPEDataset


TemplateWithSpecificLemma = namedtuple("TemplateWithSpecificLemma", ["template_id", "lemma_inflections"])


class CustomDataset(MPEDataset):
    """Dataset with explicitly listed examples."""
    def __init__(self, config):
        super().__init__(config)
        self.initialize_templates()

    @staticmethod
    def load_custom_templates_from_json(template_dir, files):
        """Loads of templates from `.jsonl` files in a certain directory.

        Args:
            template_dir (str): Directory with template files.
            files (List[str]): List of `.jsonl` files in `template_dir`.

        Returns:
            dict: Maps a filename to a list of JSON-parsed dicts, one per line.
        """
        templates = {}
        for filename in files:
            with open(os.path.join(template_dir, filename)) as f:
                templates[filename] = [json.loads(line.strip()) for line in f]
        return templates

    def initialize_templates(self):
        """Loads templates from `.jsonl` files into `self.templates`."""
        self.templates.update(CustomDataset.load_custom_templates_from_json(self.template_dir, self.template_files))

    def __iter__(self):
        """Generator for iterating through examples, template by template.

        Yields:
            Min_Pair: Next minimal pair.
        """
        for template_id, template in self.templates.items():
            max_iter = min(self.max_examples_per_template, len(template))
            for i in range(max_iter):
                yield self.make_min_pair_from_example_id(i, template_id)

    def make_min_pair_from_example_id(self, example_id, template_id):
        """Extracts minimal pair for specific template and example.

        Args:
            example_id (int): Index of example in template.
            template_id (str): Identifier for template.

        Returns:
            Min_Pair: Constructed minimal pair.
        """
        example = self.templates[template_id][example_id]

        # Remove trailing punctuation.
        correct_sentence = example['sentence_good'].strip(string.punctuation).split()
        incorrect_sentence = example['sentence_bad'].strip(string.punctuation).split()

        # Find index where sentences differ.
        mask_idx = [i for i, p in enumerate(zip(correct_sentence, incorrect_sentence)) if p[0] != p[1]]
        if len(mask_idx) == 0:
            raise ValueError(f"Template {template_id}, example {example_id}: "
                             "correct and incorrect sentences are identical.")
        elif len(mask_idx) > 1:
            raise ValueError(f"Template {template_id}, example {example_id}: "
                             "correct and incorrect sentences differ in more than 1 word.")
        mask_idx = mask_idx[0]

        # Extract label information.
        correct_verb = correct_sentence[mask_idx]
        incorrect_verb = incorrect_sentence[mask_idx]
        label = MaskedVerb_to_Number[int(example['label'])]

        # The MLMetricComputer needs knowledge of the lemma identified
        # with each example, hence we associate each example not just with
        # a template but also with the correct/incorrect inflections.
        extended_template_id = TemplateWithSpecificLemma(template_id, (correct_verb, incorrect_verb))

        left_context = correct_sentence[:mask_idx]
        right_context = correct_sentence[mask_idx + 1:]

        return self.postprocess_min_pair(left_context, right_context, label, extended_template_id, example_id)
