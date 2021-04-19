"""Defines the basic dataset that stores minimal pairs.

Attributes:
    Min_Pair (namedtuple): The namedtuple for a minimal pair consisting of left/right context (lists of tokens),
        a label of the example as singular or plural, a template ID, and an example ID.
"""


from collections import namedtuple
import numpy as np
from torch.utils.data import IterableDataset


Min_Pair = namedtuple("Min_Pair", ["left_context", "right_context", "label", "template_id", "example_id"])


class MPEDataset(IterableDataset):
    """Abstract iterable dataset storing minimal pairs.

    Contains the notion of a template, i.e. different types of sentences.
    For some datasets, this may manifest as literal rule-based templates for sentence construction.
    In other datasets, a template might simply list all examples of that type.

    Args:
        config (dict): Config for initialization.

    Attributes:
        add_trailing_period (bool): Whether sentences should have periods appended.
        capitalize_first_word (bool): Whether sentences' first words should be capitalized.
    """
    def __init__(self, config):
        super().__init__()
        self.add_trailing_period = config["add_trailing_period"]
        self.capitalize_first_word = config["capitalize_first_word"]

        self._name = config["name"]
        self._max_examples_per_template = config.get("max_examples_per_template")
        if self._max_examples_per_template is None:
            self._max_examples_per_template = np.inf
        self._template_dir = config["template_dir"]
        self._template_files = config["template_files"]
        self._templates = {}

    @property
    def name(self):
        """str: Name of dataset"""
        return self._name

    @property
    def max_examples_per_template(self):
        """int: Number of max examples in a template. May be np.inf."""
        return self._max_examples_per_template

    @property
    def template_dir(self):
        """str: Path to directory with template files."""
        return self._template_dir

    @property
    def template_files(self):
        """str: Name of file specifying a template, e.g. via some rules or as a list of examples."""
        return self._template_files

    @property
    def templates(self):
        """dict: Maps template names to template-specific content."""
        return self._templates

    def initialize_templates(self):
        """Initializes `self.templates`. Children should call this after they have initialized
        fields that this function relies on.
        """
        raise NotImplementedError

    def postprocess_min_pair(self, left_context, right_context, label, template_id, example_id):
        """Builds minimal pair, potentially with some post-processing.

        Args:
            left_context (List[str]): The tokens of the sentence left of the masked verb.
            right_context (List[str]): The tokens of the sentence right of the masked verb.
            label (..constants.Number): Whether masked verb should be singular or plural.
            template_id: Identifier for a template.
            example_id: Identified for an example.

        Returns:
            Min_Pair: Processed minimal pair.
        """
        if self.capitalize_first_word and left_context and left_context[0]:
            left_context[0] = left_context[0].capitalize()
        if self.add_trailing_period:
            right_context.append(".")
        return Min_Pair(left_context, right_context, label, template_id, example_id)
