"""Defines a base wrapper for a Transformers model."""


import torch

from .MPE_model import MPEModel


class TransformersIndexer(object):
    """Indexer for a Transformer that supports `dict`-like methods."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.cache = {}

    @property
    def tokenizer(self):
        """https://huggingface.co/transformers/main_classes/tokenizer.html"""
        return self._tokenizer

    def __getitem__(self, token):
        """Get the ID of a token.

        Args:
            token (str): Query token.

        Returns:
            int: Index in vocabulary.
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get(self, token, default_value=None):
        """Get the ID of a token, or a default value if the token isn't found.

        Args:
            token (str): Query token.
            default_value (int): Value to return when key isn't found in vocab.

        Returns:
            int: Index of key, else the default value.
        """
        value = self[token]
        if value == self.tokenizer.unk_token_id:
            return default_value
        return value

    def __contains__(self, token):
        """Returns whether a token is in the vocab.

        Args:
            token (str): Query token.

        Returns:
            bool: Whether the token is in the vocab.
        """
        return self.get(token) is not None


class TransformersPrefixSpaceIndexer(TransformersIndexer):
    """Indexer for Transformers whose tokenization scheme includes a prefix space
    such as GPT2 and RoBERTa
    """
    def __getitem__(self, token):
        encoding = self.cache.get(token)
        if encoding is not None:
            return self.cache[token]

        encoding = self.tokenizer.encode(token, add_prefix_space=True, add_special_tokens=False)
        if len(encoding) != 1:
            encoding = self.tokenizer.unk_token_id
        else:
            encoding = encoding[0]
        self.cache[token] = encoding
        return encoding


class TransformersModel(MPEModel):
    """General form of a Transformer MPE model.

    Attributes:
        device: GPU or CPU device.
        tokenizer: Transformer tokenizer.
        model: Transformer model to make predictions.
        indexer (TransformersIndexer): An indexer to convert words into indices.
    """
    def __init__(self, config, tokenizer_cls, model_cls, use_prefix_space=False, add_padding_token=False, cache_dir=None, bidirectional=True):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        version = config["name"]
        if cache_dir is not None:
            self.model = model_cls.from_pretrained(version, cache_dir=cache_dir).to(self.device)
        else:
            self.model = model_cls.from_pretrained(version).to(self.device)
        if add_padding_token:
            self.tokenizer = tokenizer_cls.from_pretrained(version)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer_cls.from_pretrained(version)

        if use_prefix_space:
            self.indexer = TransformersPrefixSpaceIndexer(self.tokenizer)
        else:
            self.indexer = TransformersIndexer(self.tokenizer)

        self.bidirectional = bidirectional

    @property
    def word_to_index(self):
        return self.indexer

    @torch.no_grad()
    def predict(self, left_context, right_context):
        pass
