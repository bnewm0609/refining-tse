"""Defines basic model to undergo minimal pairs evaluation."""


class MPEModel(object):
    """Model being evaluated with minimal pairs."""
    def __init__(self, config):
        """Initializes model.

        Args:
            config (dict): Model-level config dict.
        """
        pass

    @property
    def word_to_index(self):
        """Maps a word to an index in model logits.
        Must support `dict`-like operations of `__getitem__`, `get`, and `__contains__`.
        """
        raise NotImplementedError

    def predict(self, left_context, right_context):
        """Returns logits for masked verb between left and right context.

        Args:
            left_context (List[List[str]]): A batch of lists of tokens to the left of the masked verb.
            right_context (List[List[str]]): A batch of lists of tokens to the right of the masked verb.

        Returns:
            torch.Tensor: Predicted logits with shape (vocab_size).
        """
        raise NotImplementedError
