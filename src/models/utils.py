"""Some utility functions for string manipulation"""


# from https://stackoverflow.com/questions/15950672/join-split-words-and-punctuation-with-punctuation-in-the-right-place
def parse_respect_punctuation(seq, punctuation=".,;?!"):
    """Generator for parsing a sequence of strings, properly respecting punctuation.

    Args:
        seq (List[str]): List of strings to join.
        punctuation (str): Characters that are considered punctuation.

    Yields:
        str: Next portion of input sequence with trailing punctuation attached.
    """
    seq_iter = iter(seq)
    cur = next(seq_iter)
    for nxt in seq_iter:
        if nxt in punctuation:
            cur += nxt
        else:
            yield cur
            cur = nxt
    yield cur


def punctuated_join(seq, joining_character=" "):
    """Joins a sequence of strings, properly respecting punctuation.

    Args:
        seq (List[str]): List of strings to join.
        joining_character (str): Character to perform joining with.

    Returns:
        str: Joined string.
    """
    return joining_character.join(parse_respect_punctuation(seq))
