"""Defines constants, e.g. labeling a masked verb or an example overall."""


from enum import Enum


class Number(Enum):
    SINGULAR = 0
    PLURAL = 1


class MaskedVerb(Enum):
    MASKED_SINGULAR_VERB = -1
    MASKED_PLURAL_VERB = -2


MaskedVerb_to_Number = {
    MaskedVerb.MASKED_SINGULAR_VERB.value: Number.SINGULAR,
    MaskedVerb.MASKED_PLURAL_VERB.value: Number.PLURAL
}


INVALID_SCORE_VALUE = None


BATCH_SIZE = 64
