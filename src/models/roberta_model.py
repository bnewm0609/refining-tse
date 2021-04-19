"""Wrapper for a RoBERTa model."""

import torch
import transformers

from .transformers_model import TransformersModel
from .utils import punctuated_join


class RobertaModel(TransformersModel):
    def __init__(self, config):
        super().__init__(
            config,
            tokenizer_cls=transformers.RobertaTokenizer,
            model_cls=transformers.RobertaForMaskedLM,
            use_prefix_space=True,
        )

    @torch.no_grad()
    def predict(self, left_contexts, right_contexts):
        prior_lens = [len(l_ctx) + len(r_ctx) + 1 for l_ctx, r_ctx in zip(left_contexts, right_contexts)]
        left_contexts = [punctuated_join(l_ctx) for l_ctx in left_contexts]
        right_contexts = [punctuated_join(r_ctx) for r_ctx in right_contexts]
        mask_idx = [len(self.tokenizer.tokenize(l_ctx)) + 1 for l_ctx in left_contexts]
        inputs = [punctuated_join([l_ctx, self.tokenizer.mask_token, r_ctx]) for l_ctx, r_ctx in zip(left_contexts, right_contexts)]
        inputs = self.tokenizer(inputs, padding=True, add_special_tokens=True, add_prefix_space=True, return_tensors="pt")
        logits = self.model(input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))
        logits = logits[0]
        return logits[torch.arange(len(left_contexts)), mask_idx].cpu()
