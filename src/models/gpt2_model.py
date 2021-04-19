"""Wrapper for a GPT-2 model."""



import torch
import transformers

from .transformers_model import TransformersModel
from .utils import punctuated_join


class GPT2Model(TransformersModel):
    def __init__(self, config):
        super().__init__(
            config,
            tokenizer_cls=transformers.GPT2Tokenizer,
            model_cls=transformers.GPT2LMHeadModel,
            use_prefix_space=True,
            add_padding_token=True,
            bidirectional=False
        )

    # From: https://github.com/huggingface/transformers/issues/3021
    @torch.no_grad()
    def predict(self, left_contexts, right_contexts):
        inputs = [punctuated_join(left_context) for left_context in left_contexts]
        inputs_dict = self.tokenizer.batch_encode_plus(inputs, padding=True, add_prefix_space=False, return_tensors="pt")
        inputs = inputs_dict["input_ids"].to(self.device)
        attn_mask = inputs_dict["attention_mask"].to(self.device)

        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
        # get position ids
        position_ids = torch.tensor([list(range(inputs.shape[1])) for i in range(inputs.shape[0])]).to(self.device)
        for i, position_ids_slice in enumerate(position_ids):
            position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

        logits = self.model(inputs, attention_mask=attn_mask, position_ids=position_ids)[0]
        result = logits[torch.arange(len(left_contexts), device=self.device), last_non_masked_idx].cpu()
        return result
