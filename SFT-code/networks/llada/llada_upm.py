import copy
from tqdm import tqdm
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dnnlib
from typing import NamedTuple, Optional, List, Tuple
from networks.llada.modeling_llada import LLaDAModelLM, LLaDABlock, LLaDALlamaBlock, LLaDAOutput
from networks.llada.layers import TimestepEmbedder, AdaLayerNormContinuous


class LLaDAUPMOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """
    upm_logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """


class LLaDAWithMaskHead(LLaDAModelLM):
    def __init__(self, *args, **kwargs):
        if not hasattr(args[0], "num_head_layers"):
            setattr(args[0], "num_head_layers", kwargs.pop("num_head_layers", 1))
        if not hasattr(args[0], "use_feat_layer"):
            setattr(args[0], "use_feat_layer", kwargs.pop("use_feat_layer", -1))
        if not hasattr(args[0], "zero_init_block"):
            setattr(args[0], "zero_init_block", kwargs.pop("zero_init_block", False))
        if len(kwargs):
            print("Warning, there are unused kwargs", kwargs)
        super().__init__(*args)
        total_layers = len(self.model.transformer.blocks)
        self.mask_head = nn.ModuleList([
            LLaDABlock.build(i + total_layers, self.model.config, self.model.alibi_cache) for i in range(self.config.num_head_layers)
        ])
        self.hidden_size = self.config.hidden_size
        self.norm_out_1 = AdaLayerNormContinuous(self.hidden_size, self.hidden_size, eps=1e-4)
        self.mask_linear = nn.Linear(4096, 1)
        setattr(self.mask_linear, "is_last_linear", True)

        self.time_embedding = TimestepEmbedder(hidden_size=self.hidden_size)
        self.mask_embedding = nn.Embedding(2, 4096)
        self.norm_out_2 = AdaLayerNormContinuous(self.hidden_size, self.hidden_size, eps=1e-4)

        # self.reset_dropout()

    def reset_dropout(self):
        for m in self.modules():
            # Only override for layers where behavior changes between train/eval
            if isinstance(m, (
                nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                nn.AlphaDropout,
            )):
                m.p = 0  # Force eval behavior

    def _init_weights(self, module):
        if isinstance(module, LLaDALlamaBlock) or isinstance(module, AdaLayerNormContinuous) or isinstance(module, TimestepEmbedder):
            module.reset_parameters()
            if isinstance(module, LLaDALlamaBlock) and self.config.zero_init_block:
                module.ff_out.weight.data.zero_()
                module.attn_out.weight.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.embedding_dim == 1:
                module.weight.data.copy_(torch.tensor([[-2],[2]]))
        elif hasattr(module, "is_last_linear") or hasattr(module, "is_input_linear"):
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_()

    def get_mask_prob(self, hidden_states, timestep, **kwargs):
        temb = self.time_embedding(timestep)
        mask_feat = self.mask_embedding(kwargs["mask_index"].int())
        temb = temb + mask_feat
        hidden_states = self.norm_out_1(hidden_states, temb)
        f = hidden_states

        for layer in self.mask_head:
            f, _ = layer(
                f, 
                attention_bias=kwargs["attention_bias"],
                position_ids=kwargs["position_ids"],
            )
        f = self.norm_out_2(f, temb)
        logits = self.mask_linear(f).squeeze(-1)

        logits = logits.float()
        return logits

    def forward(self, *args, **kwargs):
        timestep = kwargs.pop("timestep")
        mask_index = kwargs.pop("mask_index")
        out = super().forward(*args, **kwargs, output_hidden_states=True)

        attention_bias = kwargs.get("attention_bias", None)
        position_id = kwargs.get("position_ids", None)
        # torch.distributed.breakpoint()
        upm_logits = self.get_mask_prob(
            out.hidden_states[self.config.use_feat_layer], 
            timestep=timestep,
            mask_index=mask_index,
            attention_bias=attention_bias,
            position_ids=position_id,
        )

        return LLaDAUPMOutput(
            logits=out.logits,
            attn_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            upm_logits=upm_logits
        )

