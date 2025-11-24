import json
from .modeling_llada import LLaDAModel
from .configuration_llada import ModelConfig


def load_llada(
    d_model: int, 
    embedding_size: int ,
    n_layers: int,
    n_heads: int,
    eos_token_id: int,
    pad_token_id: int, 
    mask_token_id: int,
    rope_theta: int,
    weight_tying: bool = False, 
):
    config = ModelConfig(
        d_model=d_model,
        embedding_size=embedding_size,
        weight_tying=weight_tying,
        n_heads=n_heads, 
        n_kv_heads=n_heads,
        n_layers=n_layers,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        rope_theta=rope_theta,
        # llada default
        rope=True, 
        block_type='llama', 
        activation_type='silu',
        alibi=False,
        alibi_bias_max=8.0,
        attention_layer_norm=False,
        attention_layer_norm_with_affine=True,
        bias_for_layer_norm=False,
        block_group_size=1,
        include_bias=False,
        include_qkv_bias=False,
        input_emb_norm=False,
        layer_norm_type='rms',
        layer_norm_with_affine=True,
        multi_query_attention=None,
        rope_full_precision=True,
        scale_logits=False, 
    )

    model = LLaDAModel(config)

    return model 


def load_llada_from_config(
    config
):
    with open(config, 'r') as fp:
        kwargs = json.load(fp)
    config = ModelConfig(**kwargs)
    model = LLaDAModel(config)

    return model 