import torch 
from .dataloader import register_reward
from latex2sympy2_extended import NormalizationConfig
from math_verify import (
    verify, 
    parse,
    LatexExtractionConfig
)

def catch_verify(*args, **kwargs):
    try:
        return verify(*args, **kwargs)
    except:
        return False

@register_reward('general-math')
def reward_func(
    batch, 
    responses
):
    extraction_config = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            # Ensures that boxed is tried first
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]
    # answer rewards
    ext_ans = [parse(batch['answers'][0], extraction_mode="first_match",)] * len(responses)
    ext_res = [parse(response, extraction_mode="first_match", extraction_config=extraction_config) for response in responses]
    rewards = torch.zeros(len(responses))
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if catch_verify(ans, res):
            rewards[i] += 1.0

    return rewards  


def _extract_answer_gsm8k(answer: str):
    return answer.split('####')[-1].strip()

@register_reward('gsm8k')
def reward_func_gsm8k(
    batch, 
    responses
):
    extraction_config = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            # Ensures that boxed is tried first
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]
    # answer rewards
    ext_ans = [parse(_extract_answer_gsm8k(batch['answers'][0]), extraction_mode="first_match",)] * len(responses)
    ext_res = [parse(response, extraction_mode="first_match", extraction_config=extraction_config) for response in responses]
    rewards = torch.zeros(len(responses))
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if catch_verify(ans, res):
            rewards[i] += 1.0

    return rewards 