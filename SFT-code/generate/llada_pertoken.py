import torch 
import numpy as np
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    elif method == 'max':
        return categorical_probs.argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    

def get_num_transfer_tokens(mask_num, steps, device):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(steps, device=device, dtype=torch.int64) + base

    num_transfer_tokens[:remainder] += 1

    return num_transfer_tokens


@torch.no_grad()
def sample(
    model, 
    batch, 
    tokenizer, 
    reward_fn, 
    device, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    gamma=0.9, 
    mask_id=126336, 
    eos_id=126081, 
    is_resample: bool = False
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (b, l).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    prob_dtype = torch.float64
    # prepare input tokens id
    problems = batch['problems']
    m = [[{"role": "user", "content": prompt}] for prompt in problems]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, )
    prompt = inputs['input_ids'].to(device)
    prompt_len = inputs['attention_mask'].sum(dim=1)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    # set eos_id to the last position of the generated answer
    for i in range(x.shape[0]):
        x[i, prompt_len[i] + gen_length:] = eos_id
        
    trajectory_inputs = []
    trajectory_outputs = []
    remask_prob = None
    to_be_remasked_list = []

    prompt_len = prompt_len.to(device)
    prompt_index = torch.arange(x.shape[1], device=device) < prompt_len[:, None]
    generation_index = ~prompt_index
    # T_max
    stop_steps = torch.full((x.shape[0],), steps, dtype=torch.long, device=device)
    for step in range(steps):
        mask_index = (x == mask_id)
        not_set_yet = (stop_steps == steps)
        finished = (mask_index.sum(dim=1) == 0)
        # finished & 标志位为设置
        newly_finished = finished & not_set_yet
        stop_steps[newly_finished] = step
        
        # record model inputs
        trajectory_inputs.append(x.clone())
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            logits = model(x).logits

        p = F.softmax(logits.to(prob_dtype), dim=-1)
        x0 = sample_categorical(p, 'max')
        # keep prompts & generated tokens unchanged
        x0 = torch.where(mask_index, x0, x)

        # TODO: 与单调递减进行对比
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            remask_prob = model(x0, x, is_predict_mask=True)
        remask_prob = remask_prob.to(prob_dtype)
        condition_index = generation_index if is_resample else mask_index
        # zero prompts' prob if is_resample 
        # else zero only mask's prob
        remask_prob = torch.where(condition_index, remask_prob, torch.zeros_like(remask_prob))

        to_be_remasked = torch.rand_like(remask_prob) < remask_prob
        # TODO: 如果 remask_flag.sum(dim=-1) 且 is_resample = False 有某个样本完全 remask 如何处理？说明当前的生成没有任何意义
        # 可以改成除 top-1 外是 sample 的？
        # 只 remask 没生成好的样本
        global_mask = (step < stop_steps).unsqueeze(1).expand_as(to_be_remasked)
        to_be_remasked = to_be_remasked & global_mask
        # remask_flag[step >= stop_steps] = False

        # record model outputs
        trajectory_outputs.append(x0.clone())
        to_be_remasked_list.append(to_be_remasked)

        x = torch.where(to_be_remasked, mask_id, x0)

    # compute reward 
    answers = batch['answers'] 
    responses = tokenizer.batch_decode(x0, skip_special_tokens=True)
    rewards = reward_fn(answers, responses, device).float()

    # discount
    ts = torch.arange(steps, device=rewards.device)  # [T] -> [0,1...,T]
    discount_factors = ((gamma ** ts) * (ts[None, :] < stop_steps[:, None])).sum(dim=1) / (stop_steps + 1e-8)
    rewards = rewards * discount_factors

    output_dict = {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'to_be_remasked_list': to_be_remasked_list,
        'samples': x,
        'prompt_len': prompt_len,
        'steps': steps,
        'gen_length': gen_length,
        'block_length': block_length,
        'rewards': rewards,
        'mask_id': mask_id,
        'stop_steps': stop_steps,
        'prompt_index': prompt_index,
        'is_resample': is_resample,
    }
    
    return output_dict