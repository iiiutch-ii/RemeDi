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
def sample_trajectory_and_score(
    model, ref_model, batch, tokenizer, reward_fn, device, num_generations, steps=128, 
    gen_length=128, block_length=128, mask_id=126336, eos_id=126081):
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
    x = x.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    trajectory_inputs = []
    trajectory_outputs = []
    ref_log_probs = []
    transfer_indices = []
    log_prob = torch.zeros((x.shape[0], block_length), device=device, dtype=prob_dtype)
    for num_block in range(num_blocks):
        num_transfer_tokens = get_num_transfer_tokens(block_length, steps, x.device) # (steps,)
        
        for step in range(steps):
            # record model inputs
            trajectory_inputs.append(x.clone())
            
            mask_index = (x == mask_id)
            logits = model(x).logits
            ref_logits = ref_model(x).logits
                
            p = F.softmax(logits.to(prob_dtype), dim=-1)
            ref_log_p = F.log_softmax(ref_logits.to(prob_dtype), dim=-1)
            x0 = sample_categorical(p)
            
            # record model outputs
            trajectory_outputs.append(x0.clone())
            
            x0_p = p.gather(dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
            ref_logpx = ref_log_p.gather(dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
            log_prob.zero_()
            for i in range(x0_p.shape[0]):
                x0_p[i, prompt_len[i] + (num_block + 1) * block_length:] = -np.inf
                block_slice = slice(prompt_len[i] + num_block * block_length, prompt_len[i] + (num_block + 1) * block_length)
                log_prob[i] = ref_logpx[i, block_slice]
            ref_log_probs.append(log_prob.clone())
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[step]) # num_transfer_tokens[:, i] all the same
                transfer_index[j, select_index] = True
            transfer_indices.append(transfer_index)
            x[transfer_index] = x0[transfer_index]

    # compute reward 
    test_inputs = batch["test_inputs"] * num_generations
    test_outputs = batch["test_outputs"] * num_generations
    responses = tokenizer.batch_decode(x, skip_special_tokens=True)
    # print(test_inputs)
    # print("---------------")
    # print(test_outputs)
    # print("---------------")
    # print(responses)
    rewards = reward_fn(responses, test_inputs, test_outputs, device).float()
    rewards_mean = rewards.view(num_generations, -1).mean(dim=0).repeat(num_generations,)
    rewards_std = rewards.view(num_generations, -1).std(dim=0).repeat(num_generations,)
    advantages = (rewards - rewards_mean) / (rewards_std + 1e-4)

    output_dict = {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'transfer_indices': transfer_indices,
        'samples': x,
        'prompt_len': prompt_len,
        'steps': steps,
        'gen_length': gen_length,
        'block_length': block_length,
        'ref_log_probs': ref_log_probs,
        'advantages': advantages,
        'rewards': rewards,
        'mask_id': mask_id
    }

    return output_dict

