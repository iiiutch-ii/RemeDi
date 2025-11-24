import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils.distributed as dist
from networks.block_llada.modeling_llada import DynamicCache
from general_data.dataloader import universal_reward_func


def construct_block_attention_bias(
    resp_seqlen: int, 
    prompt_len: int,
    block_size: int,
):
    seq_idx = torch.arange(resp_seqlen * 2, dtype=torch.long)
    is_x0 = seq_idx < resp_seqlen
    is_x0_q = is_x0.unsqueeze(1)
    is_x0_kv = is_x0.unsqueeze(0)
    block_q = torch.where(is_x0, seq_idx // block_size, (seq_idx - resp_seqlen) // block_size).unsqueeze(1)
    block_kv = torch.where(is_x0, seq_idx // block_size, (seq_idx - resp_seqlen) // block_size).unsqueeze(0)

    # 1. diagonal 
    block_diag = (block_q == block_kv) & (is_x0_q == is_x0_kv) # (L, L)
    # 2. offset block causal 
    block_offset = (~is_x0_q) & (is_x0_kv) & (block_q > block_kv)
    # 3. block causal
    block_causal = (is_x0_q) & (is_x0_kv) & (block_q >= block_kv)
    attn_mask = block_diag | block_offset | block_causal
    attn_bias = torch.where(attn_mask, 0, -torch.inf)

    # prompt attn_bias should be full-attn 
    attn_bias = F.pad(attn_bias, (prompt_len, 0, prompt_len, 0), value=0)
    attn_bias[:prompt_len, prompt_len:] = -torch.inf

    return attn_bias


@torch.no_grad()
def generate_block_diffusion_remdm(
    model, 
    batch,
    tokenizer, 
    reward_fn, 
    device,
    num_generations,
    temperature: float = 1.0,
    steps: int = 32,
    max_length = 1024,
    block_size = 32, 
    mask_token_id = 126336,
    eos_id = 126081,
    sample: bool = True,
    repeat_times = None,
    viz: bool = False,
    use_general_reward: bool = False,
    upm_temperature: float = 1.0,
    *args, **kwargs
):
    """
    Generate block-wise responses using block diffusion. KV Cache is verified.
    """
    # process batch 
    if 'fwd_num_generations' in kwargs:
        num_generations = kwargs['fwd_num_generations']
    if use_general_reward:
        problems = batch['prompts']
    else:
        problems = batch['problems']
    m = [[{"role": "user", "content": prompt}] for prompt in problems]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    x_t = inputs['input_ids'].to(device)

    attention_mask = inputs['attention_mask'].to(device)
    prompt_len = attention_mask.sum(dim=1)
    attn_bias = torch.where(
        attention_mask + attention_mask.T > 0,
        0, -torch.inf
    )[None, None].repeat(x_t.shape[0], 1, 1, 1)
    
    x_t = x_t.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)
    attn_bias = attn_bias.repeat(num_generations, 1, 1, 1)
    batch_size = x_t.shape[0]
    
    position_ids = torch.arange(x_t.shape[1], device=x_t.device, dtype=torch.long).unsqueeze(0) - (1 - attention_mask).sum(dim=-1)
    kv_cache = DynamicCache()

    # cache prompt first
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        model(
            x_t,
            kv_cache=kv_cache,
            update_kv_cache=True,
        )
    cur_blocks = 0
    responses = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)
    # return lists
    # first_kvcache = deepcopy(kv_cache)
    sample_orders = []
    trajectory_inputs = []
    trajectory_outputs = []
    ranking_logprob_list = []
    unmask_index_list = []
    token_logprob_list = []
    is_valid_steps = []

    remask_times = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)

    # for visualization
    if viz:
        cur_response = x_t
        visualize_xt = [x_t]
        visualize_x0 = [x_t]
        cur_upm_prob = torch.zeros_like(x_t)
        visualize_upm_prob = [torch.zeros_like(x_t)]
    else:
        visualize_xt = None
        visualize_x0 = None
        visualize_upm_prob = None

    total_blocks = 0
    sample_blocks = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)
    while (cur_blocks * block_size) < max_length:
        x_t = torch.full((batch_size, block_size), fill_value=mask_token_id, device=device, dtype=torch.long)
        
        #计算此时的position_id
        position_ids = torch.arange(
            cur_blocks * block_size, 
            (cur_blocks + 1) * block_size, 
            device=x_t.device, dtype=torch.long).unsqueeze(0) + prompt_len.unsqueeze(1)

        #定义每一步的token保留数目
        num_transfer_tokens = torch.tensor([block_size // steps for _ in range(steps)])
        if block_size % steps != 0:
            num_transfer_tokens[-block_size % steps:] += 1
        # cumsum 
        num_transfer_tokens = num_transfer_tokens.cumsum(dim=0)

        alpha_s = 1.0
        eta_cap = 0.04
        t_on = 0.55
        t_off = 0.05
        #进行steps去噪
        token_per_step = block_size // steps
        # for i in range(steps):
        while mask_token_id in x_t:
            t = (x_t == mask_token_id).sum().float() / block_size
            alpha_t = 1 - t 
            sigma_t = min(eta_cap, (1 - alpha_s) / alpha_t)
            if t > t_on or t < t_off:
                sigma_t = 0.0

            is_valid_steps.append((~is_eos_meet).clone())

            mask_index = (x_t == mask_token_id)
            trajectory_inputs.append(x_t.clone())
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )
            logits = out.logits.to(torch.float32)
            
            # sample tokens
            x0 = torch.argmax(logits, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x_t)

            # sample position
            x0_logits = logits.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_logits = torch.where(mask_index, x0_logits, -torch.inf, )
            samples = torch.topk(x0_logits, k=token_per_step, dim=-1).indices

            bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
            remask_index = torch.ones_like(x_t).bool()
            remask_index[bs_idx, samples] = False
            remask_index[~mask_index] = False
            sample_orders.append(samples)

            unmask_index = (~remask_index & mask_index) # & (~is_eos_meet.unsqueeze(1))
    
            x_t = torch.where(remask_index, mask_token_id, x0)

            # ReMDM step
            is_remask = (torch.rand_like(x_t.float()) < sigma_t)
            is_remask = is_remask & (~mask_index)
            x_t = torch.where(is_remask, mask_token_id, x_t)
            alpha_s = alpha_t
            remask_tokens = is_remask.sum()

            unmask_index_list.append(unmask_index)

        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        # setting x_t to eos_id if is_eos_meet
        x_t = torch.where(is_eos_meet.unsqueeze(1), eos_id, x_t)
        # gather all is_eos_meet
        tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(dist.get_world_size())]
        torch.distributed.barrier()
        torch.distributed.all_gather(tensor_list, is_eos_meet)
        is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        total_blocks += (~is_eos_meet_all_rank).float().sum()
        sample_blocks += (~is_eos_meet).float()
        if is_eos_meet_all_rank.all(): break

        # update kv_cache if not ends
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                position_ids=position_ids,
                kv_cache=kv_cache,
                update_kv_cache=True,
            )

    response_tokens = torch.cat(responses, dim=1)
    responses = []
    responses_length = []
    for i in range(batch_size):
        if eos_id in response_tokens[i]:
            eos_token_idx = (response_tokens[i] == eos_id).nonzero(as_tuple=True)[0][0].item()
            resp_token = response_tokens[i, prompt_len[i]:eos_token_idx]
        else:
            resp_token = response_tokens[i, prompt_len[i]:]
        responses.append(tokenizer.decode(resp_token, skip_special_tokens=True))
        responses_length.append(resp_token.shape[0])
    responses_length = torch.tensor(responses_length, device=device)
    if use_general_reward:
        rewards = universal_reward_func(batch, responses)
        rewards = rewards.float().to(device)
    else:
        rewards = reward_fn(batch, responses, num_generations, device).float() if reward_fn is not None else torch.zeros(batch_size, device=device)

    return {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'sample_orders': sample_orders,
        'unmask_index_list': unmask_index_list,
        'ranking_logprob_list': ranking_logprob_list,
        'is_valid_steps': is_valid_steps,
        'responses': responses,
        'response_tokens': response_tokens,
        'rewards': rewards,
        'prompt_len': prompt_len,
        'token_logprob_list': token_logprob_list,

        'block_size': block_size,
        'max_length': max_length,
        'steps': steps,
        'remask_times': remask_times,

        'visualize_x0': visualize_x0,
        'visualize_xt': visualize_xt,
        'visualize_upm_prob': visualize_upm_prob,
        'total_blocks': total_blocks,

        'responses_length': responses_length,
        'sample_blocks': sample_blocks
    }




@torch.no_grad()
def generate_block_diffusion_vanilla(
    model, 
    batch,
    tokenizer, 
    reward_fn, 
    device,
    num_generations,
    temperature: float = 1.0,
    steps: int = 32,
    max_length = 1024,
    block_size = 32, 
    mask_token_id = 126336,
    eos_id = 126081,
    sample: bool = True,
    repeat_times = None,
    viz: bool = False,
    use_general_reward: bool = False,
    upm_temperature: float = 1.0,
    *args, **kwargs
):
    """
    Generate block-wise responses using block diffusion. KV Cache is verified.
    """
    # process batch 
    if 'fwd_num_generations' in kwargs:
        num_generations = kwargs['fwd_num_generations']
    if use_general_reward:
        problems = batch['prompts']
    else:
        problems = batch['problems']
    m = [[{"role": "user", "content": prompt}] for prompt in problems]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    x_t = inputs['input_ids'].to(device)

    attention_mask = inputs['attention_mask'].to(device)
    prompt_len = attention_mask.sum(dim=1)
    attn_bias = torch.where(
        attention_mask + attention_mask.T > 0,
        0, -torch.inf
    )[None, None].repeat(x_t.shape[0], 1, 1, 1)
    
    x_t = x_t.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)
    attn_bias = attn_bias.repeat(num_generations, 1, 1, 1)
    batch_size = x_t.shape[0]
    
    position_ids = torch.arange(x_t.shape[1], device=x_t.device, dtype=torch.long).unsqueeze(0) - (1 - attention_mask).sum(dim=-1)
    kv_cache = DynamicCache()

    # cache prompt first
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        model(
            x_t,
            kv_cache=kv_cache,
            update_kv_cache=True,
        )
    cur_blocks = 0
    responses = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)
    # return lists
    # first_kvcache = deepcopy(kv_cache)
    sample_orders = []
    trajectory_inputs = []
    trajectory_outputs = []
    ranking_logprob_list = []
    unmask_index_list = []
    token_logprob_list = []
    is_valid_steps = []

    remask_times = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)

    # for visualization
    if viz:
        cur_response = x_t
        visualize_xt = [x_t]
        visualize_x0 = [x_t]
        cur_upm_prob = torch.zeros_like(x_t)
        visualize_upm_prob = [torch.zeros_like(x_t)]
    else:
        visualize_xt = None
        visualize_x0 = None
        visualize_upm_prob = None

    total_blocks = 0
    sample_blocks = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)
    while (cur_blocks * block_size) < max_length:
        x_t = torch.full((batch_size, block_size), fill_value=mask_token_id, device=device, dtype=torch.long)
        
        #计算此时的position_id
        position_ids = torch.arange(
            cur_blocks * block_size, 
            (cur_blocks + 1) * block_size, 
            device=x_t.device, dtype=torch.long).unsqueeze(0) + prompt_len.unsqueeze(1)

        #定义每一步的token保留数目
        num_transfer_tokens = torch.tensor([block_size // steps for _ in range(steps)])
        if block_size % steps != 0:
            num_transfer_tokens[-block_size % steps:] += 1
        # cumsum 
        num_transfer_tokens = num_transfer_tokens.cumsum(dim=0)

        #进行steps去噪
        token_per_step = block_size // steps
        for i in range(steps):
            is_valid_steps.append((~is_eos_meet).clone())

            mask_index = (x_t == mask_token_id)
            trajectory_inputs.append(x_t.clone())
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )
            logits = out.logits.to(torch.float32)
            prob = F.softmax(logits / temperature, dim=-1)

            x0 = torch.argmax(logits, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x_t)
            x0_p = prob.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p = torch.where(mask_index, x0_p, -torch.inf)

            samples = torch.topk(x0_p, k=token_per_step, dim=-1).indices

            bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
            remask_index = torch.ones_like(x_t).bool()
            remask_index[bs_idx, samples] = False
            remask_index[~mask_index] = False
            sample_orders.append(samples)

            unmask_index = (~remask_index & mask_index) # & (~is_eos_meet.unsqueeze(1))
            x_t = torch.where(remask_index, mask_token_id, x0)
            unmask_index_list.append(unmask_index)
                  
            remask_times += (remask_index & (~mask_index) & (~is_eos_meet.unsqueeze(1))).float().sum(dim=-1)

        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        # setting x_t to eos_id if is_eos_meet
        x_t = torch.where(is_eos_meet.unsqueeze(1), eos_id, x_t)
        # gather all is_eos_meet
        tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(dist.get_world_size())]
        torch.distributed.barrier()
        torch.distributed.all_gather(tensor_list, is_eos_meet)
        is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        total_blocks += (~is_eos_meet_all_rank).float().sum()
        sample_blocks += (~is_eos_meet).float()
        if is_eos_meet_all_rank.all(): break

        # update kv_cache if not ends
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                position_ids=position_ids,
                kv_cache=kv_cache,
                update_kv_cache=True,
            )

    response_tokens = torch.cat(responses, dim=1)
    responses = []
    responses_length = []
    for i in range(batch_size):
        if eos_id in response_tokens[i]:
            eos_token_idx = (response_tokens[i] == eos_id).nonzero(as_tuple=True)[0][0].item()
            resp_token = response_tokens[i, prompt_len[i]:eos_token_idx]
        else:
            resp_token = response_tokens[i, prompt_len[i]:]
        responses.append(tokenizer.decode(resp_token, skip_special_tokens=True))
        responses_length.append(resp_token.shape[0])
    responses_length = torch.tensor(responses_length, device=device)
    if use_general_reward:
        rewards = universal_reward_func(batch, responses)
        rewards = rewards.float().to(device)
    else:
        rewards = reward_fn(batch, responses, num_generations, device).float() if reward_fn is not None else torch.zeros(batch_size, device=device)

    return {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'sample_orders': sample_orders,
        'unmask_index_list': unmask_index_list,
        'ranking_logprob_list': ranking_logprob_list,
        'is_valid_steps': is_valid_steps,
        'responses': responses,
        'response_tokens': response_tokens,
        'rewards': rewards,
        'prompt_len': prompt_len,
        'token_logprob_list': token_logprob_list,

        'block_size': block_size,
        'max_length': max_length,
        'steps': steps,
        'remask_times': remask_times,

        'visualize_x0': visualize_x0,
        'visualize_xt': visualize_xt,
        'visualize_upm_prob': visualize_upm_prob,
        'total_blocks': total_blocks,

        'responses_length': responses_length,
        'sample_blocks': sample_blocks
    }



@torch.no_grad()
def generate_block_diffusion_adaptive(
    model, 
    batch,
    tokenizer, 
    reward_fn, 
    device,
    num_generations,
    temperature: float = 1.0,
    steps: int = 32,
    max_length = 1024,
    block_size = 32, 
    mask_token_id = 126336,
    eos_id = 126081,
    sample: bool = True,
    repeat_times = None,
    viz: bool = False,
    use_general_reward: bool = False,
    upm_temperature: float = 1.0,
    *args, **kwargs
):
    """
    Generate block-wise responses using block diffusion. KV Cache is verified.
    """
    # process batch 
    if 'fwd_num_generations' in kwargs:
        num_generations = kwargs['fwd_num_generations']
    if use_general_reward:
        problems = batch['prompts']
    else:
        problems = batch['problems']
    m = [[{"role": "user", "content": prompt}] for prompt in problems]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    x_t = inputs['input_ids'].to(device)

    attention_mask = inputs['attention_mask'].to(device)
    prompt_len = attention_mask.sum(dim=1)
    attn_bias = torch.where(
        attention_mask + attention_mask.T > 0,
        0, -torch.inf
    )[None, None].repeat(x_t.shape[0], 1, 1, 1)
    
    x_t = x_t.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)
    attn_bias = attn_bias.repeat(num_generations, 1, 1, 1)
    batch_size = x_t.shape[0]
    
    position_ids = torch.arange(x_t.shape[1], device=x_t.device, dtype=torch.long).unsqueeze(0) - (1 - attention_mask).sum(dim=-1)
    kv_cache = DynamicCache()

    # cache prompt first
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        model(
            x_t,
            kv_cache=kv_cache,
            update_kv_cache=True,
        )
    cur_blocks = 0
    responses = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)
    # return lists
    # first_kvcache = deepcopy(kv_cache)
    sample_orders = []
    trajectory_inputs = []
    trajectory_outputs = []
    ranking_logprob_list = []
    unmask_index_list = []
    token_logprob_list = []
    is_valid_steps = []

    remask_times = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)

    # for visualization
    if viz:
        cur_response = x_t
        visualize_xt = [x_t]
        visualize_x0 = [x_t]
        cur_upm_prob = torch.zeros_like(x_t)
        visualize_upm_prob = [torch.zeros_like(x_t)]
    else:
        visualize_xt = None
        visualize_x0 = None
        visualize_upm_prob = None

    total_blocks = 0
    sample_blocks = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)
    while (cur_blocks * block_size) < max_length:
        x_t = torch.full((batch_size, block_size), fill_value=mask_token_id, device=device, dtype=torch.long)
        
        #计算此时的position_id
        position_ids = torch.arange(
            cur_blocks * block_size, 
            (cur_blocks + 1) * block_size, 
            device=x_t.device, dtype=torch.long).unsqueeze(0) + prompt_len.unsqueeze(1)

        #定义每一步的token保留数目
        num_transfer_tokens = torch.tensor([block_size // steps for _ in range(steps)])
        if block_size % steps != 0:
            num_transfer_tokens[-block_size % steps:] += 1
        # cumsum 
        num_transfer_tokens = num_transfer_tokens.cumsum(dim=0)

        #进行steps去噪
        token_per_step = block_size // steps
        for i in range(steps):
            is_valid_steps.append((~is_eos_meet).clone())

            mask_index = (x_t == mask_token_id)
            trajectory_inputs.append(x_t.clone())
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )
            logits = out.logits.to(torch.float32)
            prob = F.softmax(logits / temperature, dim=-1)
            
            log_probs = F.log_softmax(logits / temperature, dim=-1)

            x0 = torch.argmax(logits, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x_t)
            top2_p = prob.topk(k=2, dim=-1).values
            x0_p = (top2_p[..., 0] - top2_p[..., 1]).abs()
            x0_p = torch.where(mask_index, x0_p, -torch.inf)

            samples = torch.topk(x0_p, k=token_per_step, dim=-1).indices
            bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
            remask_index = torch.ones_like(x_t).bool()
            remask_index[bs_idx, samples] = False
            remask_index[~mask_index] = False
            sample_orders.append(samples)

            unmask_index = (~remask_index & mask_index) # & (~is_eos_meet.unsqueeze(1))
            x_t = torch.where(remask_index, mask_token_id, x0)
            unmask_index_list.append(unmask_index)
                  
            remask_times += (remask_index & (~mask_index) & (~is_eos_meet.unsqueeze(1))).float().sum(dim=-1)

        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        # setting x_t to eos_id if is_eos_meet
        x_t = torch.where(is_eos_meet.unsqueeze(1), eos_id, x_t)
        # gather all is_eos_meet
        tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(dist.get_world_size())]
        torch.distributed.barrier()
        torch.distributed.all_gather(tensor_list, is_eos_meet)
        is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        total_blocks += (~is_eos_meet_all_rank).float().sum()
        sample_blocks += (~is_eos_meet).float()
        if is_eos_meet_all_rank.all(): break

        # update kv_cache if not ends
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                position_ids=position_ids,
                kv_cache=kv_cache,
                update_kv_cache=True,
            )

    response_tokens = torch.cat(responses, dim=1)
    responses = []
    responses_length = []
    for i in range(batch_size):
        if eos_id in response_tokens[i]:
            eos_token_idx = (response_tokens[i] == eos_id).nonzero(as_tuple=True)[0][0].item()
            resp_token = response_tokens[i, prompt_len[i]:eos_token_idx]
        else:
            resp_token = response_tokens[i, prompt_len[i]:]
        responses.append(tokenizer.decode(resp_token, skip_special_tokens=True))
        responses_length.append(resp_token.shape[0])
    responses_length = torch.tensor(responses_length, device=device)
    if use_general_reward:
        rewards = universal_reward_func(batch, responses)
        rewards = rewards.float().to(device)
    else:
        rewards = reward_fn(batch, responses, num_generations, device).float() if reward_fn is not None else torch.zeros(batch_size, device=device)

    return {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'sample_orders': sample_orders,
        'unmask_index_list': unmask_index_list,
        'ranking_logprob_list': ranking_logprob_list,
        'is_valid_steps': is_valid_steps,
        'responses': responses,
        'response_tokens': response_tokens,
        'rewards': rewards,
        'prompt_len': prompt_len,
        'token_logprob_list': token_logprob_list,

        'block_size': block_size,
        'max_length': max_length,
        'steps': steps,
        'remask_times': remask_times,

        'visualize_x0': visualize_x0,
        'visualize_xt': visualize_xt,
        'visualize_upm_prob': visualize_upm_prob,
        'total_blocks': total_blocks,

        'responses_length': responses_length,
        'sample_blocks': sample_blocks
    }