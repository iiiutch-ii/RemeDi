import torch
import torch.nn.functional as F
from torch_utils import misc
from networks.block_llada.modelling_llada_bitowel import DynamicCache
from networks.block_llada.utils import get_pt_logprobs, sample_categorical
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
def generate_block_diffusion_bitowel(
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
    token_entropy_list = {} 
    upm_entropy_list = {}

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

        num_transfer_tokens = torch.tensor([block_size // steps for _ in range(steps)])
        if block_size % steps != 0:
            num_transfer_tokens[-block_size % steps:] += 1
        # cumsum 
        num_transfer_tokens = num_transfer_tokens.cumsum(dim=0)

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
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            # entropy = - (log_probs * log_probs.exp()).sum(dim=-1)
            # entropy = torch.where(mask_index, entropy, 0)
            # token_entropy_list.setdefault(i, []).append((entropy.sum() / mask_index.sum()).item())
            
            upm_logits = out.confidences.to(torch.float32)
            upm_prob = F.softmax(upm_logits / upm_temperature, dim=-1)
            if viz:
                visualize_upm_prob.append(torch.cat([cur_upm_prob, upm_logits], dim=1))
            # upm_entropy = - (upm_prob * upm_prob.log()).sum(dim=-1)
            # upm_entropy_list.setdefault(i, []).append(upm_entropy.mean().item())
            
            # sample tokens
            if sample:
                x0 = sample_categorical(log_probs.exp())
            else:
                x0 = torch.argmax(logits, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x_t)
            trajectory_outputs.append(x0.clone())
            token_logprob_list.append(log_probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1))

            # sample position
            if sample:
                samples = torch.multinomial(upm_prob, num_samples=num_transfer_tokens[i], replacement=False)
            else:
                samples = torch.topk(upm_prob, k=num_transfer_tokens[i], dim=-1).indices
            ranking_prob = get_pt_logprobs(upm_prob, samples)
            ranking_logprob_list.append(ranking_prob)

            bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
            remask_index = torch.ones_like(x_t).bool()
            remask_index[bs_idx, samples] = False
            sample_orders.append(samples)

            unmask_index = (~remask_index & mask_index) # & (~is_eos_meet.unsqueeze(1))
    
            x_t = torch.where(remask_index, mask_token_id, x0)
            unmask_index_list.append(unmask_index)
            
            if viz:
                visualize_xt.append(torch.cat([cur_response, x_t], dim=1))
                visualize_x0.append(torch.cat([cur_response, x0], dim=1))
                
            
            remask_times += (remask_index & (~mask_index) & (~is_eos_meet.unsqueeze(1))).float().sum(dim=-1)

        if viz:
            cur_response = torch.cat([cur_response, x_t], dim=1)
            cur_upm_prob = torch.cat([cur_upm_prob, torch.zeros_like(upm_prob)], dim=1)
        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        # is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        # # setting x_t to eos_id if is_eos_meet
        # x_t = torch.where(is_eos_meet.unsqueeze(1), eos_id, x_t)
        # # gather all is_eos_meet
        # tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(dist.get_world_size())]
        # torch.distributed.barrier()
        # torch.distributed.all_gather(tensor_list, is_eos_meet)
        # is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        # total_blocks += (~is_eos_meet_all_rank).float().sum()
        # sample_blocks += (~is_eos_meet).float()
        # if is_eos_meet_all_rank.all(): break

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

        # 'token_entropy_list': token_entropy_list,
        # 'upm_entropy_list': upm_entropy_list,
        'responses_length': responses_length,
        'sample_blocks': sample_blocks
    }


def logprob_loss_block_bitowel(
    model, 
    ref_model, 
    inputs, 
    valid_samples, 
    eps = .2, beta= 0.0, gain=1.0, 
    num_generations=None, temperature=1., accelerator=None,
    mask_id=126336, 
    repeat_times=1, loss_scale: bool = False, 
    skip_step: int = 64,
    upm_temperature: float = 1.0, length_norm: bool = False,
    sync_cond: bool = False,
    *args, **kwargs
):
    advantages = inputs['advantages']
    prompt_len = inputs['prompt_len']
    trajectory_inputs = inputs['trajectory_inputs']
    trajectory_outputs = inputs['trajectory_outputs']
    sample_orders = inputs['sample_orders']
    unmask_indexs = inputs['unmask_index_list']
    token_logprob_list = inputs['token_logprob_list']
    ranking_logprob_list = inputs['ranking_logprob_list']
    response_tokens = inputs['response_tokens']
    unmask_index_list = inputs['unmask_index_list']
    is_valid_steps = inputs['is_valid_steps']
    total_blocks = inputs['total_blocks']
    sample_blocks = inputs['sample_blocks'] # (B,)

    block_size = inputs['block_size']
    steps = inputs['steps']
    batch_size = advantages.shape[0]
    device = advantages.device

    # B, L_{block}
    loss_to_log = 0.

    valid_samples = accelerator.gather(valid_samples)
    valid_samples = valid_samples.float().mean().item()

    if valid_samples < 1e-6:
        print("No valid samples, skip logprob loss")
        return None

    num_pred_tokens_per_sample = torch.stack([item.sum(-1) for item in unmask_indexs], 0).sum(0) # how many tokens are predicted per sample
    num_pred_tokens_per_sample_valid = num_pred_tokens_per_sample * (advantages != 0).float()    # only count valid samples
    total_pred_tokens = num_pred_tokens_per_sample_valid.sum()
    total_pred_tokens = accelerator.gather(total_pred_tokens).mean().item()

    if loss_scale:
        scaler_logits = total_pred_tokens * repeat_times
        scaler_top_k = steps * (steps - 1) / 2 * total_blocks * repeat_times
    else:
        scaler_logits = 1.0 # total_pred_tokens * repeat_times
        scaler_top_k = 1.0 # 32 * (steps - 1) / 2 * valid_samples
    length_normalizer = 1.0
    if length_norm:
        length_normalizer = sample_blocks * block_size

    resp_seqlen = response_tokens.size(1) - prompt_len[0]
    attn_bias = construct_block_attention_bias(
        resp_seqlen.item(), 
        prompt_len[0].item(),
        block_size,
    )[None, None].repeat(batch_size, 1, 1, 1)

    position_ids_p1 = torch.arange(response_tokens.size(1), device=device, dtype=torch.long).unsqueeze(0)
    position_ids_p2 = torch.arange(prompt_len[0], response_tokens.size(1), device=device, dtype=torch.long).unsqueeze(0)
    position_ids = torch.concat([position_ids_p1, position_ids_p2], dim=1)
    position_ids = position_ids.repeat(batch_size, 1)
    for i in range(steps):
        if i > skip_step: break
        # 1. model forward 
        # 1.1 construct input_ids (prompt, x0, xt)
        # (bs, block_num * block_size)
        x_t = torch.concat(trajectory_inputs[i::steps], dim=1) 
        x0 = torch.concat(trajectory_outputs[i::steps], dim=1) 
        # (bs, block_num, keep_token_num) 
        sample_order = torch.stack(sample_orders[i::steps], dim=1)  
        # (bs, block_num, keep_token_num) 
        old_ranking_logprob = torch.stack(ranking_logprob_list[i::steps], dim=1)
        # (bs, block_num * block_size)
        old_token_logprob = torch.concat(token_logprob_list[i::steps], dim=1)
        # (bs, block_num * block_size)
        unmask_index = torch.concat(unmask_index_list[i::steps], dim=1)
        # (bs, block_num * block_size)
        is_valid_step = torch.stack(is_valid_steps[i::steps], dim=1) # (bs, block_num)

        input_ids = torch.concat(
            [response_tokens, x_t], dim=1
        )
        
        torch.distributed.barrier()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = model(
                input_ids,
                attention_bias=attn_bias,
                position_ids=position_ids,
            )
        # select xt
        logits = out.logits.to(torch.float32)[:, -resp_seqlen:]
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        # (B, L)
        token_logprob = log_probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        upm_logits = out.confidences.to(torch.float32)[:, -resp_seqlen:]
        upm_logits = upm_logits.view(batch_size, -1, block_size)
        upm_prob = F.softmax(upm_logits / upm_temperature, dim=-1)
        # (bs, block_num, keep_token_num) 
        ranking_logprob = get_pt_logprobs(
            upm_prob.view(-1, block_size), 
            sample_order.view(batch_size * sample_order.size(1), -1)
        ).view(batch_size, sample_order.size(1), -1) 
        
        # compute loss
        logits_logprob_ratio = (token_logprob - old_token_logprob).exp()
        logits_logprob_ratio = torch.where(unmask_index, logits_logprob_ratio, 1)
        # TODO: sum dim=-1?
        pt_logprob_ratio = (ranking_logprob.sum(dim=-1) - old_ranking_logprob.sum(dim=-1)).exp()

        # mask out sequence that is already ended
        # (8,96)
        logits_logprob_ratio = torch.where(is_valid_step.unsqueeze(dim=-1).repeat(1, 1, block_size).view(batch_size, -1), logits_logprob_ratio, 1)
        pt_logprob_ratio = torch.where(is_valid_step, pt_logprob_ratio, 1)

        # perform clip on logits_logprob_ratio and logits_top_k_ratio
        logits_logprob_ratio_cliped = logits_logprob_ratio.clamp(1 - eps, 1 + eps)
        pt_logprob_ratio_cliped = pt_logprob_ratio.clamp(1 - eps, 1 + eps)

        loss_logits = - torch.min(
            logits_logprob_ratio * advantages.view(-1, 1),
            logits_logprob_ratio_cliped * advantages.view(-1, 1),
        ).sum(dim=-1)
        
        loss_top_k  = - torch.min(
            pt_logprob_ratio * advantages.view(-1, 1),
            pt_logprob_ratio_cliped * advantages.view(-1, 1),
        ).sum(dim=-1)
        
        loss = (loss_logits / length_normalizer).sum() / scaler_logits + (loss_top_k / length_normalizer).sum() / scaler_top_k
        # normalize loss by denoising steps & token length
        torch.distributed.barrier()
        with misc.ddp_sync(model, sync=sync_cond and i == steps - 1):
            if accelerator is not None:
                accelerator.backward(loss.mul(gain).sum())
            else:
                loss.mul(gain).sum().backward()
            loss_to_log = loss_to_log + loss.mul(gain ).detach().mean().item()

    torch.distributed.barrier()
    all_rewards = accelerator.gather(inputs['rewards'].detach())
    reward_mean = all_rewards.mean().item()
    reward_std = all_rewards.std().item()

    eos_id = 126081
    length = ((response_tokens != eos_id).sum(-1).to(device).float() - prompt_len)

    return {
        "reward": reward_mean,
        "length": length.mean().item(),
        "length_std": length.std().item(),
        'length_max': length.max().item(),
        "valid_samples": valid_samples,
        "num_pred_tokens_per_sample": total_pred_tokens / valid_samples * repeat_times,
        "reward_std": reward_std,
    }
