import torch 
import numpy as np 
import torch.nn.functional as F

def log_linear_noise(
    t, eps: float = 1e-3
):
    if isinstance(t, torch.Tensor):
        total_noise = -torch.log1p(-(1 - eps) * t)
    else:
        total_noise = -np.log1p(-(1 - eps) * t) 
    rate_noise = (1 - eps) / (1 - (1 - eps) * t)
    
    return total_noise, rate_noise


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    elif method == 'max':
        return categorical_probs.argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")

def absorb_transition_rate(
    xt, score, num_classes, mask_token_id
): 
    Q_t = -F.one_hot(xt, num_classes=num_classes)
    Q_t[xt == mask_token_id] += 1
    
    rate = Q_t * score
    rate.scatter_(-1, xt[..., None], torch.zeros_like(rate))
    rate.scatter_(-1, xt[..., None], -rate.sum(dim=-1, keepdim=True))
    
    return rate
    


@torch.no_grad()
def sample_and_score_sudoku_trajectory(
    model,
    ref_model,
    batch,
    tokenizer,
    reward_fn,
    device,
    num_generations: int = 1,
    noise_fn: callable = log_linear_noise,
    tokens: int = 50258,
    steps = 256,
    
):
    eps = 1e-3
    t_steps = torch.linspace(eps, 1, steps, device=device)
    t_steps = torch.concat([torch.zeros_like(t_steps[:1]), t_steps])
    t_steps = t_steps.flip(dims=[0])

    x_next = tokenizer(batch['prompts'], return_tensors='pt', padding=True)['input_ids']
    x_next = x_next.to(device)
    # repeat x_next num_generations times
    x_next = x_next.repeat(num_generations, 1)
    # answer tokens 
    answers = batch['answers'] * num_generations
    answer_tokens = tokenizer(answers, return_tensors='pt', padding=True)['input_ids']
    answer_tokens = answer_tokens.to(device)
    
    batch_size = x_next.size(0)
    empty_pos_mask = x_next == tokenizer.mask_token_id
    
    trajectory_inputs = []
    trajectory_outputs = []
    ref_log_probs = []
    ref_log_prob = torch.zeros_like(x_next).float()
    completion_mask = x_next == tokenizer.mask_token_id
    
    for cur_step, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next.detach()
        t_cur = t_cur.view(1,).repeat(batch_size, )
        sigma_cur, dsigma_cur = noise_fn(t_cur)
        trajectory_inputs.append(x_cur.clone())
        score = model(
            x_cur.to(dtype=torch.long), 
            sigma_cur.view(-1), 
        ).exp()

        ref_score = ref_model(
            x_cur.to(dtype=torch.long), 
            sigma_cur.view(-1), 
        ).exp()
        
        rate = absorb_transition_rate(
            x_cur, 
            score, 
            tokens, 
            tokenizer.mask_token_id
        )
        
        ref_rate = absorb_transition_rate(
            x_cur, 
            ref_score, 
            tokens, 
            tokenizer.mask_token_id
        )
        
        t_next = t_next.view(1, 1, 1).repeat(batch_size, 1, 1)
        
        rev_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * rate
        rev_ref_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * ref_rate
        pt = F.one_hot(x_cur, num_classes=tokens).to(rev_rate) + rev_rate
        ref_pt = F.one_hot(x_cur, num_classes=tokens).to(rev_ref_rate) + rev_ref_rate
        
        x_next = sample_categorical(pt)
        trajectory_outputs.append(x_next.clone())
        # extract updated tokens (where x_cur != x_next) probabilities
        # pt_x = torch.clamp(pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1), min=1e-3, max=1.0)
        ref_pt_x = torch.clamp(ref_pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1), min=1e-3, max=1.0)
        ref_log_prob.zero_()
        ref_log_prob[x_cur != x_next] = ref_pt_x[x_cur != x_next].log()
        ref_log_probs.append(ref_log_prob.clone())
        
    # compute the reward 
    reward = reward_fn(answer_tokens, x_next, empty_pos_mask, device) 
    # average reward over num_generations 
    reward_mean = reward.view(num_generations, -1).mean(dim=0).repeat(num_generations,)
    reward_std = reward.view(num_generations, -1).std(dim=0).repeat(num_generations,)
    advantages = (reward - reward_mean) / (reward_std + 1e-4)
    
    output_dict = {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        't_steps': t_steps,
        'noise_fn': noise_fn,
        'tokens': tokens,
        'mask_token_id': tokenizer.mask_token_id,
        'samples': x_next,
        'ref_logp': ref_log_probs,
        'completion_mask': completion_mask, 
        'advantages': advantages, 
        'rewards': reward, 
        'reward_mean': reward_mean, 
        'reward_std': reward_std
    }
    return output_dict


# -----------------------------------------------------------------------
# DEPRECATED 


def deprecated_sample_and_score_sudoku_trajectory_step_wise(
    model,
    ref_model,
    tokenizer,
    batch,
    device,
    num_generations: int = 1,
    noise_fn: callable = log_linear_noise,
    tokens: int = 50258,
    steps = 256,
    
):
    eps = 1e-3
    t_steps = torch.linspace(eps, 1, steps, device=device)
    t_steps = torch.concat([torch.zeros_like(t_steps[:1]), t_steps])
    t_steps = t_steps.flip(dims=[0])

    x_next = tokenizer.batch_encode_plus(batch['prompts'], return_tensors='pt', padding=True)['input_ids']
    x_next = x_next.to(device)
    # repeat x_next num_generations times
    x_next = x_next.repeat(num_generations, 1)
    # answer tokens 
    answers = batch['answers'] * num_generations
    answer_tokens = tokenizer.batch_encode_plus(answers, return_tensors='pt', padding=True)['input_ids']
    answer_tokens = answer_tokens.to(device)
    empty_pos_mask = x_next == tokenizer.mask_token_id
    batch_size = x_next.size(0)
    
    # default 1.0 for stable 
    sample_log_prob = torch.zeros((batch_size, ), device=device).float()
    ref_sample_log_prob = torch.zeros_like(sample_log_prob)
    completion_mask = x_next == tokenizer.mask_token_id
    
    for cur_step, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next.detach()
        t_cur = t_cur.view(1,).repeat(batch_size, )
        sigma_cur, dsigma_cur = noise_fn(t_cur)

        score = model(
            x_cur.to(dtype=torch.long), 
            sigma_cur.view(-1), 
        ).exp()

        with torch.no_grad():
            ref_score = ref_model(
                x_cur.to(dtype=torch.long), 
                sigma_cur.view(-1), 
            ).exp()
        
        rate = absorb_transition_rate(
            x_cur, 
            score, 
            tokens, 
            tokenizer.mask_token_id
        )
        
        ref_rate = absorb_transition_rate(
            x_cur, 
            ref_score, 
            tokens, 
            tokenizer.mask_token_id
        )
        
        t_next = t_next.view(1, 1, 1).repeat(batch_size, 1, 1)
        
        rev_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * rate
        rev_ref_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * ref_rate
        pt = F.one_hot(x_cur, num_classes=tokens).to(rev_rate) + rev_rate
        ref_pt = F.one_hot(x_cur, num_classes=tokens).to(rev_ref_rate) + rev_ref_rate
        
        x_next = sample_categorical(pt)
        # extract updated tokens (where x_cur != x_next) probabilities
        pt_x = torch.clamp(pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1), min=1e-6, max=1.0)
        ref_pt_x = torch.clamp(ref_pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1), min=1e-6, max=1.0)
        
        sample_log_prob += (pt_x.log() * completion_mask).sum(dim=-1)
        ref_sample_log_prob += (ref_pt_x.log() * completion_mask).sum(dim=-1)
        
        
        
    # compute the reward 
    reward = reward_sudoku(answer_tokens, x_next, blank_num=batch['blank_num']) # + \
        # reward_sudoku_format(x_next, empty_pos_mask)
    reward = reward.float()
    # average reward over num_generations 
    reward_mean = reward.view(num_generations, -1).mean(dim=0).repeat(num_generations,)
    reward_std = reward.view(num_generations, -1).std(dim=0).repeat(num_generations,)
    advantages = (reward - reward_mean) / (reward_std + 1e-4)
    
    output_dict = {
        'samples': x_next,
        'logp': sample_log_prob,
        'ref_logp': ref_sample_log_prob,
        'completion_mask': completion_mask, 
        'advantages': advantages, 
        'reward': reward, 
        'reward_mean': reward_mean, 
        'reward_std': reward_std
    }
    return output_dict


def deprecated_sample_and_score_sudoku_trajectory_token_wise(
    model,
    ref_model,
    tokenizer,
    batch,
    device,
    num_generations: int = 1,
    noise_fn: callable = log_linear_noise,
    tokens: int = 50258,
    steps = 256,
):
    eps = 1e-3
    t_steps = torch.linspace(eps, 1, steps, device=device)
    
    t_steps = t_steps.flip(dims=[0])

    x_next = tokenizer.batch_encode_plus(batch['prompts'], return_tensors='pt', padding=True)['input_ids']
    x_next = x_next.to(device)
    # repeat x_next num_generations times
    x_next = x_next.repeat(num_generations, 1)
    # answer tokens 
    answers = batch['answers'] * num_generations
    answer_tokens = tokenizer.batch_encode_plus(answers, return_tensors='pt', padding=True)['input_ids']
    answer_tokens = answer_tokens.to(device)
    empty_pos_mask = x_next == tokenizer.mask_token_id
    
    # default 1.0 for stable 
    sample_prob = torch.ones_like(x_next).float()
    ref_sample_prob = torch.ones_like(x_next).float()
    completion_mask = x_next == tokenizer.mask_token_id
    
    batch_size = x_next.size(0)
    
    for _, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next.detach()
        t_cur = t_cur.view(1,).repeat(batch_size, )
        sigma_cur, dsigma_cur = noise_fn(t_cur)

        score = model(
            x_cur.to(dtype=torch.long), 
            sigma_cur.view(-1), 
        ).exp()

        with torch.no_grad():
            ref_score = ref_model(
                x_cur.to(dtype=torch.long), 
                sigma_cur.view(-1), 
            ).exp()
        
        rate = absorb_transition_rate(
            x_cur, 
            score, 
            tokens, 
            tokenizer.mask_token_id
        )
        
        ref_rate = absorb_transition_rate(
            x_cur, 
            ref_score, 
            tokens, 
            tokenizer.mask_token_id
        )
        
        t_next = t_next.view(1, 1, 1).repeat(batch_size, 1, 1)
        
        rev_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * rate
        rev_ref_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * ref_rate
        pt = F.one_hot(x_cur, num_classes=tokens).to(rev_rate) + rev_rate
        ref_pt = F.one_hot(x_cur, num_classes=tokens).to(rev_ref_rate) + rev_ref_rate
        
        x_next = sample_categorical(pt)
        # extract updated tokens (where x_cur != x_next) probabilities
        pt_x = pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1)
        ref_pt_x = ref_pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1)
        
        sample_prob[x_cur != x_next] = pt_x[x_cur != x_next]
        ref_sample_prob[x_cur != x_next] = ref_pt_x[x_cur != x_next]
        # if no <MASK> exists, break the loop 
        if (x_next == tokenizer.mask_token_id).sum() == 0:
            break
    
    # compute the reward 
    reward = reward_sudoku(answer_tokens, x_next, blank_num=batch['blank_num']) + \
        reward_sudoku_format(x_next, empty_pos_mask)
    
    # average reward over num_generations 
    reward_mean = reward.view(num_generations, -1).mean(dim=0).repeat(num_generations,)
    reward_std = reward.view(num_generations, -1).std(dim=0).repeat(num_generations,)
    advantages = (reward - reward_mean) / (reward_std + 1e-4)
    
    output_dict = {
        'samples': x_next,
        'prob': sample_prob,
        'ref_prob': ref_sample_prob,
        'completion_mask': completion_mask, 
        'advantages': advantages, 
        'reward': reward, 
        'reward_mean': reward_mean, 
        'reward_std': reward_std
    }
    return output_dict


def sample_trajectory(
    model,
    device,
    length: int = 256,
    noise_fn: callable = log_linear_noise,
    tokens: int = 50258,
    batch_size: int = 100,
    steps = 256,
):
    eps = 1e-3
    t_steps = torch.linspace(eps, 1, steps, device=device)
        
    t_steps = t_steps.flip(dims=[0])
    x_next = torch.ones(batch_size, length, device=device, dtype=torch.long) * (tokens - 1)
    sample_prob = torch.zeros_like(x_next).float()

    for _, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_cur = torch.tensor(t_cur).view(1,).repeat(batch_size, )
        sigma_cur, dsigma_cur = noise_fn(t_cur)
        score = model(
            x_cur.to(dtype=torch.long), 
            sigma_cur.view(-1), 
        ).exp()
    
        Q_t = -F.one_hot(x_cur, num_classes=tokens)
        Q_t[x_cur == tokens - 1] += 1
        
        rate = Q_t * score
        rate.scatter_(-1, x_cur[..., None], torch.zeros_like(rate))
        rate.scatter_(-1, x_cur[..., None], -rate.sum(dim=-1, keepdim=True))
        
        t_next = torch.tensor(t_next).view(1, 1, 1).repeat(batch_size, 1, 1)
        rev_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * rate
        pt = F.one_hot(x_cur, num_classes=tokens).to(rev_rate) + rev_rate
        x_next = sample_categorical(pt)
        # extract updated tokens (where x_cur != x_next) probabilities
        pt_x = pt.gather(dim=-1, index=x_next.unsqueeze(-1)).squeeze(-1)
        generated_tokens += (x_cur != x_next).sum()
        sample_prob[x_cur != x_next] = pt_x[x_cur != x_next]

    return x_next, sample_prob
    
    
# example sample function   
@torch.no_grad()
def euler_sample(
    net,     
    device,
    length: int = 256,
    noise_fn: callable = log_linear_noise,
    tokens: int = 50258,
    batch_size: int = 100,
    steps = 256,
    seed: int = 112,
):
    torch.manual_seed(seed)
    eps = 1e-3
    t_steps = torch.linspace(eps, 1, steps, device=device)
        
    t_steps = t_steps.flip(dims=[0])
    x_next = torch.ones(batch_size, length, device=device, dtype=torch.long) * (tokens - 1)

    for _, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_cur = torch.tensor(t_cur).view(1,).repeat(batch_size, )
        sigma_cur, dsigma_cur = noise_fn(t_cur)
        score = net(
            x_cur.to(dtype=torch.long), 
            sigma_cur.view(-1), 
        ).exp()
    
        Q_t = -F.one_hot(x_cur, num_classes=tokens)
        Q_t[x_cur == tokens - 1] += 1
        
        rate = Q_t * score
        rate.scatter_(-1, x_cur[..., None], torch.zeros_like(rate))
        rate.scatter_(-1, x_cur[..., None], -rate.sum(dim=-1, keepdim=True))
        
        t_next = torch.tensor(t_next).view(1, 1, 1).repeat(batch_size, 1, 1)
        rev_rate = (t_cur[..., None, None] - t_next) * dsigma_cur[..., None, None] * rate
        pt = F.one_hot(x_cur, num_classes=tokens).to(rev_rate) + rev_rate
        x_next = sample_categorical(pt)

    return x_next
