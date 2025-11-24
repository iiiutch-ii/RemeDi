import os, json
import torch 
import click 
from transformers.cache_utils import DynamicCache
import dnnlib
import time 


@torch.no_grad()
def generate_block_diffusion(
    model, 
    conv,
    tokenizer, 
    reward_fn, 
    device,
    num_generations,
    kv_cache=None,
    steps: int = 32,
    max_length = 1024,
    block_size = 32, 
    mask_token_id = 126336,
    eos_id = 126081,
    sample: bool = True,
    repeat_times = None,
    viz: bool = False,
):
    """
    Generate block-wise responses using block diffusion. KV Cache is verified.
    """
    m = [conv]
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
    if kv_cache is None:
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
    sample_orders = []
    is_valid_steps = []

    sample_block_length = torch.zeros((batch_size,), device=device, dtype=torch.float32)
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
        for i in range(steps):
            is_valid_steps.append((~is_eos_meet).clone())

            mask_index = (x_t == mask_token_id)
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )
            logits = out.logits.to(torch.float32)
            x0 = torch.argmax(logits, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x_t)
            upm_prob = logits.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            samples = torch.topk(upm_prob, k=num_transfer_tokens[i], dim=-1).indices

            bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
            remask_index = torch.ones_like(x_t).bool()
            
            remask_index[bs_idx, samples] = False
            sample_orders.append(samples)
    
            x_t = torch.where(remask_index, mask_token_id, x0)

        if viz:
            print(tokenizer.decode(x_t[0], skip_special_tokens=True), end='')

        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        sample_block_length += (~is_eos_meet).float()
    
        # update kv_cache
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                position_ids=position_ids,
                kv_cache=kv_cache,
                update_kv_cache=True,
            )

        if is_eos_meet.all(): break
    
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
    
    return {
        'responses': responses,
        'responses_length': responses_length,
        'kv_cache': kv_cache,
    }



from train import CommandWithConfigFile
@click.command(cls=CommandWithConfigFile("config"))
@click.option("--config", type=str, default="RemeDi")
@click.option("--ckpt_path", type=str, default="")
@click.option('--temperature', type=float, default=None)
@click.option('--seed', type=int, default=113)
def main(
    config,
    ckpt_path, 
    temperature, 
    seed,
    **kwargs,
):
    torch.manual_seed(seed)
    device = 'cuda'
    tokenizer_kwargs = kwargs.pop("tokenizer_kwargs")
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    # setting pad_token_id to mask_token_id
    tokenizer.pad_token_id = 126336
    network_kwargs = kwargs.pop("network_kwargs")
    network_kwargs["torch_dtype"] = torch.bfloat16
    if ckpt_path:
        network_kwargs["pretrained_model_name_or_path"] = ckpt_path
    if "other_weights" in network_kwargs:
        del network_kwargs["other_weights"]
    print("Constructing from ", network_kwargs)

    model = dnnlib.util.construct_class_by_name(
        **network_kwargs,)
    model.eval().requires_grad_(False).to(device)

    conv = []
    while True:
        conv = []
        print('=' * 20)
        prompt = input("User: ").strip()
        print('Assistant: ', end='')

        conv = [{'role': 'user', 'content': prompt}]
        t1 = time.time()
        inputs = generate_block_diffusion(
            model,
            conv,
            tokenizer,
            reward_fn=None,
            device=device,
            viz=True,
            num_generations=1,
            steps=32, max_length=1024, block_size=32,
        )
        t2 = time.time()
        conv.append({'role': 'assistant', 'content': inputs['responses'][0]})

        print(f'\t[tokens: {inputs["responses_length"][0]}; time: {t2 - t1:.2f}s]')
        


if __name__ == "__main__":
    main()