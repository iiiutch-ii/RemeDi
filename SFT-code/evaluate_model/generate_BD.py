import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from networks.llada.modeling_llada import LLaDAModelLM
from transformers.cache_utils import DynamicCache


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
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
def generate_block_wise(model,input_ids,inputs,max_length = 4096,block_size = 32,mask_token_id = 126336,eos_id = 126081,steps = 32,
                        remasking='low_confidence',cfg_scale=0., temperature=0.,sample: bool = False,block_in_block=8):
    ########计算prompts的范围并计算postion_id的开头部分
    # 找到每行第一个1的列索引（起始）
    bs,len_prompt = input_ids.shape
    start_indices = (inputs["attention_mask"] == 1).int().argmax(dim=1)
    # 找到每行最后一个1的列索引（结束）
    # 方法：反转每行，找第一个1的位置，再用总列数减去该位置
    reversed_matrix = torch.fliplr(inputs["attention_mask"] )
    end_indices =inputs["attention_mask"] .size(1) - 1 - (reversed_matrix == 1).int().argmax(dim=1) + 1
    # 合并结果
    prompts_ranges = torch.stack([start_indices, end_indices], dim=1)
    position_id_of_prompts = [list(range(end_indices.tolist()[i]))+[0 for j in range(len_prompt - end_indices.tolist()[i])] for i in range(bs)]
    start_id = end_indices.tolist()#这个是后续的position_id开始时的id
    ########计算prompt的注意力权重
    attention_mask_prompts = torch.zeros([bs,1,len_prompt,len_prompt])
    for i in range(bs):
        attention_mask_prompts[i,:,:end_indices[i],:end_indices[i]] = 1
        

    ########开始加入block
    device = input_ids.device
    x_t = input_ids
    len_sentence = input_ids.shape[1]
    
    while (len_sentence - len_prompt) <= max_length:
        x_t = torch.cat([x_t,torch.ones([bs,block_size],device=device)*mask_token_id],dim = -1).long()
        len_sentence = x_t.shape[1]
        #print(len_sentence)
        #计算此时的attention
        n_block_response = (len_sentence - len_prompt) // block_size
        attention_masks = []
        for i in range(bs):
            attention_mask = torch.ones([n_block_response+1,n_block_response+1],device = x_t.device)
            attention_mask = torch.tril(attention_mask)
            repeat_time = torch.tensor([len_prompt]+[block_size for k in range(n_block_response)],device=x_t.device)
            attention_mask = torch.repeat_interleave(attention_mask,repeat_time,0)
            attention_mask = torch.repeat_interleave(attention_mask,repeat_time,1)
            attention_mask[:len_prompt,:len_prompt] = attention_mask_prompts[i,0,:,:]
            if len_prompt - start_id[i] != 0: #start_id是回答的位置编码开始的地方，也是问题的长度
                attention_mask[len_prompt:,start_id[i]:len_prompt] = 0
            attention_masks.append(attention_mask)
        attention_masks = torch.stack(attention_masks,0).unsqueeze(1)
        attention_masks = attention_masks == 1
        #计算此时的position_id
        position_ids = []
        for i in range(bs):
            prompt_pos_id = list(range(len_prompt))
            if len_prompt != start_id[i]:
                prompt_pos_id[start_id[i]:len_prompt] = [0 for k in range(len_prompt-start_id[i])]
            position_id = prompt_pos_id + list(range(start_id[i],start_id[i]+block_size*n_block_response))
            position_ids.append(position_id)
        position_ids = torch.tensor(position_ids,device=x_t.device)

        #定义每一步的token保留数目
        num_transfer_tokens = get_num_transfer_tokens(block_size, steps, x_t.device)

        #进行steps去噪
        for i in range(steps):
            logits = model(x_t,attention_bias=attention_masks,position_ids=position_ids).logits
            
            if not sample:
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            else:
                x0 = sample_categorical(torch.softmax(logits.to(torch.float64), dim=-1))

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            mask_index = x_t == mask_token_id
            for j in range(x0_p.shape[0]):
                x0_p[j, len_prompt:len_prompt + (n_block_response - 1) * block_size] = -np.inf
                x0_p[j,len_prompt + (n_block_response - 1) * block_size : len_prompt + int((n_block_response - 1) * block_size+(i // block_in_block)*block_in_block)] = -np.inf
                x0_p[j,len_prompt+int((n_block_response - 1) * block_size+(1+(i // block_in_block))*block_in_block) : ] = -np.inf
            x0 = torch.where(mask_index, x0, x_t)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[i]) # num_transfer_tokens[:, i] all the same
                transfer_index[j, select_index] = True
            x_t[transfer_index] = x0[transfer_index]

        #设置停止条件，若一个batch内所有回答均停止则停止
        num_ended_sentence = 0
        for i in range(bs):
            if eos_id in x_t[i,len_prompt:]:
                num_ended_sentence += 1
        if num_ended_sentence == bs:
            break

    return x_t


@torch.no_grad()
def generate_swift_block(model,input_ids,inputs,max_length = 4096,block_size = 32,mask_token_id = 126336,eos_id = 126081,steps = 32,
                        remasking='low_confidence',cfg_scale=0., temperature=0.,sample: bool = False):
    
    ########计算prompts的范围并计算postion_id的开头部分
    # 找到每行第一个1的列索引（起始）
    bs,len_prompt = input_ids.shape
    start_indices = (inputs["attention_mask"] == 1).int().argmax(dim=1)
    # 找到每行最后一个1的列索引（结束）
    # 方法：反转每行，找第一个1的位置，再用总列数减去该位置
    reversed_matrix = torch.fliplr(inputs["attention_mask"] )
    end_indices =inputs["attention_mask"] .size(1) - 1 - (reversed_matrix == 1).int().argmax(dim=1) + 1
    # 合并结果
    prompts_ranges = torch.stack([start_indices, end_indices], dim=1)
    position_id_of_prompts = [list(range(end_indices.tolist()[i]))+[0 for j in range(len_prompt - end_indices.tolist()[i])] for i in range(bs)]
    start_id = end_indices.tolist()#这个是后续的position_id开始时的id
    ########计算prompt的注意力权重
    attention_mask_prompts = torch.zeros([bs,1,len_prompt,len_prompt])
    for i in range(bs):
        attention_mask_prompts[i,:,:end_indices[i],:end_indices[i]] = 1
        #attention_mask_prompts[i,:,:end_indices[i],:end_indices[i]] = torch.tril(attention_mask_prompts[i,:,:end_indices[i],:end_indices[i]])


    ########开始加入block
    device = input_ids.device
    x_t = input_ids
    len_sentence = input_ids.shape[1]
    
    with torch.no_grad():
        while len_sentence-len_prompt <= max_length:
            if torch.sum(x_t == mask_token_id) == 0:
                reveal_length = copy.deepcopy(len_prompt)
            else:
                reveal_length = torch.argmax((x_t == mask_token_id).float(), dim=-1)[0]
            new_mask_length = block_size - (x_t.shape[1] - reveal_length)


            x_t = torch.cat([x_t,torch.ones([bs,new_mask_length],device=device)*mask_token_id],dim = -1).long()
            len_sentence = x_t.shape[1]
            #print(len_sentence)
            #计算此时的attention
            len_freeze = len_sentence - block_size
            attention_masks = []
            for i in range(bs):
                attention_mask = torch.ones([2,2],device = x_t.device)
                attention_mask = torch.tril(attention_mask)
                repeat_time = torch.tensor([len_freeze]+[block_size],device=x_t.device)
                attention_mask = torch.repeat_interleave(attention_mask,repeat_time,0)
                attention_mask = torch.repeat_interleave(attention_mask,repeat_time,1)
                attention_mask[:len_prompt,:len_prompt] = attention_mask_prompts[i,0,:,:]
                if len_prompt - start_id[i] != 0: #start_id是回答的位置编码开始的地方，也是问题的长度,在此是为了防止padding的出现
                    attention_mask[len_prompt:,start_id[i]:len_prompt] = 0
                attention_masks.append(attention_mask)
            attention_masks = torch.stack(attention_masks,0).unsqueeze(1)
            attention_masks = attention_masks == 1
            #计算此时的position_id
            position_ids = []
            for i in range(bs):
                prompt_pos_id = list(range(len_prompt))
                if len_prompt != start_id[i]:
                    prompt_pos_id[start_id[i]:len_prompt] = [0 for k in range(len_prompt-start_id[i])]
                position_id = prompt_pos_id + list(range(start_id[i],len_sentence))
                position_ids.append(position_id)
            position_ids = torch.tensor(position_ids,device=x_t.device)

            num_transfer_tokens = get_num_transfer_tokens(block_size, steps, x_t.device)[0]
            #####开始生成
            logits = model(x_t,attention_bias=attention_masks,position_ids=position_ids).logits
            
            if not sample:
                logits_with_noise = logits#add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            else:
                x0 = sample_categorical(torch.softmax(logits.to(torch.float64), dim=-1))

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            mask_index = x_t == mask_token_id
            # for j in range(x0_p.shape[0]):
            #     x0_p[j, len_prompt+(n_block_response - 1) * block_size:] = -np.inf

            x0 = torch.where(mask_index, x0, x_t)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            if torch.sum(mask_index) == 2 and mask_index[:,-32]: #对confidence进行调整，保证-32的token不在最后一次输出
                confidence[:,-32] = 1.

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens) # num_transfer_tokens[:, i] all the same
                transfer_index[j, select_index] = True
            x_t[transfer_index] = x0[transfer_index]

            #设置停止条件，若一个batch内所有回答均停止则停止
            num_ended_sentence = 0
            for i in range(bs):
                if eos_id in x_t[i,len_prompt:]:
                    if not ((x_t[i,len_prompt:] == mask_token_id) & (torch.arange(len(x_t[i,len_prompt:]),device=device) < (x_t[i,len_prompt:] == eos_id).nonzero()[0].item() if (x_t[i,len_prompt:] == eos_id).any() else len(x_t[i,len_prompt:]))).any().item():
                        num_ended_sentence += 1
            if num_ended_sentence == bs:
                break
            

    return x_t



def main():
    device = 'cuda'

    #model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, )
    input_ids = inputs['input_ids'].to(device)

    out = generate_block_wise(model,input_ids,inputs,max_length = 1024,block_size = 32,mask_token_id = 126336,eos_id = 126081,steps = 32,
                              remasking='low_confidence',cfg_scale=0., temperature=0., sample = False)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


@torch.no_grad()
def generate_block_wise_kv_cache(model,input_ids,inputs,max_length = 4096,block_size = 32,mask_token_id = 126336,eos_id = 126081,steps = 32,
                        remasking='low_confidence',cfg_scale=0., temperature=0.,sample: bool = False):
    
    bs, len_prompt = input_ids.shape
    assert bs == 1, "batch_size must be 1"
    # 合并结果
        
    device = input_ids.device
    x_t = input_ids
    kv_cache = DynamicCache()
    batch_size = x_t.shape[0] 

    # cache prompt first
    out = model(
        x_t,
        timestep=torch.ones((x_t.size(0)), device=x_t.device) * 0,
        mask_index=(x_t == mask_token_id),
        kv_cache=kv_cache,
        update_kv_cache=True,
    )
    cur_blocks = 0
    timestep = torch.ones((x_t.size(0)), device=x_t.device) 
    response = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)

    while (cur_blocks * block_size) <= max_length:
        x_t = torch.full((batch_size, block_size), fill_value=mask_token_id, device=device, dtype=torch.long)
        #计算此时的position_id
        position_ids = torch.arange(
            cur_blocks * block_size, 
            (cur_blocks + 1) * block_size, 
            device=x_t.device, dtype=torch.long).unsqueeze(0) + len_prompt

        #定义每一步的token保留数目
        # block_size // steps  #
        num_transfer_tokens = get_num_transfer_tokens(block_size, steps, x_t.device)
        
        #进行steps去噪
        for i in range(steps): # tqdm(, total=steps, desc=f"Generating block {cur_blocks+1}"):
            mask_index = (x_t == mask_token_id)
            out = model(
                x_t,
                position_ids=position_ids,
                kv_cache=kv_cache,
                timestep=timestep * (1 - i / steps),
                mask_index=mask_index,
            )
            logits = out.logits.to(torch.float32)

            remask_tokens_nums = 0 # floor(uniform_scheduler(i / steps) * block_size)
            
            if not sample:
                x0 = torch.argmax(logits, dim=-1) # b, l
            else:
                x0 = sample_categorical(torch.softmax(logits.to(torch.float64), dim=-1))

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x_t)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=num_transfer_tokens[i]
                    # 需要加上 remask_tokens_nums 确保 mask token 逐一减少
                ) # num_transfer_tokens[:, i] all the same
                transfer_index[j, select_index] = True
            x_t[transfer_index] = x0[transfer_index]

            # Remask Step
            # if remask_tokens_nums > 0:
            #     upm_logits = out.upm_logits.to(torch.float32)
            #     upm_logits_in_block = upm_logits[:, -block_size:]
            #     upm_prob_in_block = F.softmax(upm_logits_in_block, dim=-1)
            #     remask_index = torch.zeros_like(x_t, dtype=torch.bool, device=x_t.device)
            #     remask_tokens_pos = torch.topk(upm_prob_in_block, k=remask_tokens_nums, dim=-1, largest=False).indices
            #     remask_index[:, -block_size:][remask_tokens_pos] = True
            #     x_t = torch.where(remask_index, 126336, x_t)
   
        # update kv_cache
        out = model(
            x_t,
            position_ids=position_ids,
            timestep=timestep * 0,
            mask_index=(x_t == mask_token_id),
            kv_cache=kv_cache,
            update_kv_cache=True,
        )
        response.append(x_t)
        cur_blocks += 1

        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        if is_eos_meet.all(): break

    response = torch.cat(response, dim=1)

    return response



if __name__ == '__main__':
    main()