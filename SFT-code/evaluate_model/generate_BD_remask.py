import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import NamedTuple, Optional, List, Tuple
from transformers.cache_utils import DynamicCache

import torch_utils.distributed as dist
from networks.llada.modeling_llada import LLaDAModelLM
from networks.block_llada.utils import get_config, get_pt_logprobs, sample_categorical


def get_active_samples_with_fsdp_constraint(is_eos_meet, device):
    """
    获取考虑FSDP约束的活跃样本索引
    确保每个rank至少保留一个样本参与计算
    
    Args:
        is_eos_meet: [batch_size] bool tensor, 标记已完成的样本
        device: 设备
        
    Returns:
        active_mask: [batch_size] bool tensor, 标记需要参与计算的样本
        is_placeholder: [batch_size] bool tensor, 标记占位符样本
    """
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    except:
        # 非分布式环境，直接返回所有未完成的样本
        world_size = 1
        rank = 0
    
    batch_size = is_eos_meet.shape[0]
    
    if world_size == 1:
        # 非分布式环境，直接返回未完成的样本
        active_mask = ~is_eos_meet
        is_placeholder = torch.zeros_like(is_eos_meet, dtype=torch.bool)
        return active_mask, is_placeholder
    
    # 计算每个rank的样本范围
    samples_per_rank = batch_size // world_size
    rank_start = rank * samples_per_rank
    rank_end = (rank + 1) * samples_per_rank if rank < world_size - 1 else batch_size
    
    # 获取当前rank的样本状态
    rank_is_eos = is_eos_meet[rank_start:rank_end]
    rank_active = ~rank_is_eos  # 未完成的样本
    
    # 初始化结果
    active_mask = torch.zeros_like(is_eos_meet, dtype=torch.bool)
    is_placeholder = torch.zeros_like(is_eos_meet, dtype=torch.bool)
    
    # 如果当前rank有活跃样本，直接使用
    if rank_active.any():
        active_mask[rank_start:rank_end] = rank_active
    else:
        # 如果没有活跃样本，保留第一个作为占位符
        if rank_end > rank_start:
            active_mask[rank_start] = True
            is_placeholder[rank_start] = True
    
    return active_mask, is_placeholder


def create_mock_outputs_for_placeholders(batch_size, seq_len, vocab_size, device, eos_id=126081, mask_token_id=126336):
    """
    为占位符样本创建模拟的网络输出
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        vocab_size: 词汇表大小
        device: 设备
        eos_id: EOS token ID
        mask_token_id: MASK token ID
        
    Returns:
        mock_logits: [batch_size, seq_len, vocab_size] 模拟的token logits
        mock_upm_logits: [batch_size, seq_len] 模拟的UPM logits
    """
    # 创建EOS概率极高的logits
    mock_logits = torch.full((batch_size, seq_len, vocab_size), -10.0, device=device, dtype=torch.float32)
    mock_logits[:, :, eos_id] = 10.0  # EOS token概率极高
    
    # 创建均匀分布的UPM logits
    mock_upm_logits = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
    
    return mock_logits, mock_upm_logits


def filter_active_samples(tensor, active_mask):
    """
    根据活跃样本掩码过滤张量
    
    Args:
        tensor: 输入张量，第一维是batch维度
        active_mask: [batch_size] bool tensor
        
    Returns:
        filtered_tensor: 过滤后的张量
    """
    if tensor is None:
        return None
    return tensor[active_mask]


def restore_full_batch_outputs(active_outputs, active_mask, placeholder_outputs, full_batch_size):
    """
    将活跃样本的输出恢复到完整batch大小
    
    Args:
        active_outputs: 活跃样本的网络输出
        active_mask: [batch_size] bool tensor, 活跃样本掩码
        placeholder_outputs: 占位符样本的模拟输出
        full_batch_size: 完整批次大小
        
    Returns:
        full_outputs: 恢复到完整batch的输出
    """
    # 获取输出的形状信息
    if hasattr(active_outputs, 'logits'):
        # 如果是模型输出对象
        device = active_outputs.logits.device
        seq_len = active_outputs.logits.shape[1]
        vocab_size = active_outputs.logits.shape[2]
        
        # 创建完整batch的输出容器
        full_logits = torch.zeros((full_batch_size, seq_len, vocab_size), device=device, dtype=active_outputs.logits.dtype)
        full_upm_logits = torch.zeros((full_batch_size, seq_len), device=device, dtype=active_outputs.upm_logits.dtype)
        
        # 填充活跃样本的输出
        full_logits[active_mask] = active_outputs.logits
        full_upm_logits[active_mask] = active_outputs.upm_logits
        
        # 填充占位符样本的输出
        inactive_mask = ~active_mask
        if inactive_mask.any():
            full_logits[inactive_mask] = placeholder_outputs[0][inactive_mask]
            full_upm_logits[inactive_mask] = placeholder_outputs[1][inactive_mask]
        
        # 创建输出对象
        class MockOutput:
            def __init__(self, logits, upm_logits):
                self.logits = logits
                self.upm_logits = upm_logits
        
        return MockOutput(full_logits, full_upm_logits)
    else:
        # 如果是普通张量
        device = active_outputs.device
        full_shape = (full_batch_size,) + active_outputs.shape[1:]
        full_outputs = torch.zeros(full_shape, device=device, dtype=active_outputs.dtype)
        full_outputs[active_mask] = active_outputs
        return full_outputs


def extract_active_kvcache(kv_cache, active_mask):
    """
    从KV Cache中提取活跃样本的缓存
    
    Args:
        kv_cache: DynamicCache对象
        active_mask: [batch_size] bool tensor
        
    Returns:
        active_kv_cache: 活跃样本的KV Cache
    """
    if kv_cache is None:
        return None
    
    active_kv_cache = DynamicCache()
    
    # 复制活跃样本的key和value
    for layer_idx in range(len(kv_cache.key_cache)):
        if kv_cache.key_cache[layer_idx] is not None:
            active_key = kv_cache.key_cache[layer_idx][active_mask]
            active_value = kv_cache.value_cache[layer_idx][active_mask]
            active_kv_cache.update(active_key, active_value, layer_idx)
    
    return active_kv_cache


def update_kvcache_from_active(original_kv_cache, active_kv_cache, active_mask):
    """
    将活跃样本的KV Cache更新回原始缓存
    
    Args:
        original_kv_cache: 原始的KV Cache
        active_kv_cache: 活跃样本的KV Cache  
        active_mask: [batch_size] bool tensor
    """
    if original_kv_cache is None or active_kv_cache is None:
        return
    
    # 更新每一层的key和value
    for layer_idx in range(len(active_kv_cache.key_cache)):
        if active_kv_cache.key_cache[layer_idx] is not None:
            if layer_idx >= len(original_kv_cache.key_cache):
                # 如果原始缓存没有这一层，创建新的
                original_kv_cache.key_cache.append(None)
                original_kv_cache.value_cache.append(None)
            
            if original_kv_cache.key_cache[layer_idx] is None:
                # 如果这是第一次缓存这一层
                batch_size = active_mask.shape[0]
                device = active_kv_cache.key_cache[layer_idx].device
                dtype = active_kv_cache.key_cache[layer_idx].dtype
                key_shape = (batch_size,) + active_kv_cache.key_cache[layer_idx].shape[1:]
                value_shape = (batch_size,) + active_kv_cache.value_cache[layer_idx].shape[1:]
                
                original_kv_cache.key_cache[layer_idx] = torch.zeros(key_shape, device=device, dtype=dtype)
                original_kv_cache.value_cache[layer_idx] = torch.zeros(value_shape, device=device, dtype=dtype)
            
            # 更新活跃样本的缓存
            original_kv_cache.key_cache[layer_idx][active_mask] = active_kv_cache.key_cache[layer_idx]
            original_kv_cache.value_cache[layer_idx][active_mask] = active_kv_cache.value_cache[layer_idx]


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
                kv_cache=kwargs["kv_cache"],
                update_kv_cache=kwargs["update_kv_cache"]
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
        kv_cache = kwargs.get("kv_cache", None)
        update_kv_cache = kwargs.get("update_kv_cache", False)

        upm_logits = self.get_mask_prob(
            out.hidden_states[self.config.use_feat_layer], 
            timestep=timestep,
            mask_index=mask_index,
            attention_bias=attention_bias,
            position_ids=position_id,
            kv_cache=kv_cache,
            update_kv_cache=update_kv_cache
        )

        return LLaDAUPMOutput(
            logits=out.logits,
            attn_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            upm_logits=upm_logits
        )




@torch.no_grad()
def generate_block_diffusion(
    model, 
    batch,
    tokenizer, 
    reward_fn, 
    device,
    num_generations,
    temperature: float = 1.0,
    block_length: int = 32,
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
    # process batch 
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
            timestep=torch.ones((batch_size, x_t.size(1)), device=x_t.device)  * 0,
            mask_index=(x_t == mask_token_id),
            kv_cache=kv_cache,
            update_kv_cache=True,
        )
    cur_blocks = 0
    responses = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)
    timestep = torch.ones((batch_size, block_size), device=x_t.device) 
    # return lists
    first_kvcache = deepcopy(kv_cache)
    sample_orders = []
    trajectory_inputs = []
    trajectory_outputs = []
    ranking_logprob_list = []
    unmask_index_list = []
    token_logprob_list = []
    is_valid_steps = []
    token_entropy_list = {} # 不同 step 剩下所有的 mask token 预测的 entropy 差异
    upm_entropy_list = {}

    remask_times = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)

    # for visualization
    if viz:
        cur_response = x_t
        visualize_xt = [x_t]
        visualize_x0 = [x_t]
    else:
        visualize_xt = None
        visualize_x0 = None

    total_blocks = 0
    sample_block_length = torch.zeros((batch_size,), device=device, dtype=torch.float32)
    while (cur_blocks * block_size) <= max_length:
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
            trajectory_inputs.append(x_t.clone())
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    timestep=timestep * (1 - i / steps),
                    mask_index=mask_index,
                )
            logits = out.logits.to(torch.float32)
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            entropy = - (log_probs * log_probs.exp()).sum(dim=-1)
            entropy = torch.where(mask_index, entropy, 0)
            token_entropy_list.setdefault(i, []).append((entropy.sum() / mask_index.sum()).item())
            
            upm_logits = out.upm_logits.to(torch.float32)
            upm_prob = F.softmax(upm_logits, dim=-1)
            upm_entropy = - (upm_prob * upm_prob.log()).sum(dim=-1)
            upm_entropy_list.setdefault(i, []).append(upm_entropy.mean().item())
            
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
            
            remask_times += (remask_index & ~ mask_index & ~is_eos_meet.unsqueeze(1)).float().sum(dim=-1)

        if viz:
            cur_response = torch.cat([cur_response, x_t], dim=1)
        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        sample_block_length += (~is_eos_meet).float()
        # gather all is_eos_meet
        tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(tensor_list, is_eos_meet)
        is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        total_blocks += (~is_eos_meet_all_rank).float().sum()
        if is_eos_meet_all_rank.all(): break

        # update kv_cache if not ends
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                position_ids=position_ids,
                timestep=timestep * 0,
                mask_index=(x_t == mask_token_id),
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
        'first_kvcache': first_kvcache,

        'block_size': block_size,
        'max_length': max_length,
        'steps': steps,
        'remask_times': remask_times,

        'visualize_x0': visualize_x0,
        'visualize_xt': visualize_xt,

        'total_blocks': total_blocks,

        'token_entropy_list': token_entropy_list,
        'upm_entropy_list': upm_entropy_list,
        'responses_length': responses_length,
        'sample_block_length': sample_block_length,
    }





@torch.no_grad()
def generate_block_diffusion_fast(
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
    enable_dynamic_batching: bool = True,  # 添加动态批处理开关
):
    """
    Generate block-wise responses using block diffusion. KV Cache is verified.
    """
    # process batch 
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
            timestep=torch.ones((batch_size, x_t.size(1)), device=x_t.device)  * 0,
            mask_index=(x_t == mask_token_id),
            kv_cache=kv_cache,
            update_kv_cache=True,
        )
    
    cur_blocks = 0
    responses = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)
    timestep = torch.ones((batch_size, block_size), device=x_t.device) 
    # return lists
    first_kvcache = deepcopy(kv_cache)
    sample_orders = []
    trajectory_inputs = []
    trajectory_outputs = []
    ranking_logprob_list = []
    unmask_index_list = []
    token_logprob_list = []
    is_valid_steps = []
    token_entropy_list = {} # 不同 step 剩下所有的 mask token 预测的 entropy 差异
    upm_entropy_list = {}

    remask_times = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)

    # for visualization
    if viz:
        cur_response = x_t
        visualize_xt = [x_t]
        visualize_x0 = [x_t]
    else:
        visualize_xt = None
        visualize_x0 = None

    total_blocks = 0
    while (cur_blocks * block_size) <= max_length:
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
            trajectory_inputs.append(x_t.clone())
            
            # 动态批处理：获取活跃样本和占位符标记
            if enable_dynamic_batching:
                active_mask, is_placeholder = get_active_samples_with_fsdp_constraint(is_eos_meet, device)
                num_active = active_mask.sum().item()
                # 注意：is_placeholder 用于标记占位符样本，在当前实现中暂不直接使用
                
                if num_active < batch_size:
                    # 过滤活跃样本的输入
                    active_x_t = filter_active_samples(x_t, active_mask)
                    active_position_ids = filter_active_samples(position_ids, active_mask)
                    active_timestep = filter_active_samples(timestep * (1 - i / steps), active_mask)
                    active_mask_index = filter_active_samples(mask_index, active_mask)
                    active_kv_cache = extract_active_kvcache(kv_cache, active_mask)
                    
                    # 网络前向传播（只对活跃样本）
                    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                        active_out = model(
                            active_x_t,
                            position_ids=active_position_ids,
                            kv_cache=active_kv_cache,
                            timestep=active_timestep,
                            mask_index=active_mask_index,
                        )
                    
                    # 为占位符样本创建模拟输出
                    mock_logits, mock_upm_logits = create_mock_outputs_for_placeholders(
                        batch_size, block_size, out.logits.size(-1), device, eos_id, mask_token_id
                    )
                    placeholder_outputs = (mock_logits, mock_upm_logits)
                    
                    # 恢复到完整batch输出
                    out = restore_full_batch_outputs(active_out, active_mask, placeholder_outputs, batch_size)
                    
                    # 更新KV Cache
                    update_kvcache_from_active(kv_cache, active_kv_cache, active_mask)
                else:
                    # 如果所有样本都是活跃的，正常处理
                    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                        out = model(
                            x_t,
                            position_ids=position_ids,
                            kv_cache=kv_cache,
                            timestep=timestep * (1 - i / steps),
                            mask_index=mask_index,
                        )
            else:
                # 不启用动态批处理，正常处理
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    out = model(
                        x_t,
                        position_ids=position_ids,
                        kv_cache=kv_cache,
                        timestep=timestep * (1 - i / steps),
                        mask_index=mask_index,
                    )
            
            logits = out.logits.to(torch.float32)
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            entropy = - (log_probs * log_probs.exp()).sum(dim=-1)
            entropy = torch.where(mask_index, entropy, 0)
            entropy = torch.where((~is_eos_meet).unsqueeze(1), entropy, 0)
            token_entropy_list.setdefault(i, []).append((entropy.sum() / mask_index.sum()).item())
            
            upm_logits = out.upm_logits.to(torch.float32)
            upm_prob = F.softmax(upm_logits, dim=-1)
            upm_entropy = - (upm_prob * upm_prob.log()).sum(dim=-1)
            upm_entropy = torch.where(~is_eos_meet, upm_entropy, 0)
            upm_entropy_list.setdefault(i, []).append(upm_entropy.mean().item())
            
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
            
            remask_times += (remask_index & ~ mask_index & ~is_eos_meet.unsqueeze(1)).float().sum(dim=-1)

        if viz:
            cur_response = torch.cat([cur_response, x_t], dim=1)
        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        # gather all is_eos_meet
        tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(tensor_list, is_eos_meet)
        is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        total_blocks += (~is_eos_meet_all_rank).float().sum()
        if is_eos_meet_all_rank.all(): break

        # update kv_cache if not ends
        if enable_dynamic_batching:
            active_mask, is_placeholder = get_active_samples_with_fsdp_constraint(is_eos_meet, device)
            num_active = active_mask.sum().item()
            # 注意：is_placeholder 用于标记占位符样本，在当前实现中暂不直接使用
            
            if num_active < batch_size:
                # 只对活跃样本更新KV Cache
                active_x_t = filter_active_samples(x_t, active_mask)
                active_position_ids = filter_active_samples(position_ids, active_mask)
                active_mask_index = filter_active_samples((x_t == mask_token_id), active_mask)
                active_kv_cache = extract_active_kvcache(kv_cache, active_mask)
                
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    model(
                        active_x_t,
                        position_ids=active_position_ids,
                        timestep=torch.zeros_like(active_x_t[:, :1]).expand(-1, active_x_t.size(1)),
                        mask_index=active_mask_index,
                        kv_cache=active_kv_cache,
                        update_kv_cache=True,
                    )
                
                # 更新回原始KV Cache
                update_kvcache_from_active(kv_cache, active_kv_cache, active_mask)
            else:
                # 如果所有样本都活跃，正常更新
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    model(
                        x_t,
                        position_ids=position_ids,
                        timestep=timestep * 0,
                        mask_index=(x_t == mask_token_id),
                        kv_cache=kv_cache,
                        update_kv_cache=True,
                    )
        else:
            # 不启用动态批处理，正常更新
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                model(
                    x_t,
                    position_ids=position_ids,
                    timestep=timestep * 0,
                    mask_index=(x_t == mask_token_id),
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
        'first_kvcache': first_kvcache,

        'block_size': block_size,
        'max_length': max_length,
        'steps': steps,
        'remask_times': remask_times,

        'visualize_x0': visualize_x0,
        'visualize_xt': visualize_xt,

        'total_blocks': total_blocks,

        'token_entropy_list': token_entropy_list,
        'upm_entropy_list': upm_entropy_list,
        'responses_length': responses_length,
    }



def logprob_loss_block(
    model, 
    ref_model, 
    inputs, 
    valid_samples, 
    eps = .2, beta= 0.0, gain=1.0, 
    num_generations=None, temperature=1., accelerator=None,
    mask_id=126336, 
    repeat_times=1, loss_scale: bool = False, 
    skip_step: int = 64,
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
        timestep = torch.ones((batch_size, input_ids.size(1)), device=device) * 0
        timestep[:, -resp_seqlen:] = 1 - i / steps
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = model(
                input_ids,
                attention_bias=attn_bias,
                position_ids=position_ids,
                timestep=timestep,
                mask_index=(input_ids == mask_id),
            )
        # select xt
        logits = out.logits.to(torch.float32)[:, -resp_seqlen:]
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        # (B, L)
        token_logprob = log_probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        upm_logits = out.upm_logits.to(torch.float32)[:, -resp_seqlen:]
        upm_logits = upm_logits.view(batch_size, -1, block_size)
        upm_prob = F.softmax(upm_logits, dim=-1)
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
        ).sum() / scaler_logits
        # torch.distributed.breakpoint() # torch.save(inputs, 'inputs_vars.pt')
        loss_top_k  = - torch.min(
            pt_logprob_ratio * advantages.view(-1, 1),
            pt_logprob_ratio_cliped * advantages.view(-1, 1),
        ).sum() / scaler_top_k
        
        loss = loss_logits + loss_top_k
        # normalize loss by denoising steps & token length
        if accelerator is not None:
            accelerator.backward(loss.mul(gain).sum())
        else:
            loss.mul(gain).sum().backward()
        loss_to_log = loss_to_log + loss.mul(gain ).detach().mean().item()

    all_rewards = accelerator.gather(inputs['rewards'].detach())
    ds1_rwd = all_rewards[0::2]
    ds2_rwd = all_rewards[1::2]
    reward_mean = all_rewards.mean().item()
    reward_std = all_rewards.std().item()

    eos_id = 126081
    length = ((response_tokens != eos_id).sum(-1).to(device).float() - prompt_len)

    return {
        "reward": reward_mean,
        "length": length.mean().item(),
        "length_std": length.std().item(),
        'max_length': length.max().item(),
        "valid_samples": valid_samples,
        "num_pred_tokens_per_sample": total_pred_tokens / valid_samples * repeat_times,
        "reward_std": reward_std,
        "ds1_rwd": ds1_rwd.mean().item(),
        "ds2_rwd": ds2_rwd.mean().item(),
        "ds1_rwd_std": ds1_rwd.std().item(),
        "ds2_rwd_std": ds2_rwd.std().item(),
    }



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
):
    """
    Generate block-wise responses using block diffusion. KV Cache is verified.
    """
    # process batch 
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
    timestep = torch.ones((batch_size, block_size), device=x_t.device) 
    # return lists
    first_kvcache = deepcopy(kv_cache)
    sample_orders = []
    trajectory_inputs = []
    trajectory_outputs = []
    ranking_logprob_list = []
    unmask_index_list = []
    token_logprob_list = []
    is_valid_steps = []
    token_entropy_list = {} # 不同 step 剩下所有的 mask token 预测的 entropy 差异
    upm_entropy_list = {}

    remask_times = torch.zeros((batch_size,), device=x_t.device, dtype=torch.float32)

    # for visualization
    if viz:
        cur_response = x_t
        visualize_xt = [x_t]
        visualize_x0 = [x_t]
    else:
        visualize_xt = None
        visualize_x0 = None

    total_blocks = 0

    
    while (cur_blocks * block_size) <= max_length:
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
            trajectory_inputs.append(x_t.clone())
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )
            logits = out.logits.to(torch.float32)
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            entropy = - (log_probs * log_probs.exp()).sum(dim=-1)
            entropy = torch.where(mask_index, entropy, 0)
            token_entropy_list.setdefault(i, []).append((entropy.sum() / mask_index.sum()).item())
            
            upm_logits = out.confidences.to(torch.float32)
            upm_prob = F.softmax(upm_logits, dim=-1)
            upm_entropy = - (upm_prob * upm_prob.log()).sum(dim=-1)
            upm_entropy_list.setdefault(i, []).append(upm_entropy.mean().item())
            
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
            
            remask_times += (remask_index & ~ mask_index & ~is_eos_meet.unsqueeze(1)).float().sum(dim=-1)

        if viz:
            cur_response = torch.cat([cur_response, x_t], dim=1)
        responses.append(x_t.clone())
        cur_blocks += 1
        # stop condition 
        
        is_eos_meet = is_eos_meet | (x_t == eos_id).any(dim=-1)
        # gather all is_eos_meet
        tensor_list = [torch.zeros_like(is_eos_meet) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list, is_eos_meet)
        is_eos_meet_all_rank = torch.cat(tensor_list, dim=0)
        total_blocks += (~is_eos_meet_all_rank).float().sum()
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
    #rewards = reward_fn(batch, responses, num_generations, device).float() if reward_fn is not None else torch.zeros(batch_size, device=device)

    return {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'sample_orders': sample_orders,
        'unmask_index_list': unmask_index_list,
        'ranking_logprob_list': ranking_logprob_list,
        'is_valid_steps': is_valid_steps,
        'responses': responses,
        'response_tokens': response_tokens,
        #'rewards': rewards,
        'prompt_len': prompt_len,
        'token_logprob_list': token_logprob_list,
        'first_kvcache': first_kvcache,

        'block_size': block_size,
        'max_length': max_length,
        'steps': steps,
        'remask_times': remask_times,

        'visualize_x0': visualize_x0,
        'visualize_xt': visualize_xt,

        'total_blocks': total_blocks,

        'token_entropy_list': token_entropy_list,
        'upm_entropy_list': upm_entropy_list,
        'responses_length': responses_length,
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
        timestep = torch.ones((batch_size, input_ids.size(1)), device=device) * 0
        timestep[:, -resp_seqlen:] = 1 - i / steps
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
        upm_prob = F.softmax(upm_logits, dim=-1)
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
        ).sum() / scaler_logits
        # torch.distributed.breakpoint() # torch.save(inputs, 'inputs_vars.pt')
        loss_top_k  = - torch.min(
            pt_logprob_ratio * advantages.view(-1, 1),
            pt_logprob_ratio_cliped * advantages.view(-1, 1),
        ).sum() / scaler_top_k
        
        loss = loss_logits + loss_top_k
        # normalize loss by denoising steps & token length
        if accelerator is not None:
            accelerator.backward(loss.mul(gain).sum())
        else:
            loss.mul(gain).sum().backward()
        loss_to_log = loss_to_log + loss.mul(gain ).detach().mean().item()

    all_rewards = accelerator.gather(inputs['rewards'].detach())
    ds1_rwd = all_rewards[0::2]
    ds2_rwd = all_rewards[1::2]
    reward_mean = all_rewards.mean().item()
    reward_std = all_rewards.std().item()

    eos_id = 126081
    length = ((response_tokens != eos_id).sum(-1).to(device).float() - prompt_len)

    return {
        "reward": reward_mean,
        "length": length.mean().item(),
        "length_std": length.std().item(),
        'max_length': length.max().item(),
        "valid_samples": valid_samples,
        "num_pred_tokens_per_sample": total_pred_tokens / valid_samples * repeat_times,
        "reward_std": reward_std,
        "ds1_rwd": ds1_rwd.mean().item(),
        "ds2_rwd": ds2_rwd.mean().item(),
        "ds1_rwd_std": ds1_rwd.std().item(),
        "ds2_rwd_std": ds2_rwd.std().item(),
    }