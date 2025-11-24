"""Main training loop."""
'''
FSDP将模型的参数、梯度和优化器状态分片到多个GPU上,每个GPU仅存储和处理部分参数,而非完整副本。
这与传统数据并行(如DDP)中每个GPU保存完整模型的方式不同, 从而大幅减少显存需求
'''

import os
import time
import psutil
import numpy as np
import torch
import torch.distributed
import dnnlib
import copy
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch_utils import distributed as dist
from torch_utils import misc
import functools
import json


#----------------------------------------------------------------------------
def retry_on_oserror(func):
    """
    一个装饰器，当被装饰的函数抛出 OSError 时，会等待 30 秒后重试。
    这个过程会无限重复，直到函数成功执行。
    """
    # functools.wraps 会保留原函数的元信息(如函数名、文档字符串等)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                # 尝试执行原函数，并返回其结果
                return func(*args, **kwargs)
            except OSError as e:
                # 捕获到 OSError，打印提示信息并等待
                print(f"Error occurred: {e}. Retry in 30s ...")
                time.sleep(30)
            # 如果成功，循环会通过 return 退出；如果失败，则会 sleep 后继续下一次循环
    return wrapper

def training_loop(
    run_dir             = '.',      # Output directory.
    batch_size          = 512,
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    total_steps         = 200000,   # Training duration, measured in thousands of training images.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    step_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?#当设置为 True 时，cuDNN 会在首次运行时对当前硬件和输入尺寸进行基准测试（benchmark），为每个卷积层选择最快的计算算法。后续运行将复用该算法，避免重复搜索，从而加速计算
    device              = torch.device('cuda'),
    grad_accumulation   = 1,
    lr_scheduler_kwargs = {},
    precision           = "fp16",
    resume_pt           = None,
    resume_state_dump   = None,
    resume_step         = 0,
    max_grad_norm       = 1000,
    val_ticks           = 5,
    skip_spike_grad     = 10e10,
    tokenizer_kwargs    = {},
    loss_kwargs         = {},
    activation_checkpointing = 'whole_layer',
    training_state_dir  = None,
    start_step = 0,
    *args, **kwargs
):
    accelerator = dist.get_accelerator()
    dist.print0(f"Useless parameters: \n {args}\n {kwargs}")
    opts = {
        "batch_size": batch_size,
        "data_loader_kwargs": data_loader_kwargs,
        "network_kwargs": network_kwargs,
        "optimizer_kwargs": optimizer_kwargs,
        "seed": seed,
        "total_steps": total_steps,
        "loss_scaling": loss_scaling,
        "step_per_tick": step_per_tick,
        "snapshot_ticks": snapshot_ticks,
        "state_dump_ticks": state_dump_ticks,
        "grad_accumulation": grad_accumulation,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "precision": precision,
        "resume_pt": resume_pt,
        "resume_state_dump": resume_state_dump,
        "resume_step": resume_step,
        "max_grad_norm": max_grad_norm,
        "val_ticks": val_ticks,
        "skip_spike_grad": skip_spike_grad,
        "activation_checkpointing": activation_checkpointing,
        "start_step":start_step,
    }
    # Initialize.
    # dist.print0("Setting anomaly mode TRUE, training will be slower")
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    rank = dist.get_rank()
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + rank) % (1 << 31))#通过 seed * world_size + rank 生成唯一种子，确保不同进程的随机数序列不同，避免数据重复或同步问题。1 << 31 是 2³¹（2147483648），限制种子范围为 32 位有符号整数。
    torch.manual_seed(np.random.randint(1 << 31))#基于 NumPy 的随机数进一步设置 PyTorch 的随机种子，保证模型初始化、数据增强等操作的随机性可控
    torch.backends.cudnn.benchmark = cudnn_benchmark
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    
    #torch.distributed.breakpoint()
    curr_dir = copy.deepcopy(run_dir)
    curr_dir = os.path.dirname(curr_dir)
    print(curr_dir)
    print(start_step)
    #assert os.path.exists(curr_dir)

    # Load dataset.
    dist.print0('Loading dataset...')
    dataloader_iterator = dnnlib.util.construct_class_by_name(**data_loader_kwargs)
    
    accelerator.wait_for_everyone()
    # Construct network.
    dist.print0('Constructing network...')
    model = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    # model.train().requires_grad_(True).to(device)
    model.eval().to(device)
    model_params = misc.count_parameters(model)
    
    model.model.set_activation_checkpointing(activation_checkpointing)
    

    # tokenizer
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    if 'gpt2' in tokenizer_kwargs.get('pretrained_model_name_or_path', ''):
        dist.print0("Adding <MASK> token to the tokenizer.")
        mask_token = "<MASK>"
        tokenizer.add_tokens([mask_token])
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'llada' in tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower():
        dist.print0("Setting pad_token_id to mask_token_id for LLaDA.")
        tokenizer.pad_token_id = 126336

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) 
    # optimizer = dnnlib.util.construct_class_by_name(
    #     params=[p for p in model.parameters() if p.requires_grad],
    #     **optimizer_kwargs
    # )
    if optimizer_kwargs.get('lr2', None) is not None:
        base_model = [p for p in model.model.transformer.parameters() if p.requires_grad]
        old_params = {
            'params': base_model,
            'lr': optimizer_kwargs.get('lr2')
        }
        new_params = {
            'params': [p for p in model.parameters() if p.requires_grad and all(p is not p2 for p2 in base_model)],
            'lr': optimizer_kwargs.get('lr')
        }
        optimizer_kwargs.pop('lr2')
        optimizer_kwargs.pop('lr')
        params = [old_params, new_params]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = dnnlib.util.construct_class_by_name(
        params=params,
        **optimizer_kwargs
    )
    
    # Setup LR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=dnnlib.util.construct_class_by_name(**lr_scheduler_kwargs))
    
    
    assert accelerator is not None
    model, optimizer, dataloader_iterator, scheduler = accelerator.prepare(
       model, optimizer, dataloader_iterator, scheduler
    )
    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        dist.print0(f"Resume from {resume_state_dump}")
        accelerator.load_state(resume_state_dump)
    
    
    dataloader_iterator = iter(dataloader_iterator)
    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        accelerator.print(f"Resume from step {resume_step}, skipping training data ...")
        for i in range(resume_step):
            next(dataloader_iterator)

    
    # Train.
    cur_tick = resume_step
    cur_nsamples = 0
    training_step = resume_step # 0 for default
    tick_start_step = training_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    
    # dist.print0("parameters Required grad:")
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         dist.print0(name, p.shape)
    
    # tensorboard 
    if rank == 0:
        wandb.init(
            entity='fetchniches',
            project="rl-discrete-diffusion", 
            name=':'.join(run_dir.split('/')[-2:]),
            dir=run_dir, 
            config=opts,
            mode='offline'
        )
        eval_dir = os.path.join(run_dir, 'evaluations')
        os.makedirs(eval_dir, exist_ok=True)
        text_table = wandb.Table(
            columns=['step', 'prompt', 'response'], 
        )
        
    dist.print0(f'Training for {total_steps} steps in {precision_dtype}...')
    dist.print0(f"Model with Param: {model_params}")
    dist.print0()

    batch_total = batch_size * dist.get_world_size() * grad_accumulation
    
    torch.distributed.barrier()
    while True:
        training_step += 1

        if rank == 0 and not os.path.exists(run_dir): 
            raise SystemError(f'Run directory "{run_dir}" does not exist.')
        
        optimizer.zero_grad(set_to_none=True)

        all_loss_los_kwargs = []
        for round_idx in range(grad_accumulation):
        # generate data and score the completions
            with misc.ddp_sync(model, sync=round_idx == grad_accumulation - 1):
                with torch.autocast(device_type="cuda", enabled=True, dtype=precision_dtype):
                    batch = next(dataloader_iterator)
                    loss_dict = loss_fn(model, batch, device=device)
                    loss = loss_dict['loss']

                    # normalised loss 
                    loss = loss.sum().mul(loss_scaling / (grad_accumulation*batch_size) )
                    accelerator.backward(loss)

                    all_loss_los_kwargs.append({
                        "loss": loss.item(),
                        'nll': loss_dict['nll'],
                        'upm_loss': loss_dict['upm_loss'],
                    })
        # torch.
        loss_log_kwargs = {
            k: sum([item[k] for item in all_loss_los_kwargs]) / len(all_loss_los_kwargs) for k in all_loss_los_kwargs[0]
        }

        # maintenance
        # for param in model.parameters():
        #     if param.grad is not None:
        #         torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

        _grad_norm = accelerator.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        grad_norm = model.get_global_grad_norm() if hasattr(model, "get_global_grad_norm") else _grad_norm
        # In some cases the grad norm may not return a float
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        
        scheduler.step(training_step)
        optimizer.step()

        if rank == 0:
            wandb.log({
                'lr': scheduler.get_lr()[0],
                'grad_norm': grad_norm,
                **loss_log_kwargs
            }, step=training_step)

        cur_nsamples += batch_total
        done = (training_step >= total_steps)
        accelerator.wait_for_everyone()
        # Perform maintenance tasks once per tick.
        if (not done) and (cur_tick != 0) and (training_step < tick_start_step + step_per_tick):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        
        fields = {
            'tick': cur_tick,
            'step': training_step,
            'time': dnnlib.util.format_time(tick_end_time - start_time),
            'sec-per-tick': f"{(tick_end_time - tick_start_time):<7.1f}",
            'sec-per-samples': f"{((tick_end_time - tick_start_time) / cur_nsamples):<7.2f}",
            'maintenance': f"{maintenance_time:<6.1f}",
            'resource/cpumem': f"{(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}",
            'resource/gpumem': f"{(torch.cuda.memory_allocated(device) / 2**30):<6.2f}",
            'resource/reserved': f"{(torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            'grad_norm': grad_norm,
            'lr': scheduler.get_lr()[0],
            **loss_log_kwargs,
        }


        torch.cuda.reset_peak_memory_stats()
        for key, value in fields.items():
            dist.print0(f"{key} {value}", end='\t')
        dist.print0()

        # Update logs.
        if rank == 0:
            # delete useless fields
            fields.pop('tick'); fields.pop('step'); fields.pop('time'); fields.pop('reward', None); fields.pop('gsm_reward', None); fields.pop('math_reward', None); fields.pop('loss', None)
            # convert string to float
            wandb.log({k: float(v) for k, v in fields.items()}, step=training_step)
        

        if cur_tick % snapshot_ticks == 0 and cur_tick != 0:
            @retry_on_oserror
            def save_ckpt():
                state_dict = accelerator.get_state_dict(model)
                save_path = os.path.join(training_state_dir, f'training-state-{training_step:06d}')
                accelerator.save_state(save_path)

                if rank == 0:
                    save_path = os.path.join(run_dir, f'ckpt-{training_step:06d}')
                    accelerator.unwrap_model(model).save_pretrained(
                        save_path, state_dict=state_dict, safe_serialization=True
                    )
            save_ckpt()
        accelerator.wait_for_everyone()

        # Update state.
        cur_tick += 1
        cur_nsamples = 0
        tick_start_step = training_step
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
       
        if done:
            save_ckpt()
            accelerator.wait_for_everyone()
            break
    if rank == 0:
        wandb.log({
            'text_response': text_table,
        })
        
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
