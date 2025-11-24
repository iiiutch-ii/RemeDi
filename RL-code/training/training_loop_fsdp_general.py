"""Main training loop."""

import os
import time
from typing import Sequence
import psutil
import numpy as np
import torch
import dnnlib
import wandb
import json 
from torch.optim.lr_scheduler import LambdaLR
from torch_utils import distributed as dist
from torch_utils import misc
from training.utils.organizer import reorganize_input_chunks, split_input_chunks_local
from training.utils.replay_buffer import ReplayBuffer
from general_data.general import base_url
import requests
from training.utils.surgeon import TrainingSurgeon


def check_enough_storage(rund_dir, required_space: float = 150):
    """
    required_space: GB
    """
    while True:
        free_space = psutil.disk_usage(rund_dir).free / 2**30
        if free_space < required_space:
            dist.print0(f"Not enough storage. Required: {required_space}GB, Free: {free_space}GB")
            time.sleep(60)
        else:
            break

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    batch_size          = 512,
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    data_loader2_kwargs = {},
    network_kwargs      = {},       # Options for model and preconditioning.
    ref_network_kwargs  = {},       # Options for ref model and preconditioning.
    loss_kwargs         = {},
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    total_steps         = 200000,   # Training duration, measured in thousands of training images.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    step_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
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
    infer_kwargs        = {},
    tokenizer_kwargs    = {},
    activation_checkpointing = 'whole_layer',
    training_state_dir  = None,
    use_filter_out: bool = False,
    use_batch_std: bool = False,
    use_optimal_baseline: bool = False,
    enable_surgeon: bool = False,
    record_every: int = 10000000,
    do_organize = False,
    buffer_size = 0,
    fwd_repeat_times: int = None,
    use_fsdp2: bool = False,
    not_resume_optimizer: bool = False,
    *args, **kwargs
):
    opts = {
        "batch_size": batch_size,
        "data_loader_kwargs": data_loader_kwargs,
        "network_kwargs": network_kwargs,
        "loss_kwargs": loss_kwargs,
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
        "infer_kwargs": infer_kwargs,
        "activation_checkpointing": activation_checkpointing,
        "do_organize": do_organize,
        "buffer_size": buffer_size,
    }
    # Initialize.
    accelerator = dist.get_accelerator()
    # if debugging:
    # torch.autograd.set_detect_anomaly(True)
    accelerator.print(f"Useless parameters: \n {args}\n {kwargs}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    
    # Check reward model
    # check if reward model is serving:
    # accelerator.print("Checking reward model...")
    # try:
    #     # 发送一个简单的 GET 请求测试连通性
    #     response = requests.get(base_url)
    #     accelerator.print(f"State Code: {response.status_code}")
    # except requests.exceptions.ConnectionError:
    #     raise Exception("Cannot connect to reward model, please check if the service is running")
    # except Exception as e:
    #     raise Exception(f"Error: {e}")


    # Load dataset.
    accelerator.print('Loading dataset...')

    dataloader_iterator = dnnlib.util.construct_class_by_name(
        **data_loader_kwargs) # , rank=rank, num_replicas=world_size
    
    # Construct network.
    accelerator.print('Constructing network...')
    model = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    # model.train().requires_grad_(True).to(device)
    model.eval().requires_grad_(True).to(device)
    model_params = misc.count_trainable_parameters(model)
    
    model.model.set_activation_checkpointing(activation_checkpointing)

    if loss_kwargs.get('llada_kl_coef') is not None:
        accelerator.print('Creating reference model for KL loss...')
        ref_model = dnnlib.util.construct_class_by_name(**ref_network_kwargs)
        ref_model.requires_grad_(False).eval().to(device)
    else:
        ref_model = None

    # tokenizer
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    if 'gpt2' in tokenizer_kwargs.get('pretrained_model_name_or_path', ''):
        accelerator.print("Adding <MASK> token to the tokenizer.")
        mask_token = "<MASK>"
        tokenizer.add_tokens([mask_token])
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'llada' in tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower():
        accelerator.print("Setting pad_token_id to mask_token_id for LLaDA.")
        tokenizer.pad_token_id = 126336

    # Setup optimizer.
    accelerator.print('Setting up optimizer...')
    if optimizer_kwargs.get('lr2', None) is not None:
        base_model = [p for p in model.model.parameters() if p.requires_grad]
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
    # model, optimizer, dataloader_iterator, scheduler = accelerator.prepare(
    #    model, optimizer, dataloader_iterator, scheduler
    # )
    model, optimizer, scheduler = accelerator.prepare(
       model, optimizer, scheduler
    )

    # model = torch.compile(model, mode="reduce-overhead")

    if ref_model is not None:
        ref_model = accelerator.prepare(ref_model)

    if ref_model in accelerator._models:
        accelerator._models.remove(ref_model)

    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        if not_resume_optimizer:
            accelerator.print("Not resuming optimizer state")
            original_optimizer_state = optimizer.state_dict()
        accelerator.print(f"Resume from {resume_state_dump}")
        accelerator.load_state(resume_state_dump)
        if not_resume_optimizer:
            accelerator.print("Resuming optimizer state back to random state")
            optimizer.load_state_dict(original_optimizer_state)
    
    # reset lr in optimizer and scheduler
    accelerator.print(f"Resetting lr to {optimizer_kwargs.get('lr')}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = optimizer_kwargs.get('lr')
    scheduler.scheduler.base_lrs = [optimizer_kwargs.get('lr') for _ in scheduler.scheduler.base_lrs]


    dataloader_iterator = iter(dataloader_iterator)
    replay_buffer = ReplayBuffer(buffer_size=buffer_size)
    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        accelerator.print(f"Resume from step {resume_step}, skipping training data ...")
        for _ in range(resume_step):
            next(dataloader_iterator)
        
    # Train.
    cur_tick = resume_step
    cur_nsamples = 0
    training_step = resume_step # 0 for default
    tick_start_step = training_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    
    # tensorboard 
    eval_dir = os.path.join(run_dir, 'evaluations')
    obj_dir = os.path.join(run_dir, 'stat_objs')
    surgeon_dir = os.path.join(run_dir, 'surgeon')
    ds_dir = os.path.join(run_dir, 'ds_snapshot')
    if rank == 0:
        wandb.init(
            entity='fetchniches',
            project="rl-discrete-diffusion", 
            name=':'.join(run_dir.split('/')[-2:]),
            dir=run_dir, 
            config=opts,
            mode='offline'
        )
        
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(ds_dir, exist_ok=True)
        if enable_surgeon:
            os.makedirs(surgeon_dir, exist_ok=True)
        text_table = wandb.Table(
            columns=['step', 'prompt', 'response'], 
        )
        
    accelerator.print(f'Training for {total_steps} steps in {precision_dtype}...')
    accelerator.print(f"Model with Trainable Param: {model_params}")
    accelerator.print()

    batch_total = batch_size * dist.get_world_size() * grad_accumulation * infer_kwargs['num_generations']
    # preference: 0, math: 1, code: 2
    prompt_type_dict = {
        'preference': 0,
        'gsm8k': 1,
        'general-math': 1,
        'taco': 2,
        'kodcode': 2,
        'leetcode': 2,
        'rstar': 2,
        'stem': 3,
        'ifeval': 4,
        'scp': 3,
        'mmlu': 5,
        'arc': 6,

    }
    index_list = [os.path.basename(fn) for fn in os.listdir(data_loader_kwargs['index_dir']) if fn.endswith('.index')]
    index_list.sort()
    ds_name_list = [''.join(fn.split('.')[:-1]) for fn in index_list]
    
    while True:
        if rank == 0 and not os.path.exists(run_dir): 
            raise SystemError(f'Run directory "{run_dir}" does not exist.')
        optimizer.zero_grad(set_to_none=True)
        replay_buffer.reset_flag()
        all_loss_los_kwargs = []
        accelerator.wait_for_everyone() 
        with TrainingSurgeon(
            model, 
            enabled=False, # enable_surgeon,
            output_dir=surgeon_dir,
            record_every=record_every,
            save_memory=False,
            ) as surgeon:
            
            for round_idx in range(grad_accumulation):
            # generate data and score the completions
                # with misc.ddp_sync(model, sync=round_idx == grad_accumulation - 1):
                model.eval()
                accelerator.print("Start Sampling...")
                if rank == 0: 
                    t1 = time.time()
                batch = next(dataloader_iterator)
                # ds_name = os.path.join(ds_dir, f's{training_step}_ga{round_idx}_rank{rank}.json')
                # with open(ds_name, 'w') as f:
                #     json.dump(batch, f)

                inputs_chunks = []
                if use_fsdp2:
                    model.set_reshard_after_forward(False)
                
                for _ in range(0, infer_kwargs['fwd_repeat_times']):
                    inputs = dnnlib.util.call_func_by_name(
                        **infer_kwargs,
                        model=model, # accelerator.unwrap_model(model),
                        tokenizer=tokenizer,
                        batch=batch,
                        reward_fn=None,
                        device=device,
                    )
                    inputs_chunks.append(inputs)

                if use_fsdp2:
                    for layer in model.model.transformer['blocks']:
                        layer.reshard()
                # regroup inputs
                inputs_chunks = split_input_chunks_local(inputs_chunks, infer_kwargs['fwd_num_generations'], infer_kwargs['fwd_num_generations'] // infer_kwargs['num_generations'])

                accelerator.wait_for_everyone() 
                if rank == 0:
                    print(f"Sampling time: {time.time() - t1:.2f}s ") # (max gen-len: {max(inputs['responses_length'])}; total-len: {inputs['response_tokens'].size(1)})
                # gather inputs to get advantages and valid_samples

                rewards_list = [inputs['rewards'] for inputs in inputs_chunks]
                rewards = torch.cat(rewards_list, dim=0)
                # length filter
            
                rewards_mean = (rewards.view(infer_kwargs['num_generations']*infer_kwargs['repeat_times'], -1).mean(dim=0)).repeat(infer_kwargs['num_generations']*infer_kwargs['repeat_times'],)
                rewards_std = rewards.view(infer_kwargs['num_generations']*infer_kwargs['repeat_times'], -1).std(dim=0).repeat(infer_kwargs['num_generations']*infer_kwargs['repeat_times'],)
                if use_optimal_baseline:
                    sample_block_length = [inputs['sample_block_length'] for inputs in inputs_chunks]
                    sample_block_length = torch.cat(sample_block_length, dim=0)
                    weighted_rewards = rewards * sample_block_length
                    rewards_mean = weighted_rewards.view(infer_kwargs['num_generations']*infer_kwargs['repeat_times'], -1).mean(dim=0).repeat(infer_kwargs['num_generations']*infer_kwargs['repeat_times'],)
                
                # batch-std
                if use_batch_std: 
                    cur_problem_type = torch.tensor(prompt_type_dict[batch['reward_tag']], device=device)
                    rewards_type = accelerator.gather(cur_problem_type)
                    rewards_all = accelerator.gather(rewards.unsqueeze(dim=0))
                    rewards_std = rewards_all[rewards_type == cur_problem_type.item()].std()

                # reward to log
                ds_name = os.path.basename(batch['sample_metadata'][0]).split('.')[0]
                ds_id = batch['sample_metadata'][2]
                all_rewards = accelerator.gather(rewards.unsqueeze(dim=0))
                all_ds_id = accelerator.gather(torch.tensor([ds_id], device=device))


                # 将收集到的数据按数据集名称分组
                reward_dict = {}
                for ds_id, rwd in zip(all_ds_id, all_rewards):
                    ds_name = ds_name_list[ds_id]
                    if f'reward_{ds_name}' not in reward_dict:
                        reward_dict[f'reward_{ds_name}'] = []
                    reward_dict[f'reward_{ds_name}'].append(rwd.mean().item())
                
                # 计算每个数据集的平均奖励
                for key in reward_dict:
                    rwd_mean = sum(reward_dict[key]) / len(reward_dict[key])
                    reward_dict[key] = rwd_mean

                advantages = (rewards - rewards_mean) / (rewards_std + 1e-4)

                # save dataname and advantages
                save_path = os.path.join(ds_dir, f's{training_step}_ga{round_idx}_rank{rank}.json')
                with open(save_path, 'w') as f:
                    json.dump({
                        'ds_name': ds_name,
                        'advantages': advantages.tolist(),
                        'reward_std': rewards_std.tolist(),
                        'prompt': batch['prompts'][0],
                        # **batch
                    }, f, indent=4)
                
                # advantages shaping
                valid_samples = (advantages != 0).sum()

                
                split_advantages = advantages.split(infer_kwargs['num_generations'], dim=0) 
                for chunk, adv in zip(inputs_chunks, split_advantages):
                    chunk["advantages"] = adv
                
                samples = inputs_chunks[0]['response_tokens']

                if valid_samples > 0 and buffer_size > 0:
                    replay_buffer.save_in_buffer(inputs_chunks)
                elif replay_buffer.is_ready(len(inputs_chunks)):
                    inputs_chunks = replay_buffer.sample_from_buffer(len(inputs_chunks), device)
                    valid_samples = torch.cat([item["advantages"] != 0 for item in inputs_chunks]).sum()
                accelerator.wait_for_everyone()
                # organize the input chunks
                if do_organize:
                    inputs_chunks = reorganize_input_chunks(
                        inputs_chunks, 
                        accelerator, 
                        pad_fields={
                            "trajectory_inputs": 0, 
                            "trajectory_outputs": 0,
                            "attention_mask": 0
                        }, 
                        sample_pad_field="attention_mask",
                    )
                # return rewards too
                # fsdp_model.train()
                accelerator.print("Start Loss Calcing...")
                if rank == 0:
                    t1 = time.time()
                model.train()
                if use_fsdp2:
                    model.set_reshard_after_forward(True)
                
                for loss_idx, inputs in enumerate(inputs_chunks):
                    sync_cond = round_idx == grad_accumulation - 1 and loss_idx == len(inputs_chunks) - 1
                    # with misc.ddp_sync(model, sync=sync_cond):
                    loss_log_kwargs = dnnlib.util.call_func_by_name(
                        **loss_kwargs, 
                        model=model, 
                        ref_model=ref_model,
                        inputs=inputs, 
                        gain=loss_scaling, 
                        accelerator=accelerator, 
                        valid_samples=valid_samples, 
                        repeat_times=infer_kwargs['repeat_times'],
                        sync_cond=sync_cond,
                    )
                    surgeon.step()
                    all_loss_los_kwargs.append(loss_log_kwargs)
                if rank == 0:
                    print(f"Loss Calculation time: {time.time() - t1:.2f}s")
                accelerator.wait_for_everyone()

            # ---------- # 
            # validation # 
            # ---------- # 
            check_enough_storage(run_dir, 1)
            if cur_tick % val_ticks == 0:
                # save the inputs
                source = os.path.basename(batch['sample_metadata'][0]).split('.')[0]
                if "answers" in batch:
                    text_inputs = f'source: {source}\n' + batch['prompts'][0] + '\n\n[answer]: ' + batch['answers'][0]
                else:
                    text_inputs = f'source: {source}\n' + batch['prompts'][0] + '\n\n[no answer]'
                text_responses = ("\n" + "*" * 20 + "\n").join(
                    tokenizer.batch_decode(samples, skip_special_tokens=False)
                )
                if rank == 0:
                    text_table.add_data(str(training_step), text_inputs, text_responses)

                with open(os.path.join(eval_dir, f'evaluate_step{training_step}_rk{rank}.txt'), 'w') as f:
                    f.write(text_inputs + '\n' + '=' * 20 + '\n' + text_responses)
            # sync
            accelerator.wait_for_everyone()

            for key in list(inputs.keys()):
                del inputs[key]

        all_loss_los_kwargs = [item for item in all_loss_los_kwargs if item is not None]
        if len(all_loss_los_kwargs) == 0:
            accelerator.print(f"All loss kwargs are None, skip this step {training_step}")
            continue
        loss_log_kwargs = {
            k: sum([item[k] for item in all_loss_los_kwargs]) / len(all_loss_los_kwargs) for k in all_loss_los_kwargs[0]
        }
        # for key in ['ds1_rwd', 'ds2_rwd', 'ds1_rwd_std', 'ds2_rwd_std']:
        #     loss_log_kwargs.pop(key)
        
        # maintenance
        contains_nan = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    contains_nan = True
                    break
        
        if not contains_nan:
            _grad_norm = accelerator.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
            grad_norm = model.get_global_grad_norm() if hasattr(model, "get_global_grad_norm") else _grad_norm
            # In some cases the grad norm may not return a float
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
            # check
            # model.mask_linear.weight.grad
            scheduler.step(training_step)
            optimizer.step()
        else:
            accelerator.print(f"Contains nan, skip this step {training_step}")

        if rank == 0:
            wandb.log({
                'lr': optimizer.param_groups[0]['lr'], # scheduler.get_lr()[0],
                'grad_norm': grad_norm,
                **loss_log_kwargs,
                **reward_dict
                # 'remask_times': ['remask_times']
            }, step=training_step)

        cur_nsamples += batch_total
        done = (training_step >= total_steps)
        training_step += 1
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
            **reward_dict,
        }
        
        torch.cuda.reset_peak_memory_stats()
        for key, value in fields.items():
            accelerator.print(f"{key} {value}", end='\t')
        accelerator.print()

        # Update logs.
        if rank == 0:
            # delete useless fields
            fields.pop('tick'); fields.pop('step'); fields.pop('time'); fields.pop('reward', None); fields.pop('gsm_reward', None); fields.pop('math_reward', None); fields.pop('loss', None)
            # convert string to float
            wandb.log({k: float(v) for k, v in fields.items()}, step=training_step)
        

        if cur_tick % snapshot_ticks == 0:
            check_enough_storage(run_dir, 150)
            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model)
            save_path = os.path.join(training_state_dir, f'training-state-{training_step:06d}')
            accelerator.save_state(save_path)

            save_path = os.path.join(training_state_dir, f'replay-buffer-{training_step:06d}')
            replay_buffer.save_checkpoint(save_path)

            if dist.get_rank() == 0:
                save_path = os.path.join(run_dir, f'ckpt-{training_step:06d}')
                accelerator.unwrap_model(model).save_pretrained(
                    save_path, state_dict=state_dict, safe_serialization=True
                )
        accelerator.wait_for_everyone()

        # Update state.
        cur_tick += 1
        cur_nsamples = 0
        tick_start_step = training_step
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
       
        if done:
            break
    if rank == 0:
        wandb.log({
            'text_response': text_table,
        })
        
    # Done.
    accelerator.print()
    accelerator.print('Exiting...')

#----------------------------------------------------------------------------
