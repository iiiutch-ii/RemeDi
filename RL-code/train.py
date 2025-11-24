import os
import re
import json
import yaml
import click
import torch
import dnnlib
from datetime import datetime
from torch_utils import distributed as dist

def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file) as f:
                    config_data = yaml.load(f, Loader=yaml.FullLoader)
                    for key, value in config_data.items():
                        ctx.params[key] = value
            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass

#----------------------------------------------------------------------------

@click.command(cls=CommandWithConfigFile("config"))
@click.option("--config",    help="config file path",    type=click.Path(exists=True))
@click.option('--fsdp', is_flag=True, help='use fsdp')
def main(**kwargs):
    kwargs.pop("config")
    use_fsdp = kwargs.pop("fsdp", False)
    training_args = dnnlib.EasyDict(kwargs.pop("training_args"))
    opts = dnnlib.EasyDict(kwargs)
    # torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0("Distributed initialized.")
    
    try:
        training_args.batch_size = opts.data_loader_kwargs.get('batch_size', 1)
    except Exception:
        dist.print0("Batch size not set?")

    if training_args.get("resume", None) is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(training_args.resume))
        if not match:
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        training_args.resume_pt = os.path.join(os.path.dirname(training_args.resume), f'network-snapshot-{match.group(1)}.pt')
        training_args.resume_step = int(match.group(1))
        training_args.resume_state_dump = training_args.resume
        training_args.pop("resume")

    # Description string.
    dtype_str = training_args.precision
    desc = f'gpus{dist.get_world_size():d}-batch{training_args.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.pop("desc")}'
        
    date = datetime.now().strftime("%Y-%m-%d")
    training_args.run_dir = os.path.join(training_args.run_dir, date)
    
    # Pick output directory.
    # if dist.get_rank() != 0:
    #     training_args.run_dir = None
        
    # else:
    prev_run_dirs = []
    if os.path.isdir(training_args.run_dir):
        prev_run_dirs = [x for x in os.listdir(training_args.run_dir) if os.path.isdir(os.path.join(training_args.run_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id =  max(prev_run_ids, default=-1) + 1
    training_args.run_dir = os.path.join(training_args.run_dir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(training_args.run_dir)
    training_args.training_state_dir = os.path.join(training_args.run_dir, 'states')
    dist.print0(f"Save training_args at {training_args.training_state_dir}")

    # Print options.
    dump_dict = opts.copy()
    dump_dict.update(training_args)
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(dump_dict, indent=2))
    dist.print0()
    dist.print0(f'Slurm Job Id:            {int(os.environ.get("SLURM_JOB_ID", -1))}')
    dist.print0(f'Output directory:        {training_args.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {training_args.batch_size}')
    dist.print0(f'Precision:               {training_args.precision}')
    dist.print0()
    # sync before creating output dir
    torch.distributed.barrier()
    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(training_args.run_dir, exist_ok=True)
        with open(os.path.join(training_args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(dump_dict, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(training_args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    if "func_name" not in training_args:
        dist.print0(f"***Compatible*** No func_name in training_args, use_fsdp: {use_fsdp}")
        if use_fsdp:
            training_args["func_name"] = "training.training_loop_fsdp.training_loop"
        else:
            training_args["func_name"] = "training.training_loop.training_loop"
    dnnlib.util.call_func_by_name(
        **opts,
        **training_args,
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

