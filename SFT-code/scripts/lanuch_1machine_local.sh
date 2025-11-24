#!/bin/bash


export PROC_PER_NODES=8
export BETTER_EXCEPTIONS=1


accelerate \
launch --config_file config/accelerate/fsdp_origin_1machine_bitower.yaml \
train.py --config config/remask_sft/mdm_remask_sft_bitower_3stage_s2_4machine_binary_v2_1.yaml
