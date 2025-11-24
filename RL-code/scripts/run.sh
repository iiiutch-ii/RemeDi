#!/bin/bash
ACCELERATE_CONFIG=config/accelerate/fsdp2_towel_single.yaml
export TOKENIZERS_PARALLELISM=true  

accelerate launch \
--config_file $ACCELERATE_CONFIG --num_machines 1 --num_processes 8 train.py \
--config config/remedi.yaml