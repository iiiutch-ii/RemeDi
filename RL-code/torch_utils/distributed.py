# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
import random 
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
#----------------------------------------------------------------------------

ACCELERATOR = None

def init():
    global ACCELERATOR
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ACCELERATOR = Accelerator(kwargs_handlers=[timeout])

#----------------------------------------------------------------------------

def get_accelerator():
    return ACCELERATOR

def get_rank():
    if ACCELERATOR is None:
        return 0
    return ACCELERATOR.process_index

#----------------------------------------------------------------------------

def get_local_rank():
    if ACCELERATOR is None:
        return 0
    return ACCELERATOR.local_process_index

#----------------------------------------------------------------------------

def get_world_size():
    if ACCELERATOR is None:
        return 1
    return ACCELERATOR.num_processes

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    init()
    a = torch.zeros(3, device="cuda") + get_rank()
    aa = ACCELERATOR.gather(a)
    print(aa)
